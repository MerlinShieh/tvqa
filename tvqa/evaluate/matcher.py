# -*- coding: utf-8 -*-
"""帧区间匹配：检测事件 vs 真值 episodes。

贪心一对一匹配（IoU 降序），指标：
- recall = 匹配真值数 / 真值数；precision = 匹配数 / 检测数
- start_err / end_err（帧）、duration_err（帧）、IoU 均值
- 未匹配真值 → miss；未匹配检测 → false_positive
"""


def _interval_metrics(det, gt):
    """原始区间 IoU（报告口径，不受匹配容忍影响）。"""
    overlap = max(0, min(det["end_frame"], gt["end_frame"]) - max(det["start_frame"], gt["start_frame"]) + 1)
    union = (det["end_frame"] - det["start_frame"] + 1) + (gt["end_frame"] - gt["start_frame"] + 1) - overlap
    iou = overlap / union if union > 0 else 0.0
    return iou


def _expanded_overlap(det, gt, tolerance_frames):
    """真值区间双向扩张 tolerance_frames 后与检出区间的重叠帧数。"""
    gt_start = gt["start_frame"] - tolerance_frames
    gt_end = gt["end_frame"] + tolerance_frames
    return max(0, min(det["end_frame"], gt_end) - max(det["start_frame"], gt_start) + 1)


def match_events(events, gt_episodes, min_iou=0.05, tolerance_frames=0):
    """events: 检测事件 dict 列表（含 start_frame/end_frame/type）。

    匹配命中 = 原始 IoU ≥ min_iou，或（容忍帧数 > 0 时）真值区间双向扩张
    tolerance_frames 后仍有重叠——即「起止差 N 帧内也算命中」（30 帧 ≈ 1 秒）。
    报告的 iou 与 start/end 误差始终按原始区间计算，不受扩张影响。
    返回 {matches, misses, false_positives}。
    """
    candidates = []
    for di, det in enumerate(events):
        for gi, gt in enumerate(gt_episodes):
            iou = _interval_metrics(det, gt)
            if iou >= min_iou or (tolerance_frames > 0
                                  and _expanded_overlap(det, gt, tolerance_frames) > 0):
                candidates.append((iou, di, gi))
    candidates.sort(reverse=True)
    used_det, used_gt, matches = set(), set(), []
    for iou, di, gi in candidates:
        if di in used_det or gi in used_gt:
            continue
        used_det.add(di)
        used_gt.add(gi)
        det, gt = events[di], gt_episodes[gi]
        matches.append({
            "gt": gt, "det": det, "iou": round(iou, 3),
            "start_err_frames": det["start_frame"] - gt["start_frame"],
            "end_err_frames": det["end_frame"] - gt["end_frame"],
            "duration_err_frames": (det["end_frame"] - det["start_frame"] + 1)
            - (gt["end_frame"] - gt["start_frame"] + 1),
            "event_id": det.get("event_id"),
        })
    misses = [gt for gi, gt in enumerate(gt_episodes) if gi not in used_gt]
    false_positives = [det for di, det in enumerate(events) if di not in used_det]
    return {"matches": matches, "misses": misses, "false_positives": false_positives}


STYLE_TO_EVENT = {"frozen_region": "local_freeze", "mosaic": "blocking"}


def score_dataset(spec, events, extra_metrics=None, evaluate_cfg=None):
    """对一个数据集出记分卡。spec: DatasetSpec；events: 该数据集全部事件。

    evaluate_cfg（来自 config 的 evaluate 节）控制评测口径，全部可配置：
    - match_tolerance_frames：匹配容忍帧数（真值区间双向扩张，30≈1 秒）
    - match_min_iou：算作匹配的最小重叠度
    - ignore_events_below_frames：短于该帧数的事件不计入评测（噪声过滤）
    corruption 类数据集按 episode 的 style 限定候选事件类型
    （frozen_region→local_freeze，mosaic→blocking），逐事件匹配后合并。
    """
    evaluate_cfg = evaluate_cfg or {}
    min_iou = float(evaluate_cfg.get("match_min_iou", 0.05))
    tolerance = int(evaluate_cfg.get("match_tolerance_frames", 0))
    min_frames = int(evaluate_cfg.get("ignore_events_below_frames", 0))
    if min_frames > 0:
        events = [ev for ev in events if ev.get("duration_frames", 0) >= min_frames]

    if spec.manifest_type == "corruption" and spec.episodes:
        all_matches, misses, used_event_ids = [], [], set()
        for episode in spec.episodes:
            expected_type = STYLE_TO_EVENT.get(episode.get("style"), spec.expected_event)
            candidates = [ev for ev in events if ev["type"] == expected_type]
            result = match_events(candidates, [episode], min_iou=min_iou, tolerance_frames=tolerance)
            all_matches.extend(result["matches"])
            misses.extend(result["misses"])
            for m in result["matches"]:
                if m.get("event_id"):
                    used_event_ids.add(m["event_id"])
        corruption_types = set(STYLE_TO_EVENT.values()) | {spec.expected_event}
        false_positives = [ev for ev in events
                           if ev.get("event_id") not in used_event_ids
                           and ev.get("type") in corruption_types]
        relevant_events = [ev for ev in events if ev.get("type") in corruption_types]
        matched = len(all_matches)
        gt_total = len(spec.episodes)
        metrics = {
            "dataset": spec.name, "kind": spec.kind, "manifest_type": spec.manifest_type,
            "expected_event": spec.expected_event, "gt_count": gt_total,
            "detected_count": len(events), "relevant_detected_count": len(relevant_events),
            "matched": matched,
            "recall": round(matched / gt_total, 3) if gt_total else None,
            "precision": round(matched / len(relevant_events), 3) if relevant_events else None,
            "misses": [{"seq": m.get("seq"), "start": m["start_frame"], "end": m["end_frame"]}
                       for m in misses],
            "false_positives": [{"event_id": e.get("event_id"), "type": e["type"],
                                 "start": e["start_frame"], "end": e["end_frame"]}
                                for e in false_positives],
            "matches": all_matches,
        }
        if all_matches:
            metrics["mean_iou"] = round(sum(m["iou"] for m in all_matches) / len(all_matches), 3)
            metrics["mean_abs_start_err_frames"] = round(
                sum(abs(m["start_err_frames"]) for m in all_matches) / len(all_matches), 2)
        if extra_metrics:
            metrics.update(extra_metrics)
        return metrics

    gt_type_events = events
    if spec.expected_event:
        gt_type_events = [ev for ev in events if ev["type"] == spec.expected_event]
    result = (match_events(gt_type_events, spec.episodes, min_iou=min_iou, tolerance_frames=tolerance)
              if spec.episodes else {"matches": [], "misses": [], "false_positives": events})
    matched = len(result["matches"])
    gt_total = len(spec.episodes)
    metrics = {
        "dataset": spec.name,
        "kind": spec.kind,
        "manifest_type": spec.manifest_type,
        "expected_event": spec.expected_event,
        "gt_count": gt_total,
        "detected_count": len(events),
        "relevant_detected_count": len(gt_type_events),
        "matched": matched,
        "recall": round(matched / gt_total, 3) if gt_total else None,
        "precision": round(matched / len(gt_type_events), 3) if gt_type_events else None,
        "misses": [{"seq": m.get("seq"), "start": m["start_frame"], "end": m["end_frame"]}
                   for m in result["misses"]],
        "false_positives": [{"event_id": e.get("event_id"), "type": e["type"],
                             "start": e["start_frame"], "end": e["end_frame"]}
                            for e in result["false_positives"]],
        "matches": result["matches"],
    }
    if result["matches"]:
        metrics["mean_iou"] = round(sum(m["iou"] for m in result["matches"]) / len(result["matches"]), 3)
        metrics["mean_abs_start_err_frames"] = round(
            sum(abs(m["start_err_frames"]) for m in result["matches"]) / len(result["matches"]), 2)
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics
