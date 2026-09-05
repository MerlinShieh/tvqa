# -*- coding: utf-8 -*-
"""评测/现场主循环：源产帧 → 检测器 → 证据回捞 → 归档。

关注点分离：
- 检测器只产事件 dict（帧区间 + metrics），不碰文件；
- pipeline 负责事件闭合后的证据回捞（按帧号向 source 取原帧存图）、
  event_id 分配、归档写入与日志。
"""

from .logging_setup import get_logger
from .sources.frames_dir import FramesDirSource
from .utils import save_image

log = get_logger("pipeline")

_EVIDENCE_POINTS = ("start", "mid", "end")


def _evidence_frames(ev):
    start, end = ev["start_frame"], ev["end_frame"]
    mid = (start + end) // 2
    return {"start": start, "mid": mid, "end": end}


def attach_evidence(session, source, ev, evidence_points=("start", "mid", "end")):
    """事件闭合时回捞代表帧存图；失败不阻断主流程。"""
    seq = len(ev.get("event_id", "") )
    folder = f"{ev['type']}_f{ev['start_frame']:05d}-{ev['end_frame']:05d}"
    ev_dir = session.evidence_dir(ev.get("dataset", "-"), folder)
    frames_saved = {}
    points = _evidence_frames(ev)
    for point in evidence_points:
        idx = points.get(point)
        if idx is None:
            continue
        frame = None
        try:
            frame = source.get_frame(idx)
        except Exception as error:  # noqa: BLE001
            log.debug(f"证据回捞失败 frame={idx}: {error}")
        if frame is None:
            continue
        name = f"img_{point}_f{idx:05d}.jpg"
        if save_image(ev_dir / name, frame):
            frames_saved[point] = session.rel(ev_dir / name)
    session.write_evidence_json(ev_dir, "event.json", ev)
    ev["evidence_dir"] = session.rel(ev_dir)
    ev["evidence_frames"] = frames_saved


def run_frames_dataset(cfg, session, source, dataset_name):
    """在 frames_dir 数据集上跑全部启用的视觉检测器。返回事件列表。"""
    from .detectors import build_visual_detectors
    detectors = build_visual_detectors(cfg, session, dataset_name)
    events = []
    for packet in source.packets():
        for detector in detectors:
            for event in detector.safe_process(packet):
                finalize_event(cfg, session, source, event)
                events.append(event)
    for detector in detectors:
        try:
            for event in detector.flush():
                finalize_event(cfg, session, source, event)
                events.append(event)
        except Exception as error:  # noqa: BLE001
            log.warning(f"[{detector.name()}] flush 异常：{error}")
    return events


def finalize_event(cfg, session, source, event):
    event["dataset"] = event.get("dataset") or "-"
    event["event_id"] = session.next_event_id(event["type"], event["dataset"])
    if cfg.get("evidence.frames_per_event") is not None:
        attach_evidence(session, source, event)
    log.info(f"事件 {event['event_id']} frames={event['start_frame']}..{event['end_frame']} "
             f"dur={event['duration_frames']}f level={event['level']} metrics={event.get('metrics')}")
    session.record_event(event)
