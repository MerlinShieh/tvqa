# -*- coding: utf-8 -*-
"""HTML 会话报告（零外部依赖：内联 CSS/SVG）。

读取会话目录的 scorecard.json + events.jsonl + run_meta.json，生成 index.html：
总览记分卡 → 音画对账表 → 事件明细表（证据/日志可点击回溯）→ AV 偏移散点图。
"""

import html
import json
from pathlib import Path


def generate_report(session_root):
    root = Path(session_root)
    meta = _read_json(root / "run_meta.json", {})
    scorecards = _read_json(root / "scorecard.json", [])
    events = _read_jsonl(root / "events.jsonl")
    html_text = _render(meta, scorecards, events)
    out = root / "index.html"
    out.write_text(html_text, encoding="utf-8")
    return out


def _read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _read_jsonl(path):
    items = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
    except Exception:  # noqa: BLE001
        pass
    return items


def _esc(value):
    return html.escape(str(value))


def _render(meta, scorecards, events):
    cards_visual = [c for c in scorecards if c.get("kind") == "visual"]
    cards_baseline = [c for c in scorecards if c.get("kind") == "baseline"]
    cards_av = [c for c in scorecards if c.get("kind") == "av"]
    cards_l3 = [c for c in scorecards if c.get("kind") == "av_l3"]
    total_events = len(events)

    parts = [_HEADER_CSS, "<html><head><meta charset='utf-8'><title>tvqa 检测报告</title></head><body>"]
    session_name = (meta or {}).get("session", Path.cwd().name)
    parts.append(f"<h1>电视音画质量检测报告 <small>{_esc(session_name)}</small></h1>")
    parts.append(f"<p class='meta'>模式：{_esc(meta.get('mode'))}　开始：{_esc(meta.get('start_bj_time'))}"
                 f"　事件总数：{total_events}　代码：{_esc((meta.get('code_version') or {}).get('git') or (meta.get('code_version') or {}).get('package_version'))}</p>")

    # ---------- 记分卡 ----------
    parts.append("<h2>一、视觉故障记分卡</h2><table><tr><th>数据集</th><th>故障类型</th><th>真值</th>"
                 "<th>检出</th><th>匹配</th><th>召回</th><th>精确率</th><th>平均IoU</th>"
                 "<th>起点误差(帧)</th><th>漏检</th><th>误报</th><th>结论</th></tr>")
    for card in cards_visual:
        verdict = "PASS" if card.get("recall") == 1.0 and not card.get("false_positives") else "CHECK"
        cls = "pass" if verdict == "PASS" else "warn"
        parts.append(
            f"<tr><td>{_esc(card['dataset'])}</td><td>{_esc(card.get('manifest_type'))}</td>"
            f"<td>{card['gt_count']}</td><td>{card['relevant_detected_count']}</td><td>{card['matched']}</td>"
            f"<td>{card.get('recall')}</td><td>{card.get('precision')}</td><td>{card.get('mean_iou')}</td>"
            f"<td>{card.get('mean_abs_start_err_frames')}</td><td>{len(card.get('misses', []))}</td>"
            f"<td>{len(card.get('false_positives', []))}</td><td class='{cls}'>{verdict}</td></tr>")
    for card in cards_baseline:
        fps = len(card.get("false_positives", []))
        verdict = "PASS" if fps == 0 else f"{fps} FP"
        cls = "pass" if fps == 0 else "warn"
        parts.append(f"<tr><td>{_esc(card['dataset'])}</td><td>正常基线</td><td>0</td>"
                     f"<td>{card['detected_count']}</td><td>-</td><td>-</td><td>-</td><td>-</td>"
                     f"<td>-</td><td>-</td><td>{fps}</td><td class='{cls}'>{verdict}</td></tr>")
    parts.append("</table>")

    # ---------- 音画 ----------
    if cards_av or cards_l3:
        parts.append("<h2>二、音画同步对账</h2>")
        if cards_av:
            parts.append("<h3>L1 被动测量（依赖内容语义锚点；低置信=内容无锚点，属预期）</h3>"
                         "<table><tr><th>文件</th><th>说明</th><th>真值offset(ms)</th>"
                         "<th>实测offset(ms)</th><th>误差(ms)</th><th>误差(帧)</th>"
                         "<th>主簇/总配对</th><th>置信度</th><th>结论</th></tr>")
            for card in cards_av:
                for row in card.get("cases", []):
                    err = row.get("error_frames")
                    conf = row.get("confidence")
                    verdict = "无锚点(预期)" if conf in ("low", "none") else (
                        "PASS" if (err is not None and err <= 1.0 and row.get("sign_correct")) else "CHECK")
                    cls = "pass" if verdict == "PASS" else "warn"
                    parts.append(f"<tr><td>{_esc(row['file'])}</td><td>{_esc(row.get('desc'))}</td>"
                                 f"<td>{row.get('expected_offset_ms')}</td><td>{row.get('measured_offset_ms')}</td>"
                                 f"<td>{row.get('error_ms')}</td><td>{row.get('error_frames')}</td>"
                                 f"<td>{row.get('cluster_pairs')}/{row.get('pairs')}</td>"
                                 f"<td>{conf}</td><td class='{cls}'>{verdict}</td></tr>")
            parts.append("</table>")
        if cards_l3:
            parts.append("<h3>L3 主动测试片自检（测量链路正确性基准）</h3>"
                         "<table><tr><th>测试片</th><th>注入offset(ms)</th><th>实测offset(ms)</th>"
                         "<th>误差(帧)</th><th>结论</th></tr>")
            for card in cards_l3:
                for row in card.get("cases", []):
                    cls = "pass" if row.get("pass") else "warn"
                    parts.append(f"<tr><td>{_esc(row['file'])}</td><td>{row.get('expected_offset_ms')}</td>"
                                 f"<td>{row.get('measured_offset_ms')}</td><td>{row.get('error_frames')}</td>"
                                 f"<td class='{cls}'>{'PASS' if row.get('pass') else 'CHECK'}</td></tr>")
            summary = cards_l3[0]
            parts.append(f"<p class='meta'>L3 通过 {summary.get('pass_count')}/{summary.get('case_count')}，"
                         f"最大误差 {summary.get('max_abs_error_frames')} 帧</p></table>")
            parts.append(_av_scatter_svg(cards_l3, cards_av))

    # ---------- 事件明细 ----------
    parts.append("<h2>三、事件明细（可回溯）</h2>"
                 "<table><tr><th>event_id</th><th>类型</th><th>数据集</th><th>帧区间</th>"
                 "<th>时长(帧)</th><th>等级</th><th>触发</th><th>指标</th><th>证据</th><th>日志行</th></tr>")
    for ev in events:
        metrics = ev.get("metrics") or {}
        metrics_text = "，".join(f"{k}={v}" for k, v in metrics.items()) if metrics else "-"
        evidence = ev.get("evidence_dir")
        evidence_link = f"<a href='{_esc(evidence)}/event.json'>{_esc(evidence)}</a>" if evidence else "-"
        parts.append(f"<tr><td>{_esc(ev.get('event_id'))}</td><td>{_esc(ev.get('type'))}</td>"
                     f"<td>{_esc(ev.get('dataset'))}</td><td>{ev.get('start_frame')}..{ev.get('end_frame')}</td>"
                     f"<td>{ev.get('duration_frames')}</td><td>{_esc(ev.get('level'))}</td>"
                     f"<td>{_esc(ev.get('trigger'))}</td><td class='metrics'>{_esc(metrics_text)}</td>"
                     f"<td>{evidence_link}</td><td>{ev.get('log_line')}</td></tr>")
    parts.append("</table>")

    parts.append("<p class='meta'>追溯链：报告行 → event_id → evidence 目录（event.json + 截图）→ "
                 "logs/tvqa.log 对应行号。</p></body></html>")
    return "".join(parts)


def _av_scatter_svg(cards_l3, cards_av=()):
    """期望 vs 实测 offset 散点（内联 SVG，含 y=x 参考线）。优先 L3 结果。"""
    points = []
    for card in cards_l3:
        for row in card.get("cases", []):
            if row.get("expected_offset_ms") is not None and row.get("measured_offset_ms") is not None:
                points.append((row["expected_offset_ms"], row["measured_offset_ms"], row.get("file", "")))
    if not points:
        for card in cards_av:
            for row in card.get("cases", []):
                if row.get("expected_offset_ms") is not None and row.get("measured_offset_ms") is not None:
                    points.append((row["expected_offset_ms"], row["measured_offset_ms"], row.get("file", "")))
    if not points:
        return ""
    width, height, pad = 460, 300, 40
    xs = [p[0] for p in points] + [0]
    ys = [p[1] for p in points] + [0]
    x_min, x_max = min(xs) - 50, max(xs) + 50
    y_min, y_max = min(ys) - 50, max(ys) + 50

    def sx(value):
        return pad + (value - x_min) / max(x_max - x_min, 1) * (width - 2 * pad)

    def sy(value):
        return height - pad - (value - y_min) / max(y_max - y_min, 1) * (height - 2 * pad)

    parts = [f"<h3>音画 offset：真值 vs 实测</h3><svg width='{width}' height='{height}' "
             f"style='background:#fafafa;border:1px solid #ddd'>"]
    parts.append(f"<line x1='{sx(x_min):.1f}' y1='{sy(y_min):.1f}' x2='{sx(x_max):.1f}' y2='{sy(y_max):.1f}' "
                 f"stroke='#bbb' stroke-dasharray='4 3'/>")
    for x, y, name in points:
        parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='5' fill='#2b6cb8'><title>{_esc(name)}"
                     f"：期望{x}ms 实测{y}ms</title></circle>")
    parts.append(f"<text x='{pad}' y='{height - 10}' font-size='11'>真值 offset (ms) →</text>"
                 f"<text x='6' y='{pad}' font-size='11' transform='rotate(-90 12 {pad})'>实测 (ms)</text></svg>")
    return "".join(parts)


_HEADER_CSS = """<style>
body{font-family:'Microsoft YaHei',sans-serif;margin:24px;color:#222}
h1 small{font-size:14px;color:#666;font-weight:normal;margin-left:12px}
h2{border-bottom:2px solid #2b6cb8;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th,td{border:1px solid #ccc;padding:5px 8px;text-align:left}
th{background:#eef3fa}
tr:nth-child(even){background:#f7f9fc}
.pass{color:#0a7d32;font-weight:bold}.warn{color:#b35900;font-weight:bold}
.metrics{font-family:Consolas,monospace;font-size:12px;color:#444}
.meta{color:#666;font-size:13px}
a{color:#2b6cb8}
</style>"""
