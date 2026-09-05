# -*- coding: utf-8 -*-
"""归档/配置/匹配器单测：CSV BOM 修复、JSONL 追溯、区间匹配、配置合并。"""

import csv
import json

from tvqa.archive import Session
from tvqa.config import load_config
from tvqa.evaluate.matcher import match_events, score_dataset
from tvqa.evaluate.manifest_loader import DatasetSpec


def test_csv_no_repeated_bom_on_append(tmp_path):
    """追加行不得再次写入 BOM（修复原脚本 bug）。"""
    session = Session(tmp_path / "out", mode="test", config_snapshot={})
    for i in range(3):
        session.record_event({"type": "black", "dataset": "d", "start_frame": i,
                              "end_frame": i + 1, "start_t_sec": 0.0, "end_t_sec": 0.1,
                              "duration_sec": 0.1, "level": "SUSPECT", "trigger": "T",
                              "status": "confirmed", "metrics": {}})
    session.finish()
    raw = (session.root / "black_events_summary.csv").read_bytes()
    assert raw.count(b"\xef\xbb\xbf") == 1  # 仅表头一个 BOM
    with (session.root / "black_events_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 4 and rows[0][0] == "event_id"


def test_events_jsonl_trace_fields(tmp_path):
    session = Session(tmp_path / "out", mode="test", config_snapshot={"k": "v"})
    session.record_event({"type": "stutter", "dataset": "d", "start_frame": 1, "end_frame": 9,
                          "start_t_sec": 0.0, "end_t_sec": 0.3, "duration_sec": 0.3,
                          "level": "SUSPECT", "trigger": "X", "status": "confirmed", "metrics": {}})
    session.finish()
    lines = (session.root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    assert event["event_id"].startswith("stutter@")
    assert "log_line" in event


def test_match_interval_basic():
    gt = [{"start_frame": 100, "end_frame": 120}]
    det = [{"start_frame": 102, "end_frame": 118, "type": "black", "event_id": "x"}]
    result = match_events(det, gt)
    assert len(result["matches"]) == 1 and not result["misses"] and not result["false_positives"]
    assert abs(result["matches"][0]["start_err_frames"]) == 2


def test_match_tolerance_frames():
    """匹配容忍帧数可配置：真值区间双向扩张后再算重叠，误差仍按原始区间。"""
    gt = [{"start_frame": 100, "end_frame": 104}]
    det = [{"start_frame": 125, "end_frame": 129, "type": "black", "event_id": "x"}]
    # 无容忍：差 21 帧，不匹配
    assert not match_events(det, gt)["matches"]
    # 容忍 30 帧（≈1 秒）：匹配成功
    result = match_events(det, gt, tolerance_frames=30)
    assert len(result["matches"]) == 1
    assert result["matches"][0]["start_err_frames"] == 25  # 误差不受扩张影响


def test_ignore_events_below_frames():
    """评测口径：短于阈值的事件不参与记分。"""
    spec = DatasetSpec(name="黑屏", path=".", kind="visual", manifest_type="black_screen",
                       episodes=[{"start_frame": 100, "end_frame": 120}],
                       expected_event="black")
    events = [{"type": "black", "start_frame": 100, "end_frame": 101, "duration_frames": 2, "event_id": "tiny"},
              {"type": "black", "start_frame": 105, "end_frame": 119, "duration_frames": 15, "event_id": "real"}]
    card = score_dataset(spec, events, evaluate_cfg={"ignore_events_below_frames": 5})
    assert card["detected_count"] == 1
    assert card["matched"] == 1 and card["false_positives"] == []


def test_score_corruption_by_style():
    spec = DatasetSpec(name="花屏", path=".", kind="visual", manifest_type="corruption",
                       episodes=[{"start_frame": 10, "end_frame": 19, "style": "frozen_region"},
                                 {"start_frame": 600, "end_frame": 640, "style": "mosaic"}],
                       expected_event="local_freeze")
    events = [{"type": "local_freeze", "start_frame": 10, "end_frame": 18, "event_id": "a"},
              {"type": "blocking", "start_frame": 601, "end_frame": 639, "event_id": "b"},
              # 非 corruption 类事件（如其他检测器的输出）不计入花屏误报
              {"type": "black", "start_frame": 900, "end_frame": 910, "event_id": "c"},
              {"type": "local_freeze", "start_frame": 900, "end_frame": 910, "event_id": "d"}]
    card = score_dataset(spec, events)
    assert card["matched"] == 2 and card["recall"] == 1.0
    assert len(card["false_positives"]) == 1  # 只有未匹配的 local_freeze 计为误报


def test_config_profile_merge_and_overrides(tmp_path):
    (tmp_path / "default.yaml").write_text("a: {b: 1, c: 2}\n", encoding="utf-8")
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "p.yaml").write_text("a: {c: 9}\n", encoding="utf-8")
    cfg = load_config("p", overrides=["a.b=7"], config_dir=tmp_path)
    assert cfg.get("a.b") == 7 and cfg.get("a.c") == 9
