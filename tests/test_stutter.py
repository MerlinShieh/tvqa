# -*- coding: utf-8 -*-
"""stutter 检测器单测：精确重复帧冻结的触发、持续与恢复。"""

from tvqa.detectors.stutter import StutterDetector

EVAL_STUTTER = {
    "enabled": True, "analysis_every_n_frames": 1, "analysis_width": 320, "analysis_height": 180,
    "changed_pixel_threshold": 5, "duplicate_mean_diff": 0.05, "duplicate_changed_ratio": 0.0,
    "motion_mean_diff": 2.8, "motion_changed_ratio": 0.03,
    "required_motion_frames": 6, "motion_break_tolerance_frames": 8, "motion_arm_timeout_frames": 45,
    "duplicate_trigger_frames": 4, "duplicate_break_tolerance_frames": 6,
    "window_frames": 180, "duplicate_ratio_threshold": 0.10, "min_duplicate_bursts": 2,
    "active_bad_burst_frames": 4, "active_window_frames": 180,
    "recovery_quiet_frames": 45, "dark_brightness_threshold": 25.0, "dark_black_ratio": 0.95,
    "serious_stutter_frames": 90,
}


def run(detector, packets):
    events = []
    for packet in packets:
        events.extend(detector.safe_process(packet))
    events.extend(detector.flush())
    return events


def _motion_frames(count, start, brightness=120.0, speed=7):
    """持续位移的内容帧（每帧移动 speed 像素，帧间差显著）。"""
    return [(start + i, dict(brightness=brightness, offset=start + i)) for i in range(count)]


def test_freeze_after_motion_triggers(packet_factory, frame_factory):
    detector = StutterDetector(dict(EVAL_STUTTER), dataset="t")
    packets = []
    from tvqa.sources.base import FramePacket
    for i in range(50):  # 0..49 持续运动
        packets.append(FramePacket(frame=frame_factory(offset=i * 7), t=i / 30.0,
                                   frame_idx=i, fps=30.0))
    # 50..64 冻结：重复同一帧（精确相同；末帧 49 与冻结帧像素一致属像素真值边界）
    frozen = packets[-1].frame.copy()
    for i in range(50, 65):
        packets.append(FramePacket(frame=frozen.copy(), t=i / 30.0, frame_idx=i, fps=30.0))
    for i in range(65, 110):  # 65..110 恢复运动
        packets.append(FramePacket(frame=frame_factory(offset=i * 7), t=i / 30.0,
                                   frame_idx=i, fps=30.0))
    events = run(detector, packets)
    stutters = [e for e in events if e["type"] == "stutter"]
    assert len(stutters) == 1
    event = stutters[0]
    assert event["trigger"] == "CONTINUOUS_DUPLICATE_FRAMES"
    assert event["start_frame"] in (49, 50)   # 49 与冻结帧像素相同，两答皆可
    assert abs(event["end_frame"] - 64) <= 2


def test_static_scene_without_recent_motion_no_event(packet_factory, frame_factory):
    """静止场景（运动早已远离武装窗口）不应触发卡顿。

    运动段后插入「噪声微变」段（帧间差 > 重复阈、< 运动阈且无运动），使武装
    超时先解除，再进入完全静止段——像素上与冻结一致但无运动前置，不应建事件。
    """
    import numpy as np
    from tvqa.sources.base import FramePacket
    detector = StutterDetector(dict(EVAL_STUTTER), dataset="t")
    packets = []
    for i in range(30):  # 运动段
        packets.append(FramePacket(frame=frame_factory(offset=i * 7), t=i / 30.0,
                                   frame_idx=i, fps=30.0))
    base_frame = frame_factory(offset=0, brightness=120)
    rng = np.random.RandomState(0)
    for i in range(30, 110):  # 噪声微变段：非重复、非运动、无跳变
        noisy = base_frame.astype(np.int16) + rng.randint(-2, 3, size=base_frame.shape)
        packets.append(FramePacket(frame=noisy.astype(np.uint8), t=i / 30.0,
                                   frame_idx=i, fps=30.0))
    static_frame = packets[-1].frame.copy()
    for i in range(110, 220):  # 完全静止段（精确重复帧）
        packets.append(FramePacket(frame=static_frame.copy(), t=i / 30.0,
                                   frame_idx=i, fps=30.0))
    events = run(detector, packets)
    assert [e for e in events if e["type"] == "stutter"] == []


def test_black_frame_interrupts_stutter(packet_factory, frame_factory):
    from tvqa.sources.base import FramePacket
    detector = StutterDetector(dict(EVAL_STUTTER), dataset="t")
    packets = []
    for i in range(30):
        packets.append(FramePacket(frame=frame_factory(offset=i * 7), t=i / 30.0, frame_idx=i, fps=30.0))
    frozen = packets[-1].frame.copy()
    for i in range(30, 45):
        packets.append(FramePacket(frame=frozen.copy(), t=i / 30.0, frame_idx=i, fps=30.0))
    for i in range(45, 60):  # 黑屏打断
        packets.append(FramePacket(frame=frame_factory(polarity="black"), t=i / 30.0,
                                   frame_idx=i, fps=30.0))
    events = run(detector, packets)
    stutters = [e for e in events if e["type"] == "stutter"]
    assert len(stutters) == 1
    assert stutters[0]["end_frame"] < 45  # 事件在黑屏前闭合
