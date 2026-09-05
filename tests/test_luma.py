# -*- coding: utf-8 -*-
"""luma 检测器行为单测：黑/白屏确认与闭合、闪烁跳变率、同源合并。"""

from tvqa.detectors.luma import LumaDetector

EVAL_LUMA = {
    "enabled": True, "thumbnail_width": 85, "thumbnail_height": 48,
    "black_brightness_threshold": 25.0, "black_pixel_threshold": 30,
    "black_ratio_threshold": 0.95, "black_confirm_frames": 4,
    "black_serious_frames": 600, "black_end_tolerance_frames": 2,
    "white_brightness_threshold": 235.0, "white_pixel_threshold": 225,
    "white_ratio_threshold": 0.95, "white_confirm_frames": 4,
    "white_serious_frames": 600, "white_end_tolerance_frames": 2,
    "flicker_window_frames": 30, "flicker_min_amplitude": 10.0,
    "flicker_min_rate": 3.0, "flicker_end_tolerance_frames": 9,
}


def run(detector, packets):
    events = []
    for packet in packets:
        events.extend(detector.safe_process(packet))
    events.extend(detector.flush())
    return events


def test_black_event_confirmed_and_closed(packet_factory):
    detector = LumaDetector(dict(EVAL_LUMA), dataset="t")
    packets = [packet_factory(i, brightness=120) for i in range(60)]
    packets += [packet_factory(i, polarity="black") for i in range(60, 70)]
    packets += [packet_factory(i, brightness=120) for i in range(70, 100)]
    events = run(detector, packets)
    black = [e for e in events if e["type"] == "black"]
    assert len(black) == 1
    event = black[0]
    assert event["start_frame"] == 60 and event["end_frame"] == 69
    assert event["duration_frames"] == 10


def test_black_below_confirm_not_reported(packet_factory):
    detector = LumaDetector(dict(EVAL_LUMA), dataset="t")
    packets = [packet_factory(i, brightness=120) for i in range(50)]
    packets += [packet_factory(i, polarity="black") for i in range(50, 53)]  # 3 帧 < 确认 4
    packets += [packet_factory(i, brightness=120) for i in range(53, 80)]
    events = run(detector, packets)
    assert not [e for e in events if e["type"] in ("black", "flicker")]


def test_white_symmetric(packet_factory):
    detector = LumaDetector(dict(EVAL_LUMA), dataset="t")
    packets = [packet_factory(i, brightness=120) for i in range(40)]
    packets += [packet_factory(i, polarity="white") for i in range(40, 50)]
    packets += [packet_factory(i, brightness=120) for i in range(50, 70)]
    events = run(detector, packets)
    white = [e for e in events if e["type"] == "white"]
    assert len(white) == 1 and white[0]["start_frame"] == 40


def test_flicker_jump_rate(packet_factory):
    detector = LumaDetector(dict(EVAL_LUMA), dataset="t")
    packets = []
    for i in range(30):
        packets.append(packet_factory(i, brightness=120))
    for i in range(30, 90):  # 每 3 帧一次暗→亮往复：2 跳变/3 帧 = 20 跳变/秒
        brightness = 20.0 if (i - 30) % 3 == 0 else 120.0
        packets.append(packet_factory(i, brightness=brightness))
    for i in range(90, 130):
        packets.append(packet_factory(i, brightness=120))
    events = run(detector, packets)
    flicker = [e for e in events if e["type"] == "flicker"]
    assert len(flicker) == 1
    assert flicker[0]["metrics"]["measured_flip_rate_per_sec"] >= 15
    # 事件窗口应覆盖闪烁段
    assert flicker[0]["start_frame"] <= 35 and flicker[0]["end_frame"] >= 85


def test_flicker_suppresses_black_double_report(packet_factory):
    """黑闪（黑帧与亮帧往复）只报 flicker，不报 black。"""
    detector = LumaDetector(dict(EVAL_LUMA), dataset="t")
    packets = [packet_factory(i, brightness=120) for i in range(30)]
    for i in range(30, 80):
        packets.append(packet_factory(i, polarity="black" if (i - 30) % 2 == 0 else None))
    packets += [packet_factory(i, brightness=120) for i in range(80, 120)]
    events = run(detector, packets)
    assert [e for e in events if e["type"] == "black"] == []
    assert len([e for e in events if e["type"] == "flicker"]) >= 1
