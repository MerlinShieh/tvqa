# -*- coding: utf-8 -*-
"""corruption 检测器单测：撕裂位移差、局部冻结矛盾、块效应。

合成素材使用随机纹理 + 水平滚动（有水平结构，位移估计才有效）。
"""

import cv2
import numpy as np

from tvqa.detectors.corruption import CorruptionDetector

EVAL_CORRUPTION = {
    "enabled": True, "tear_downscale_width": 256, "tear_boundary_step": 8,
    "tear_shift_search": 24, "tear_min_shift_diff": 6, "tear_band_frames": 30,
    "tear_min_event_frames": 3, "tear_end_tolerance_frames": 3, "tear_max_y_spread": 24,
    "block_grid_cols": 16, "block_grid_rows": 9, "frozen_block_max_diff": 0.4,
    "frozen_min_texture": 6.0, "global_motion_min_ratio": 0.65,
    "frozen_min_blocks": 10, "frozen_min_event_frames": 6,
    "blocking_grid": 8, "blocking_min_span": 0.25,
}

RNG = np.random.RandomState(42)
# 低通后的纹理：保证 320→256 降采样后水平结构仍可对齐（真实视频本身是低通的）
_TEXTURE = cv2.GaussianBlur((RNG.rand(180, 320) * 255).astype(np.uint8), (9, 9), 3)


def _textured_frame(shift):
    """随机纹理水平滚动帧（uint8 BGR），水平结构丰富。"""
    gray = np.roll(_TEXTURE, shift, axis=1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _packets(frame_fn, count, start=0):
    from tvqa.sources.base import FramePacket
    return [FramePacket(frame=frame_fn(i), t=(start + i) / 30.0, frame_idx=start + i, fps=30.0)
            for i in range(count)]


def _run(detector, packets):
    events = []
    for packet in packets:
        events.extend(detector.safe_process(packet))
    events.extend(detector.flush())
    return events


def test_tear_detected_on_band_shift():
    """底部 1/3 水平位移逐帧交替（±20px）——帧间位移差显著，应检出撕裂。

    注：撕裂检测量的是「上下行带帧间位移差」，恒定偏移不产生位移差
    （与数据集注入的逐帧变化 shift_x 口径一致）。
    """

    def frame_fn(i):
        frame = _textured_frame(i * 8)
        y0 = int(frame.shape[0] * 0.55)
        extra = 12 if i % 2 == 0 else -12   # 摆幅 24px < 搜索半宽×全分辨率比例
        frame[y0:, :] = np.roll(frame[y0:, :], extra, axis=1)
        return frame

    detector = CorruptionDetector(dict(EVAL_CORRUPTION), dataset="t")
    events = _run(detector, _packets(frame_fn, 12))
    tears = [e for e in events if e["type"] == "tear"]
    assert len(tears) == 1
    assert tears[0]["metrics"]["max_shift_diff_px"] >= 12
    assert 0 < tears[0]["metrics"]["tear_line_y"] < 180


def test_no_tear_on_normal_motion():
    detector = CorruptionDetector(dict(EVAL_CORRUPTION), dataset="t")
    events = _run(detector, _packets(lambda i: _textured_frame(i * 8), 40))
    assert [e for e in events if e["type"] == "tear"] == []


def test_local_freeze_contradiction():
    """整幅滚动 + 一块纹理冻结区域 → local_freeze。"""
    anchor = {}

    def frame_fn(i):
        frame = _textured_frame(i * 8)
        h, w = frame.shape[:2]
        region = (slice(int(h * 0.15), int(h * 0.65)), slice(int(w * 0.1), int(w * 0.6)))
        if i == 0:
            anchor["frozen"] = frame[region].copy()
        frame[region] = anchor["frozen"]
        return frame

    detector = CorruptionDetector(dict(EVAL_CORRUPTION), dataset="t")
    events = _run(detector, _packets(frame_fn, 14))
    freezes = [e for e in events if e["type"] == "local_freeze"]
    assert len(freezes) == 1


def test_blocking_artifact_spike():
    def mosaic(i):
        frame = _textured_frame(i * 8)
        h, w = frame.shape[:2]
        region = (slice(int(h * 0.2), int(h * 0.7)), slice(int(w * 0.2), int(w * 0.7)))
        patch = frame[region]
        small = cv2.resize(patch, (max(1, patch.shape[1] // 8), max(1, patch.shape[0] // 8)),
                           interpolation=cv2.INTER_LINEAR)
        frame[region] = cv2.resize(small, (patch.shape[1], patch.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        return frame

    detector = CorruptionDetector(dict(EVAL_CORRUPTION), dataset="t")
    packets = _packets(lambda i: _textured_frame(i * 8), 40)
    packets += _packets(mosaic, 12, start=40)
    packets += _packets(lambda i: _textured_frame((52 + i) * 8), 20, start=52)
    events = _run(detector, packets)
    assert len([e for e in events if e["type"] == "blocking"]) >= 1
