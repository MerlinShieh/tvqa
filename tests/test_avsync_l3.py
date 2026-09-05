# -*- coding: utf-8 -*-
"""L3 主动测试片链路单测：生成已知偏移 → 测量 → 误差 ≤ 1 帧。

需要 imageio-ffmpeg（工程 requirements 已含）；用短测试片控制耗时。
"""

from tvqa.detectors.avsync import L3Analyzer, generate_test_clip


def test_l3_roundtrip(tmp_path):
    clip = tmp_path / "l3_short.mp4"
    generate_test_clip(clip, offset_ms=200, seconds=12.0)  # 4 个标记对
    result = L3Analyzer().measure(clip)
    assert result["confidence"] == "high"
    error_frames = abs(result["offset_ms"] - 200) / 1000 * 30
    assert error_frames <= 1.5, f"实测 {result} 偏差过大"
