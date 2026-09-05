# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def frame_factory():
    """合成帧工厂：base 亮度 + 移动条带（可观测运动）+ 可选极性覆盖。"""

    def make(width=320, height=180, brightness=120.0, offset=0, polarity=None):
        frame = np.full((height, width, 3), brightness, dtype=np.uint8)
        y = (offset * 3) % height
        frame[max(0, y - 20):y, :, :] = np.clip(brightness + 60, 0, 255)
        if polarity == "black":
            frame[:] = 0
        elif polarity == "white":
            frame[:] = 255
        return frame

    return make


@pytest.fixture
def packet_factory(frame_factory):
    from tvqa.sources.base import FramePacket

    def make(frame_idx, polarity=None, brightness=120.0, offset=None, width=320, height=180):
        if offset is None:
            offset = frame_idx
        frame = frame_factory(width=width, height=height, brightness=brightness,
                              offset=offset, polarity=polarity)
        return FramePacket(frame=frame, t=frame_idx / 30.0, frame_idx=frame_idx, fps=30.0)

    return make
