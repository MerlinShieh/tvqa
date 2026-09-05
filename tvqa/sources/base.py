# -*- coding: utf-8 -*-
"""输入数据包的统一协议。检测器只认这些字段，与后端解耦。"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray        # BGR 图像
    t: float                 # 时间轴秒（真实或虚拟）
    frame_idx: int           # 绝对帧号（数据集=文件序号；采集卡=单调递增计数）
    fps: float


@dataclass
class AudioPacket:
    samples: np.ndarray      # 单声道 float32 一帧音频块
    t: float                 # 该块起始秒
    samplerate: int
    seq: int = 0


@dataclass
class StreamMeta:
    fps: float
    width: int
    height: int
    total_frames: int = -1   # 未知=-1
    extra: dict = field(default_factory=dict)
