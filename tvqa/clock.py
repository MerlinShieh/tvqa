# -*- coding: utf-8 -*-
"""统一时间轴。

检测器只消费「秒」时间戳，不关心时间来自真实 monotonic 还是数据集虚拟时钟。
- RealClock：现场模式，包装 time.monotonic()。
- VirtualClock：批量/仿真模式，帧号 / fps 推进，可加速运行（评测时不真正 sleep）。
这样 frames_dir / capture_card 两种后端对检测器完全等价。
"""

import time


class Clock:
    def now(self):
        raise NotImplementedError

    @property
    def real_time(self):
        return True


class RealClock(Clock):
    def now(self):
        return time.monotonic()


class VirtualClock(Clock):
    """数据集驱动的时钟：外部每次喂帧调用 advance(1/fps)。

    start: 起始时间戳；帧 t = start + frame_index * frame_interval。
    """

    def __init__(self, fps, start=0.0):
        if fps <= 0:
            raise ValueError(f"fps 必须为正，当前={fps}")
        self.interval = 1.0 / fps
        self._t = start
        self._real_time = False

    def now(self):
        return self._t

    def advance(self, steps=1):
        self._t += self.interval * steps
        return self._t

    def set_frame(self, frame_index):
        """按绝对帧号定位时间（frames_dir 每帧带真实 frame_idx 时用）。"""
        self._t = frame_index * self.interval
        return self._t

    @property
    def real_time(self):
        return self._real_time


class ClockOffset:
    """外部通道（ADB/串口设备）与 PC 时钟的偏移估计。

    周期性采集 (pc_monotonic, device_epoch) 配对样本，用中位数 + 线性漂移
    拟合 offset，供 probe 折算设备日志时间戳。首轮实现保留接口与最小实现，
    真实校时在硬件到位后联调。
    """

    def __init__(self):
        self._samples = []  # (pc_time, device_time)

    def add_sample(self, pc_time, device_time):
        self._samples.append((pc_time, device_time))
        if len(self._samples) > 200:
            self._samples = self._samples[-200:]

    @property
    def available(self):
        return len(self._samples) >= 2

    def to_pc_time(self, device_time):
        """把设备侧 epoch 时间折算到 PC monotonic 轴。无足够样本时按恒定偏移 0。"""
        if not self.available:
            return device_time
        offsets = sorted(pc - dev for pc, dev in self._samples)
        return device_time + offsets[len(offsets) // 2]
