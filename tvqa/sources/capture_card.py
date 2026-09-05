# -*- coding: utf-8 -*-
"""采集卡后端（HDMI 采集，Windows 用 DSHOW）。硬件到位后 video.backend: capture_card。

- 时间戳来自 RealClock（monotonic），帧号单调计数。
- 读帧失败重试与降级由主循环处理；此处只做设备打开与逐帧产出。
- 设备编号/分辨率/帧率全部走配置，与仿真后端完全同构。
"""

import time

import cv2

from ..logging_setup import get_logger
from ..utils import safe_float
from .base import FramePacket, StreamMeta

log = get_logger("sources.capture_card")


class CaptureCardSource:
    def __init__(self, cfg_section, clock):
        import platform
        self.cfg_section = cfg_section
        index = int(cfg_section.get("index", 0))
        if platform.system() == "Windows":
            self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开采集设备 index={index}（可尝试 1/2/3）")
        self._stop = False
        if cfg_section.get("set_resolution"):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg_section.get("width", 1920)))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg_section.get("height", 1080)))
        if cfg_section.get("set_fps"):
            self._cap.set(cv2.CAP_PROP_FPS, float(cfg_section.get("fps", 30)))
        reported_fps = safe_float(self._cap.get(cv2.CAP_PROP_FPS), 0.0)
        self.fps = reported_fps if 5 <= reported_fps <= 240 else float(cfg_section.get("fps", 30))
        self._clock = clock
        self.meta = StreamMeta(fps=self.fps,
                               width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               total_frames=-1)

    def get_frame(self, frame_idx):
        """采集卡不可回看历史帧：返回 None，证据用实时模式另存。"""
        return None

    def packets(self):
        frame_idx = 0
        failures = 0
        max_failures = int(self.cfg_section.get("max_read_failures", 150))  # 约 7.5s
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= max_failures:
                    raise RuntimeError(
                        f"采集设备连续 {failures} 次读帧失败（设备未连接/被占用/信号丢失）")
                time.sleep(0.05)
                continue
            failures = 0
            t = self._clock.now() if self._clock is not None else time.monotonic()
            yield FramePacket(frame=frame, t=t, frame_idx=frame_idx, fps=self.fps)
            frame_idx += 1

    def close(self):
        self._stop = True
        try:
            self._cap.release()
        except Exception:  # noqa: BLE001
            pass
