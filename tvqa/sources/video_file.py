# -*- coding: utf-8 -*-
"""视频文件后端：cv2.VideoCapture 读 mp4，产 FramePacket。

音画不同步数据集（input/音画不同步/*.mp4）用它；时间戳用帧号/fps 的虚拟轴，
与音频流的采样时刻共用同一零点（由 avsync 对齐）。
"""

import cv2

from ..logging_setup import get_logger
from ..utils import safe_float
from .base import FramePacket, StreamMeta

log = get_logger("sources.video_file")


class VideoFileSource:
    def __init__(self, cfg_section, clock=None):
        self.path = cfg_section.get("path") or cfg_section.get("video_file")
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件：{self.path}")
        reported_fps = safe_float(self._cap.get(cv2.CAP_PROP_FPS), 0.0)
        self.fps = safe_float(cfg_section.get("fps"), reported_fps if reported_fps > 0 else 30.0)
        self._clock = clock
        self.meta = StreamMeta(
            fps=self.fps,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def get_frame(self, frame_idx):
        """按帧号 seek 回捞（重解码，偶尔调用可接受）。"""
        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = self._cap.read()
            return frame if ok else None
        except Exception:  # noqa: BLE001
            return None

    def packets(self):
        frame_idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            t = frame_idx / self.fps
            if self._clock is not None and hasattr(self._clock, "set_frame"):
                t = self._clock.set_frame(frame_idx)
            yield FramePacket(frame=frame, t=t, frame_idx=frame_idx, fps=self.fps)
            frame_idx += 1

    def close(self):
        self._cap.release()
