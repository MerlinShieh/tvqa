# -*- coding: utf-8 -*-
"""帧目录后端：读取 input/<故障目录>/frame_%05d.png 序列，产出与采集卡同构的 FramePacket。

- 帧号即 manifest 的真值坐标；fps 优先取数据集 manifest.json，其次配置。
- 不真正 sleep（VirtualClock 由主循环按帧推进），批量评测可全速跑。
"""

import re
from pathlib import Path

import cv2
import numpy as np

from ..logging_setup import get_logger
from ..utils import safe_float
from .base import FramePacket, StreamMeta

log = get_logger("sources.frames_dir")

_FRAME_NAME_RE = re.compile(r"^frame[_-](\d+)\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)


class FramesDirSource:
    """按文件名帧号升序产帧。"""

    def __init__(self, cfg_section, clock=None):
        self.path = Path(cfg_section.get("path") or cfg_section.get("frames_dir"))
        if not self.path.is_dir():
            raise FileNotFoundError(f"帧目录不存在：{self.path}")
        self.fps = safe_float(cfg_section.get("fps", 30.0), 30.0)
        self._clock = clock
        self._frames = self._scan()
        if not self._frames:
            raise FileNotFoundError(f"帧目录中没有匹配 frame_*.png 的文件：{self.path}")
        first = cv2.imdecode(np.fromfile(str(self._frames[0][1]), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.meta = StreamMeta(fps=self.fps, width=first.shape[1], height=first.shape[0],
                               total_frames=len(self._frames), extra={"dataset_dir": str(self.path)})

    def _scan(self):
        indexed = []
        for entry in self.path.iterdir():
            match = _FRAME_NAME_RE.match(entry.name)
            if match:
                indexed.append((int(match.group(1)), entry))
        indexed.sort(key=lambda pair: pair[0])
        self._by_index = dict(indexed)
        return indexed

    def get_frame(self, frame_idx):
        """按帧号回捞原帧（事件证据用）；帧目录可随机读，代价低。"""
        path = self._by_index.get(int(frame_idx))
        if path is None:
            return None
        raw = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)

    def packets(self):
        """产出 (frame_idx, FramePacket)。frame_idx 用文件名编号（真值坐标），
        若编号不从任意值开始也保持原值——评测匹配直接用该编号对 manifest。"""
        for position, (frame_idx, path) in enumerate(self._frames):
            raw = np.fromfile(str(path), dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                log.warning(f"帧解码失败，跳过：{path}")
                continue
            if self._clock is not None and hasattr(self._clock, "set_frame"):
                t = self._clock.set_frame(frame_idx)
            else:
                t = frame_idx / self.fps
            yield FramePacket(frame=frame, t=t, frame_idx=frame_idx, fps=self.fps)

    def close(self):
        pass
