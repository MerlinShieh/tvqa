# -*- coding: utf-8 -*-
"""音频文件后端：imageio-ffmpeg 子进程把 mp4/m4a/wav 解码为单声道 float32 PCM，
按固定块长产出 AudioPacket（t = 样本偏移 / samplerate，与视频共用同一零点）。"""

import subprocess

import numpy as np

from ..logging_setup import get_logger
from .base import AudioPacket

log = get_logger("sources.audio_file")

DEFAULT_SAMPLERATE = 48000
DEFAULT_BLOCK = 1024  # 约 21ms @48k，兼顾 onset 时间分辨率与吞吐


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("缺少 ffmpeg 支持：pip install imageio-ffmpeg") from error


class AudioFileSource:
    def __init__(self, cfg_section):
        self.path = cfg_section.get("path") or cfg_section.get("audio_file") or cfg_section.get("video_file")
        self.samplerate = int(cfg_section.get("samplerate", DEFAULT_SAMPLERATE))
        self.block = int(cfg_section.get("block_samples", DEFAULT_BLOCK))

    def decode_all(self):
        """一次性解码为整段 float32 单声道（评测短片够用，实现最简）。"""
        ffmpeg = _ffmpeg_exe()
        command = [ffmpeg, "-v", "error", "-i", str(self.path), "-vn",
                   "-ac", "1", "-ar", str(self.samplerate), "-f", "f32le", "-"]
        process = subprocess.run(command, capture_output=True, timeout=120)
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg 解码音频失败：{process.stderr.decode(errors='ignore')[:300]}")
        return np.frombuffer(process.stdout, dtype=np.float32)

    def packets(self):
        samples = self.decode_all()
        seq = 0
        for start in range(0, len(samples), self.block):
            chunk = samples[start:start + self.block]
            yield AudioPacket(samples=chunk, t=start / self.samplerate, samplerate=self.samplerate, seq=seq)
            seq += 1

    def close(self):
        pass
