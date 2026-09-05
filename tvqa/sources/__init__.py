# -*- coding: utf-8 -*-
"""输入后端注册表：backend 名称 → 类。新增后端只改这里，上层零感知。"""

from .base import AudioPacket, FramePacket, StreamMeta
from .frames_dir import FramesDirSource
from .video_file import VideoFileSource
from .audio_file import AudioFileSource
from .capture_card import CaptureCardSource

VIDEO_BACKENDS = {
    "frames_dir": FramesDirSource,
    "video_file": VideoFileSource,
    "capture_card": CaptureCardSource,
}


def create_video_source(cfg_section, clock):
    backend = cfg_section.get("backend", "frames_dir")
    if backend not in VIDEO_BACKENDS:
        raise ValueError(f"未知视频后端：{backend}，可选 {sorted(VIDEO_BACKENDS)}")
    return VIDEO_BACKENDS[backend](cfg_section, clock)


def create_audio_source(cfg_section):
    backend = cfg_section.get("backend", "audio_file")
    if backend == "audio_file":
        return AudioFileSource(cfg_section)
    if backend == "sounddevice":
        from .sounddevice_stream import SounddeviceAudioSource  # 可选依赖，懒加载
        return SounddeviceAudioSource(cfg_section)
    raise ValueError(f"未知音频后端：{backend}")


__all__ = ["FramePacket", "AudioPacket", "StreamMeta", "create_video_source",
           "create_audio_source", "VIDEO_BACKENDS"]
