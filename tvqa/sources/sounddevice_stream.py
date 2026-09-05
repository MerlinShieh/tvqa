# -*- coding: utf-8 -*-
"""声卡采集后端（采集卡音频口，Windows WASAPI 默认设备流）。

sounddevice 为可选依赖：缺失或无设备时构造抛错，由上层降级（run 模式可关音频检测）。
设备到位后在 config 里把 audio.backend 改成 sounddevice、device_name 填枚举到的名称即可。
"""

import numpy as np

from ..logging_setup import get_logger
from .base import AudioPacket

log = get_logger("sources.sounddevice")


def list_input_devices():
    import sounddevice as sd  # 懒加载
    return [(index, device["name"]) for index, device in enumerate(sd.query_devices())
            if device.get("max_input_channels", 0) > 0]


class SounddeviceAudioSource:
    def __init__(self, cfg_section):
        import sounddevice as sd  # 懒加载；缺依赖时此处抛 ImportError
        self.samplerate = int(cfg_section.get("samplerate", 48000))
        self.block = int(cfg_section.get("block_samples", 1024))
        device = cfg_section.get("device_name", "")
        self._stream = sd.InputStream(samplerate=self.samplerate, channels=1,
                                      dtype="float32", blocksize=self.block, device=device or None)
        self._seq = 0

    def packets(self):
        with self._stream:
            while True:
                chunk, overflowed = self._stream.read(self.block)
                if overflowed:
                    log.warning("音频输入溢出（丢块），测量可能受影响")
                samples = chunk[:, 0].astype(np.float32, copy=False)
                yield AudioPacket(samples=samples, t=self._seq * self.block / self.samplerate,
                                  samplerate=self.samplerate, seq=self._seq)
                self._seq += 1

    def close(self):
        try:
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass
