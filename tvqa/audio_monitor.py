# -*- coding: utf-8 -*-
"""现场模式音频监视线程：采集卡音频口的信号丢失检测（AUDIO_DROPOUT）。

最小可用实现：后台线程持续读音频块，RMS 持续低于阈值超过 N 秒判信号丢失，
与视频事件共用同一归档会话（type=audio_dropout）。音画偏移测量（L1/L3）
由 avsync 模块单独承担；此线程只管「有没有声」。
"""

import threading

from .logging_setup import get_logger
from .sources import create_audio_source

log = get_logger("audio_monitor")


class AudioMonitor:
    def __init__(self, cfg_section, session, drop_seconds=2.0):
        self.source = create_audio_source(cfg_section)
        self.session = session
        self.drop_seconds = float(drop_seconds)
        self.rms_floor = float(cfg_section.get("rms_floor", 0.003))
        self._running = False
        self._thread = None
        self._seq = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="audio-monitor")
        self._thread.start()

    def _loop(self):
        import time
        silent_since = None
        while self._running:
            try:
                for packet in self.source.packets():
                    if not self._running:
                        break
                    rms = float((packet.samples ** 2).mean()) ** 0.5
                    now = packet.t
                    if rms < self.rms_floor:
                        if silent_since is None:
                            silent_since = now
                        elif now - silent_since >= self.drop_seconds:
                            self._emit_dropout(now - silent_since)
                            silent_since = now  # 避免重复连发
                    else:
                        silent_since = None
                    self._seq += 1
                break  # 文件源读完即退出；声卡源 packets() 为无限流
            except Exception as error:  # noqa: BLE001
                log.warning(f"音频监视异常（不影响视频链路）：{type(error).__name__}: {error}")
                break

    def _emit_dropout(self, seconds):
        event = {"type": "audio_dropout", "dataset": "live", "start_frame": 0, "end_frame": 0,
                 "start_t_sec": 0.0, "end_t_sec": 0.0, "duration_sec": round(seconds, 2),
                 "level": "SUSPECT", "trigger": "AUDIO_SILENCE", "status": "confirmed",
                 "metrics": {"silent_seconds": round(seconds, 2)}}
        try:
            self.session.record_event(event)
            log.info(f"音频信号丢失 {seconds:.1f}s，已记录 audio_dropout 事件")
        except Exception as error:  # noqa: BLE001
            log.warning(f"audio_dropout 归档失败：{error}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.source.close()
