# -*- coding: utf-8 -*-
"""ADB 通道（root）：框架层信号源 + UI 操作注入。

- RealAdbChannel：subprocess 调 adb。logcat -v epoch 后台流；shell 命令同步执行；
  周期 date 校时由 probe 负责。
- MockAdbChannel：无设备时的仿真——shell 返回预置输出，可编程注入
  logcat 行（Choreographer jank / MediaCodec error / AudioTrack underrun），
  供归因链路纯软件验证。

ADB 断连不抛出：shell 返回 (ok=False)，由 probe 记 ADB_LOSS 信号。
"""

import subprocess
import threading
import time
from collections import deque

from ..logging_setup import get_logger

log = get_logger("channels.adb")


class AdbChannel:
    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def shell(self, command, timeout=5.0):
        raise NotImplementedError

    def read_logcat_lines(self):
        raise NotImplementedError

    @property
    def alive(self):
        return False

    def inject_logcat(self, tag, text):
        raise NotImplementedError("仅 mock 通道支持注入")


class RealAdbChannel(AdbChannel):
    def __init__(self, cfg_section):
        self.device = cfg_section.get("device", "")
        self._base = ["adb"] + (["-s", self.device] if self.device else [])
        self._buffer = deque()
        self._lock = threading.Lock()
        self._running = False
        self._process = None
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._logcat_loop, daemon=True, name="adb-logcat")
        self._thread.start()

    def _logcat_loop(self):
        command = self._base + ["logcat", "-v", "epoch"]
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL)
        except Exception as error:  # noqa: BLE001
            log.warning(f"logcat 启动失败（现场无 ADB 时正常）：{error}")
            return
        for raw in self._process.stdout:
            if not self._running:
                break
            with self._lock:
                self._buffer.append(raw.decode(errors="replace").rstrip())
        self._process = None

    def shell(self, command, timeout=5.0):
        try:
            result = subprocess.run(self._base + ["shell", command], capture_output=True,
                                    text=True, timeout=timeout)
            return result.stdout, result.returncode == 0
        except Exception as error:  # noqa: BLE001
            log.debug(f"adb shell 失败：{error}")
            return "", False

    def read_logcat_lines(self):
        with self._lock:
            lines = list(self._buffer)
            self._buffer.clear()
        return lines

    @property
    def alive(self):
        return self._running and self._process is not None

    def stop(self):
        self._running = False
        if self._process:
            try:
                self._process.terminate()
            except Exception:  # noqa: BLE001
                pass
        if self._thread:
            self._thread.join(timeout=2)


class MockAdbChannel(AdbChannel):
    DEFAULT_SHELL = {
        "dumpsys display": "mDisplayState=ON\nHDMI: connected\nmHdmiOutput=1920x1080@60",
        "dumpsys audio": "players: 1 active\nAudioTrack latency=80ms",
        "dumpsys media.metrics": "video_skipped=0 audio_underrun=0",
        "date +%s.%N": "1736000000.500",
    }

    def __init__(self, cfg_section=None):
        self._buffer = deque()
        self._shell_outputs = dict(self.DEFAULT_SHELL)
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def set_shell_output(self, key, text):
        self._shell_outputs[key] = text

    def shell(self, command, timeout=5.0):
        for key, text in self._shell_outputs.items():
            if key in command:
                return text, True
        return "", False

    def read_logcat_lines(self):
        lines = list(self._buffer)
        self._buffer.clear()
        return lines

    def inject_logcat(self, tag, text):
        self._buffer.append(f"{tag}: {text}")

    @property
    def alive(self):
        return self._running

    def stop(self):
        self._running = False


def create_adb_channel(cfg_section):
    backend = cfg_section.get("backend", "mock")
    if backend == "real":
        return RealAdbChannel(cfg_section)
    if backend == "mock":
        return MockAdbChannel(cfg_section)
    raise ValueError(f"未知 ADB 后端：{backend}")
