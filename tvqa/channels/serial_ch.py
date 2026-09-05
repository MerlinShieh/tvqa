# -*- coding: utf-8 -*-
"""串口通道：控制面主路径 + 内核日志源（方案 v1.1 §3.7）。

- PyserialChannel：真实串口（pyserial 可选依赖，懒加载）。后台线程持续读日志；
  控制命令 = 写入 + 等待回显/prompt（弱约定，机型差异进配置）。
- MockSerialChannel：无硬件时的仿真 console——命令返回预置应答，可编程注入
  内核日志事件（HDMI 重训练、背光异常等），用于归因链路的纯软件验证。

约定接口：
  start()/stop()          生命周期
  read_lines()            取走新到的日志行（每行一条 str）
  send_command(cmd, timeout) -> list[str]   控制命令应答
  inject_kernel_line(text)  仅 mock：注入一条内核日志
"""

import re
import threading
import time
from collections import deque

from ..logging_setup import get_logger

log = get_logger("channels.serial")


class SerialChannel:
    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def read_lines(self):
        raise NotImplementedError

    def send_command(self, command, timeout=3.0):
        raise NotImplementedError

    @property
    def alive(self):
        return False

    def inject_kernel_line(self, text):
        raise NotImplementedError("仅 mock 通道支持注入")


class PyserialChannel(SerialChannel):
    """真实串口。pyserial 缺失/端口打不开时构造抛错，由上层降级。"""

    PROMPT_RE = re.compile(r"[#$]\s*$")  # 弱约定：常见 shell prompt 结尾

    def __init__(self, cfg_section):
        import serial  # 懒加载可选依赖
        self.port = cfg_section.get("port", "COM3")
        self.baudrate = int(cfg_section.get("baudrate", 115200))
        self.timeout = float(cfg_section.get("timeout", 1.0))
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self._buffer = deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="serial-reader")
        self._thread.start()

    def _read_loop(self):
        partial = b""
        while self._running:
            try:
                chunk = self._serial.read(4096)
                if chunk:
                    partial += chunk
                    while b"\n" in partial:
                        line, partial = partial.split(b"\n", 1)
                        with self._lock:
                            self._buffer.append(line.decode(errors="replace").rstrip())
            except Exception as error:  # noqa: BLE001
                log.warning(f"串口读取异常：{error}")
                time.sleep(0.5)
        if partial:
            with self._lock:
                self._buffer.append(partial.decode(errors="replace"))

    def read_lines(self):
        with self._lock:
            lines = list(self._buffer)
            self._buffer.clear()
        return lines

    def send_command(self, command, timeout=3.0):
        """写入命令并等待回显直到 prompt 或超时。"""
        self._serial.write((command + "\n").encode())
        self._serial.flush()
        deadline = time.time() + timeout
        echo = []
        while time.time() < deadline:
            lines = self.read_lines()
            echo.extend(lines)
            if lines and self.PROMPT_RE.search(lines[-1]):
                break
            time.sleep(0.05)
        return echo

    @property
    def alive(self):
        return self._running and self._serial.is_open

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self._serial.close()
        except Exception:  # noqa: BLE001
            pass

    def inject_kernel_line(self, text):
        log.debug("真实串口不支持注入，忽略")


class MockSerialChannel(SerialChannel):
    """仿真 console：可编程应答与内核日志注入（归因链路联调用）。"""

    DEFAULT_RESPONSES = {
        "echo adbd-on": ["adbd 已启动", "# "],
        "reboot": ["broadcast: reboot requested", "# "],
    }

    def __init__(self, cfg_section=None):
        self._buffer = deque()
        self._responses = dict(self.DEFAULT_RESPONSES)
        self._running = False

    def set_response(self, command, lines):
        self._responses[command] = list(lines) + ["# "]

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def read_lines(self):
        lines = list(self._buffer)
        self._buffer.clear()
        return lines

    def send_command(self, command, timeout=3.0):
        for key, response in self._responses.items():
            if key in command:
                self._buffer.extend(response)
                return list(response)
        self._buffer.append(f"mock-console: unknown command '{command}'")
        return [f"mock-console: unknown command '{command}'"]

    @property
    def alive(self):
        return self._running

    def inject_kernel_line(self, text):
        self._buffer.append(f"[kernel] {text}")


def create_serial_channel(cfg_section):
    backend = cfg_section.get("backend", "mock")
    if backend == "pyserial":
        return PyserialChannel(cfg_section)
    if backend == "mock":
        return MockSerialChannel(cfg_section)
    raise ValueError(f"未知串口后端：{backend}")
