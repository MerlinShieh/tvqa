# -*- coding: utf-8 -*-
"""DeviceProbe：串口 + ADB 双通道系统信号采集（方案 §3.7）。

- 后台线程轮询：串口日志行、logcat 流、周期 dumpsys 快照、date 校时。
- 信号分类为统一 SystemSignal(t, source, kind, text)，入环形缓冲（5 分钟）。
- 事件发生时 get_window(t0, t1) 取切片 → correlate 归因 + 归档 system_logs/。
- ADB 断连记 ADB_LOSS 信号（结合串口日志可区分电视重启 vs 链路问题）；
  任何探测故障只降级，不影响视觉检测主流程。
"""

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .channels import create_adb_channel, create_serial_channel
from .clock import ClockOffset
from .logging_setup import get_logger

log = get_logger("probe")

SIGNAL_PATTERNS = [
    (re.compile(r"HDMI.*(disconnect|unplug|remove)|hdmi.*cable", re.I), "HDMI_DISCONNECT"),
    (re.compile(r"HDMI.*(connect|plug|training|renegotiat)", re.I), "HDMI_RENEGOTIATION"),
    (re.compile(r"Choreographer.*Skipped\s+(\d+)\s+frames", re.I), "UI_JANK"),
    (re.compile(r"(MediaCodec|OMX|C2|Codec).*?(error|fail)", re.I), "CODEC_ERROR"),
    (re.compile(r"AudioTrack.*underrun|underrun", re.I), "AUDIO_UNDERRUN"),
    (re.compile(r"(panic|Oops|BUG:)", re.I), "KERNEL_PANIC"),
    (re.compile(r"(backlight|pwm).*(err|fail)", re.I), "BACKLIGHT_ERROR"),
]

POLL_INTERVAL = 1.0
SNAPSHOT_INTERVAL = 1.0
CLOCK_INTERVAL = 30.0
RING_SECONDS = 300.0


@dataclass
class SystemSignal:
    t: float                    # PC monotonic 轴
    source: str                 # serial | adb | snapshot
    kind: str                   # 分类标签或 "raw"
    text: str
    extra: dict = field(default_factory=dict)


class DeviceProbe:
    def __init__(self, cfg, clock=None):
        self.cfg = cfg
        self._clock = clock
        self.serial = create_serial_channel(cfg.section("serial"))
        self.adb = create_adb_channel(cfg.section("adb"))
        self.offset = ClockOffset()
        self.ring = deque(maxlen=4000)
        self._running = False
        self._thread = None
        self._last_snapshot = 0.0
        self._last_clock_pair = 0.0
        self._adb_was_alive = False
        self._lock = threading.Lock()

    # ---------- 生命周期 ----------
    def start(self):
        self.serial.start()
        self.adb.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="device-probe")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self.serial.stop()
        self.adb.stop()

    # ---------- 主循环 ----------
    def _loop(self):
        while self._running:
            now = self._clock.now() if self._clock else time.monotonic()
            try:
                self._collect(now)
            except Exception as error:  # noqa: BLE001
                log.warning(f"probe 采集异常（不影响检测主流程）：{type(error).__name__}: {error}")
            time.sleep(POLL_INTERVAL)

    def _collect(self, now):
        # 1) 串口日志 → 分类
        for line in self.serial.read_lines():
            self._classify_and_push(now, "serial", line)
        # 2) ADB logcat → 分类
        for line in self.adb.read_logcat_lines():
            self._classify_and_push(now, "adb", line)
        # 3) ADB 存活监测
        alive = self.adb.alive or bool(self.adb.shell("echo ok", timeout=2.0)[1])
        if self._adb_was_alive and not alive:
            self._push(SystemSignal(now, "adb", "ADB_LOSS", "adb 通道失联"))
        self._adb_was_alive = alive
        # 4) 周期快照
        if now - self._last_snapshot >= SNAPSHOT_INTERVAL and alive:
            self._last_snapshot = now
            for command in ("dumpsys display", "dumpsys audio"):
                text, ok = self.adb.shell(command, timeout=2.0)
                if ok:
                    self._push(SystemSignal(now, "snapshot", command, text[:400]))
        # 5) 校时
        if now - self._last_clock_pair >= CLOCK_INTERVAL and alive:
            self._last_clock_pair = now
            text, ok = self.adb.shell("date +%s.%N", timeout=2.0)
            if ok:
                try:
                    self.offset.add_sample(time.time(), float(text.strip()))
                except ValueError:
                    pass

    def _classify_and_push(self, now, source, line):
        kind = "raw"
        for pattern, label in SIGNAL_PATTERNS:
            match = pattern.search(line)
            if match:
                kind = label
                break
        self._push(SystemSignal(now, source, kind, line))

    def _push(self, signal):
        with self._lock:
            self.ring.append(signal)
            oldest = signal.t - RING_SECONDS
            while self.ring and self.ring[0].t < oldest:
                self.ring.popleft()

    # ---------- 查询与归档 ----------
    def get_window(self, t_start, t_end, kinds=None):
        with self._lock:
            signals = [s for s in self.ring if t_start <= s.t <= t_end]
        if kinds:
            signals = [s for s in signals if s.kind in kinds]
        return signals

    def write_evidence_slice(self, session, event_dir, t_start, t_end):
        """把事件时间窗内的系统信号按通道写切片文件，返回 {source: 相对路径}。"""
        signals = self.get_window(t_start, t_end)
        paths = {}
        for source in ("serial", "adb", "snapshot"):
            rows = [s for s in signals if s.source == source]
            if not rows:
                continue
            target = event_dir / "system_logs" / f"{source}_slice.log"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(f"{s.t:.3f} [{s.kind}] {s.text}" for s in rows), encoding="utf-8")
            paths[source] = session.rel(target)
        return paths

    @property
    def alive(self):
        return self._running
