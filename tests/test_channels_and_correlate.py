# -*- coding: utf-8 -*-
"""系统信号通道与归因单测：mock 串口/ADB → 信号分类 → 归因规则。"""

from tvqa.channels import MockAdbChannel, MockSerialChannel
from tvqa.correlate import attribute


class _FakeClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t


def _build_probe_with(monkeypatch, serial_signals=(), adb_signals=()):
    from tvqa.config import Cfg
    from tvqa.probe import DeviceProbe

    cfg = Cfg({"serial": {"backend": "mock"}, "adb": {"backend": "mock"}})
    probe = DeviceProbe(cfg, clock=_FakeClock())
    # 直接注入信号（不启动线程，避免 sleep）
    for text in serial_signals:
        probe._classify_and_push(1000.0, "serial", text)
    for text in adb_signals:
        probe._classify_and_push(1000.5, "adb", text)
    return probe


def test_signal_classification():
    probe = _build_probe_with(None,
                              serial_signals=("hdmi cable disconnected, retraining link",),
                              adb_signals=("Choreographer: Skipped 45 frames!",))
    kinds = {s.kind for s in probe.ring}
    assert "HDMI_DISCONNECT" in kinds or "HDMI_RENEGOTIATION" in kinds
    assert "UI_JANK" in kinds


def test_attribute_stutter_with_jank():
    probe = _build_probe_with(None, adb_signals=("Choreographer: Skipped 60 frames!",))
    event = {"type": "stutter", "start_t_sec": 1000.0, "end_t_sec": 1001.0}
    attribution, evidence = attribute(event, probe.get_window(900, 1100))
    assert attribution == "TV_RENDER_DECODE"
    assert evidence


def test_attribute_black_with_hdmi_loss():
    probe = _build_probe_with(None, serial_signals=("hdmitx: cable disconnected",))
    event = {"type": "black", "start_t_sec": 1000.0, "end_t_sec": 1005.0}
    attribution, _ = attribute(event, probe.get_window(900, 1100))
    assert attribution == "SIGNAL_LINK_LOSS"


def test_attribute_unattributed_when_no_signals():
    probe = _build_probe_with(None)
    event = {"type": "stutter", "start_t_sec": 1000.0, "end_t_sec": 1001.0}
    attribution, evidence = attribute(event, probe.get_window(900, 1100))
    assert attribution == "UNATTRIBUTED"
    assert evidence == []


def test_mock_channels_roundtrip():
    serial = MockSerialChannel({})
    adb = MockAdbChannel({})
    serial.start(); adb.start()
    assert "adbd" in "".join(serial.send_command("echo adbd-on"))
    adb.inject_logcat("AudioTrack", "underrun: 12 frames")
    lines = adb.read_logcat_lines()
    assert any("underrun" in line for line in lines)
    text, ok = adb.shell("dumpsys display")
    assert ok and "HDMI" in text
    serial.stop(); adb.stop()
