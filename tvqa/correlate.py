# -*- coding: utf-8 -*-
"""EventCorrelator：视觉事件 × 系统信号 → 归因结论（方案 §3.5 规则表）。

输入：事件时间窗 + DeviceProbe 信号切片；输出 (attribution, evidence)。
规则按优先级从上到下命中即止；无任何系统信号证据时归 "UNATTRIBUTED"。
真实/ mock 通道同构——eval 模式用注入信号即可验证全部规则分支。
"""

from .logging_setup import get_logger

log = get_logger("correlate")

LUMA_KINDS = {"black", "white", "flicker"}

RULES = [
    # (事件类型集合, 需要的信号 kind, 归因结论)
    (LUMA_KINDS, {"HDMI_DISCONNECT", "HDMI_RENEGOTIATION"}, "SIGNAL_LINK_LOSS"),
    (LUMA_KINDS, {"KERNEL_PANIC"}, "SYSTEM_CRASH"),
    (LUMA_KINDS, {"BACKLIGHT_ERROR"}, "TV_FIRMWARE_DISPLAY"),
    ({"stutter"}, {"UI_JANK", "CODEC_ERROR"}, "TV_RENDER_DECODE"),
    ({"stutter"}, {"HDMI_RENEGOTIATION"}, "SIGNAL_RENEGOTIATION"),
    ({"tear"}, {"HDMI_RENEGOTIATION", "HDMI_DISCONNECT"}, "SIGNAL_LINK_TIMING"),
    ({"tear"}, {"CODEC_ERROR"}, "TV_DECODE_CORRUPTION"),
    ({"local_freeze", "blocking"}, {"CODEC_ERROR"}, "TV_DECODE_CORRUPTION"),
    ({"av_offset"}, {"AUDIO_UNDERRUN"}, "TV_AUDIO_PIPELINE"),
]


def attribute(event, signals, window_before=10.0, window_after=5.0):
    """返回 (attribution, evidence_texts)。signals: SystemSignal 列表。"""
    t0 = event.get("start_t_sec", 0) - window_before
    t1 = event.get("end_t_sec", 0) + window_after
    in_window = [s for s in signals if t0 <= s.t <= t1 and s.kind != "raw"]
    event_types = {event.get("type", "")}
    for types, needed, conclusion in RULES:
        if event_types & types and in_window and {s.kind for s in in_window} & needed:
            evidence = [f"[{s.kind}@{s.t:.3f}] {s.text[:160]}" for s in in_window]
            return conclusion, evidence
    if in_window:
        evidence = [f"[{s.kind}@{s.t:.3f}] {s.text[:160]}" for s in in_window]
        return "SYSTEM_SIGNALS_PRESENT", evidence
    return "UNATTRIBUTED", []
