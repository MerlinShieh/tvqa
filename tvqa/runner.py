# -*- coding: utf-8 -*-
"""现场模式 runner：采集卡视频 + 声卡音频 + 串口/ADB 探针 → 检测 → 归因 → 归档。

与 eval 的差别：
- RealClock 实时驱动；证据从「环形帧缓冲」回捞（采集卡无法回看历史帧）；
- 每个事件闭合即做 correlate 归因 + 系统日志切片归档；
- 支持 OpenCV 实时预览窗（Q/ESC 退出），Ctrl+C 优雅结算（flush 未闭合事件）。
"""

import time
from collections import deque

import cv2

from .archive import Session
from .clock import RealClock
from .correlate import attribute
from .detectors import build_visual_detectors
from .logging_setup import get_logger, setup_logging
from .probe import DeviceProbe
from .utils import save_image

log = get_logger("runner")


class _FrameRing:
    """证据回捞用的帧缓冲（存缩放副本，控内存）。"""

    def __init__(self, maxlen, store_width=852):
        self._buf = deque(maxlen=maxlen)
        self._store_width = store_width
        self._last = None

    def add(self, packet):
        frame = packet.frame
        if frame.shape[1] > self._store_width:
            scale = self._store_width / frame.shape[1]
            frame = cv2.resize(frame, (self._store_width, int(frame.shape[0] * scale)))
        self._last = (packet.frame_idx, frame)
        self._buf.append(self._last)

    def get(self, frame_idx):
        for idx, frame in reversed(self._buf):
            if idx == frame_idx:
                return frame
        return None


def run_field(cfg, session):
    """现场模式主循环。返回事件总数。"""
    from .sources import create_video_source
    counting = setup_logging(session.logs_dir / "tvqa.log",
                             cfg.get("log.console_level"), cfg.get("log.file_level"))
    session.set_log_handler(counting)
    probe = DeviceProbe(cfg)
    probe.start()
    clock = RealClock()
    try:
        source = create_video_source(cfg.section("video"), clock)
    except Exception as error:  # noqa: BLE001
        log.error(f"视频源打开失败，现场模式退出：{type(error).__name__}: {error}")
        probe.stop()
        session.finish({"events": 0, "exit": "source_error"})
        return 0
    detectors = build_visual_detectors(cfg, session, dataset="live")
    ring = _FrameRing(maxlen=int(cfg.section("evidence").get("ring_buffer_frames", 90)),
                      store_width=int(cfg.section("evidence").get("frame_store_width", 852)))
    show_preview = bool(cfg.get("preview", False))
    event_count = 0
    audio_thread = _start_audio_side(cfg, session)  # 可选；失败仅告警

    log.info(f"现场模式启动：video={cfg.get('video.backend')} audio={cfg.get('audio.backend')} "
             f"serial={cfg.get('serial.backend')} adb={cfg.get('adb.backend')}")
    try:
        for packet in source.packets():
            ring.add(packet)
            for detector in detectors:
                for event in detector.safe_process(packet):
                    _settle_event(cfg, session, probe, ring, event)
                    event_count += 1
            if show_preview:
                if not _show_preview(packet, detectors):
                    break
    except KeyboardInterrupt:
        log.info("收到 Ctrl+C，开始结算未闭合事件……")
    except Exception as error:  # noqa: BLE001
        log.error(f"采集链路异常退出：{type(error).__name__}: {error}")
    finally:
        for detector in detectors:
            try:
                for event in detector.flush():
                    _settle_event(cfg, session, probe, ring, event)
                    event_count += 1
            except Exception as error:  # noqa: BLE001
                log.warning(f"[{detector.name()}] flush 异常：{error}")
        source.close()
        probe.stop()
        if audio_thread is not None:
            audio_thread.stop()
        cv2.destroyAllWindows()
        session.finish({"events": event_count, "exit": "settled"})
        log.info(f"现场模式退出，共记录事件 {event_count} 个；归档：{session.root}")
    return event_count


def _settle_event(cfg, session, probe, ring, event):
    """事件闭合统一处理：证据 → 归因 → 日志切片 → 归档。"""
    folder = f"{event['type']}_f{event['start_frame']:05d}-{event['end_frame']:05d}"
    ev_dir = session.evidence_dir(event.get("dataset", "live"), folder)
    saved = {}
    for point, idx in (("start", event["start_frame"]),
                       ("mid", (event["start_frame"] + event["end_frame"]) // 2),
                       ("end", event["end_frame"])):
        frame = ring.get(idx)
        if frame is not None and save_image(ev_dir / f"img_{point}_f{idx:05d}.jpg", frame):
            saved[point] = session.rel(ev_dir / f"img_{point}_f{idx:05d}.jpg")
    event["evidence_dir"] = session.rel(ev_dir)
    event["evidence_frames"] = saved
    if probe is not None:
        t0, t1 = event.get("start_t_sec", 0), event.get("end_t_sec", 0)
        signals = probe.get_window(t0 - 10, t1 + 5)
        attribution, evidence = attribute(event, signals)
        event["attribution"] = attribution
        event["attribution_evidence"] = evidence[:8]
        slices = probe.write_evidence_slice(session, ev_dir, t0 - 10, t1 + 5)
        if slices:
            event["system_log_slices"] = slices
    session.write_evidence_json(ev_dir, "event.json", event)
    log.info(f"事件 {event['type']} frames={event['start_frame']}..{event['end_frame']} "
             f"level={event['level']} attribution={event.get('attribution', 'N/A')}")
    session.record_event(event)


def _start_audio_side(cfg, session):
    """音频检测线程（现场模式可选）。失败仅告警降级，不影响视频链路。"""
    try:
        from .audio_monitor import AudioMonitor
    except ImportError:
        return None
    try:
        monitor = AudioMonitor(cfg.section("audio"), session)
        monitor.start()
        return monitor
    except Exception as error:  # noqa: BLE001
        log.warning(f"音频检测未启用：{error}")
        return None


def _show_preview(packet, detectors):
    display = packet.frame.copy()
    lines = [f"frame={packet.frame_idx} t={packet.t:.1f}s"]
    for detector in detectors:
        state = getattr(detector, "preview_state", lambda: {})()
        for key, value in state.items():
            lines.append(f"{key}: {value}")
    y = 40
    for line in lines:
        cv2.putText(display, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        y += 36
    cv2.imshow("tvqa live", display)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), ord("Q"), 27)
