# -*- coding: utf-8 -*-
"""卡顿检测器：原脚本 StutterDetector 的帧数空间移植。

保留核心规则骨架（运动前置 arming → 连续重复触发 + 窗口间歇触发 →
事件维持窗 → 静默恢复 + 结束时间反推），把所有「秒」参数改为「帧」参数
（对评测帧目录与恒定帧率采集都直接成立）；读取停顿阈值仅作为事件辅助
证据与统计量保留（默认不能单独建事件，与原策略一致）。
"""

from collections import deque

import cv2
import numpy as np

from .base import Detector


class _MotionTracker:
    def __init__(self, required_frames, break_tolerance, arm_timeout):
        self.required = required_frames
        self.break_tol = break_tolerance
        self.arm_timeout = arm_timeout
        self.start = None
        self.last = None
        self.armed = False

    def update(self, has_motion, frame_idx):
        if has_motion:
            if self.start is None:
                self.start = frame_idx
            self.last = frame_idx
            if frame_idx - self.start >= self.required:
                self.armed = True
        else:
            if self.last is None or frame_idx - self.last > self.break_tol:
                self.start = None

    def expire(self, frame_idx):
        if self.armed and self.last is not None and frame_idx - self.last > self.arm_timeout:
            self.armed = False

    def reset(self):
        self.start = None
        self.last = None
        self.armed = False


class StutterDetector(Detector):
    event_type = "stutter"

    def __init__(self, cfg, session=None, dataset=""):
        super().__init__(cfg, session, dataset)
        self.every_n = max(1, int(self.conf("analysis_every_n_frames", 1)))
        self.sig_w = int(self.conf("analysis_width", 320))
        self.sig_h = int(self.conf("analysis_height", 180))
        self.changed_pixel = int(self.conf("changed_pixel_threshold", 5))
        self.dup_mean = float(self.conf("duplicate_mean_diff", 2.5))
        self.dup_ratio = float(self.conf("duplicate_changed_ratio", 0.02))
        self.motion_mean = float(self.conf("motion_mean_diff", 2.8))
        self.motion_ratio = float(self.conf("motion_changed_ratio", 0.03))
        self.dup_trigger = int(self.conf("duplicate_trigger_frames", 4))
        self.dup_break_tol = int(self.conf("duplicate_break_tolerance_frames", 6))
        self.window_frames = int(self.conf("window_frames", 180))
        self.dup_ratio_th = float(self.conf("duplicate_ratio_threshold", 0.10))
        self.min_bursts = int(self.conf("min_duplicate_bursts", 2))
        self.active_bad_burst = int(self.conf("active_bad_burst_frames", 4))
        self.active_window = int(self.conf("active_window_frames", 180))
        self.active_dup_ratio_th = float(self.cfg.get("active_duplicate_ratio", self.dup_ratio_th))
        self.active_min_bursts = int(self.cfg.get("active_min_duplicate_bursts", self.min_bursts))
        self.quiet_frames = int(self.conf("recovery_quiet_frames", 45))
        self.dark_brightness = float(self.conf("dark_brightness_threshold", 25.0))
        self.dark_black_ratio = float(self.conf("dark_black_ratio", 0.95))
        self.white_flat_brightness = float(self.conf("white_flat_brightness", 235.0))
        self.white_flat_ratio = float(self.conf("white_flat_ratio", 0.95))
        self.read_stall_factor = float(self.cfg.get("read_interval_factor", 3.5))
        self.read_stall_can_trigger = bool(self.cfg.get("read_stall_can_trigger", False))
        self.serious_frames = int(self.conf("serious_stutter_frames", 90))

        self.previous = None
        self.prev_analysis_idx = None
        self.last_input_time = None
        self.max_read_interval = 0.0
        self.tracker = _MotionTracker(int(self.conf("required_motion_frames", 6)),
                                      int(self.conf("motion_break_tolerance_frames", 8)),
                                      int(self.conf("motion_arm_timeout_frames", 180)))
        self.dup_start = None
        self.dup_last = None
        self.window = deque()
        self._reset_event_state()
        self._last_packet_idx = 0
        self._fps = 30.0

    def _reset_event_state(self):
        self.in_event = False
        self.event_start = None
        self.last_confirmed = None
        self.active_samples = deque()
        self.max_dup_ratio = 0.0
        self.max_bursts = 0
        self.trigger_type = ""

    # ---------- 帧度量 ----------
    def _signature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        black_ratio = float(np.mean(gray < 30))
        sig = cv2.resize(gray, (self.sig_w, self.sig_h), interpolation=cv2.INTER_AREA)
        sig = cv2.GaussianBlur(sig, (3, 3), 0)
        return sig, brightness, black_ratio

    def _compare(self, prev, cur):
        diff = cv2.absdiff(prev, cur)
        mean_diff = float(diff.mean())
        changed_ratio = float(np.mean(diff >= self.changed_pixel))
        is_duplicate = mean_diff <= self.dup_mean and changed_ratio <= self.dup_ratio
        has_motion = not is_duplicate and (mean_diff >= self.motion_mean or changed_ratio >= self.motion_ratio)
        return is_duplicate, has_motion, mean_diff, changed_ratio

    # ---------- 主入口 ----------
    def process_frame(self, packet):
        frame_idx = packet.frame_idx
        self._last_packet_idx = frame_idx
        self._fps = packet.fps or 30.0
        read_interval = 0.0
        if self.last_input_time is not None:
            read_interval = max(0.0, packet.t - self.last_input_time)
            self.max_read_interval = max(self.max_read_interval, read_interval)
        self.last_input_time = packet.t
        read_stall = read_interval > self.read_stall_factor / max(self._fps, 1e-6)

        if frame_idx % self.every_n != 0:
            return []

        signature, brightness, black_ratio = self._signature(packet.frame)
        white_ratio = float(np.mean(signature > self.white_flat_brightness - 10))
        events = []

        # 极性平坦帧守卫：黑屏/白屏的纯色帧天然是精确重复帧，若不排除会与
        # luma 检测器同源双报（黑屏事件 + 卡顿事件）。白屏与黑屏同等处理。
        is_flat = ((brightness <= self.dark_brightness and black_ratio >= self.dark_black_ratio)
                   or (brightness >= self.white_flat_brightness and white_ratio >= self.white_flat_ratio))
        if is_flat:
            if self.in_event:
                events.append(self._finish_event(self.prev_analysis_idx if self.prev_analysis_idx else frame_idx,
                                                 note_close="dark_frame"))
            self._reset_analysis_state()
            return events

        if self.previous is None:
            self.previous = signature
            self.prev_analysis_idx = frame_idx
            return events

        is_duplicate, has_motion, mean_diff, changed_ratio = self._compare(self.previous, signature)
        self.tracker.update(has_motion, frame_idx)
        if not self.in_event:
            self.tracker.expire(frame_idx)

        # 连续重复跟踪（带短暂中断容忍）
        break_is_short = False
        if is_duplicate:
            if self.dup_start is None:
                self.dup_start = self.prev_analysis_idx
            self.dup_last = frame_idx
            dup_duration = frame_idx - self.dup_start
        else:
            break_is_short = (self.dup_start is not None and self.dup_last is not None
                              and frame_idx - self.dup_last <= self.dup_break_tol)
            if not break_is_short:
                self.dup_start = None
                self.dup_last = None
        effective_duplicate = is_duplicate or break_is_short
        dup_duration = (frame_idx - self.dup_start) if self.dup_start is not None else 0

        # 触发窗口
        self.window.append((frame_idx, effective_duplicate))
        while self.window and self.window[0][0] < frame_idx - self.window_frames:
            self.window.popleft()
        dup_ratio, bursts = self._window_stats(self.window)

        # 触发判定（需要运动前置）
        if not self.in_event and self.tracker.armed:
            trigger = None
            suggested_start = None
            continuous = dup_duration >= self.dup_trigger and self.dup_start is not None
            window_ok = (len(self.window) >= self.window_frames // 2 and dup_ratio >= self.dup_ratio_th
                         and bursts >= self.min_bursts)
            read_ok = self.read_stall_can_trigger and read_stall
            if continuous:
                trigger, suggested_start = "CONTINUOUS_DUPLICATE_FRAMES", self.dup_start
            elif window_ok:
                trigger, suggested_start = "REPEATED_FRAME_BURSTS", self.window[0][0]
            elif read_ok:
                trigger, suggested_start = "CAPTURE_READ_STALL", frame_idx
            if trigger:
                self._start_event(suggested_start, trigger, dup_ratio, bursts)

        # 事件维持与恢复
        if self.in_event:
            self.max_dup_ratio = max(self.max_dup_ratio, dup_ratio)
            self.max_bursts = max(self.max_bursts, bursts)
            self.active_samples.append((frame_idx, effective_duplicate, is_duplicate))
            while self.active_samples and self.active_samples[0][0] < frame_idx - self.active_window:
                self.active_samples.popleft()
            active_ratio, active_bursts = self._window_stats(self.active_samples)
            self.max_dup_ratio = max(self.max_dup_ratio, active_ratio)
            self.max_bursts = max(self.max_bursts, active_bursts)

            active_continuous = is_duplicate and dup_duration >= self.active_bad_burst
            active_window_bad = (len(self.active_samples) >= self.active_window // 2
                                 and active_ratio >= self.active_dup_ratio_th
                                 and active_bursts >= self.active_min_bursts)
            if active_continuous or active_window_bad or read_stall:
                latest_evidence = frame_idx
                if active_window_bad and self.active_samples:
                    # 只有「直接重复」帧才算明确证据；容忍间隙（break_is_short）
                    # 只服务触发窗口统计，不得把恢复时间往后拖。
                    duplicates = [f for f, _eff, direct in self.active_samples if direct]
                    if duplicates:
                        latest_evidence = max(latest_evidence, duplicates[-1])
                # 运动新鲜度上限：静止场景的像素重复与冻结在像素上不可分，
                # 用「最后运动帧 + 武装超时」封顶，防止静止段把事件无限拖长。
                if self.tracker.last is not None:
                    latest_evidence = min(latest_evidence, self.tracker.last + self.tracker.arm_timeout)
                if self.last_confirmed is None or latest_evidence > self.last_confirmed:
                    self.last_confirmed = latest_evidence

            quiet = frame_idx - self.last_confirmed if self.last_confirmed is not None else 0
            if quiet >= self.quiet_frames:
                events.append(self._finish_event(self.last_confirmed))
                self.tracker.reset()
                self.window.clear()
                self.dup_start = None
                self.dup_last = None

        self.previous = signature
        self.prev_analysis_idx = frame_idx
        return events

    @staticmethod
    def _window_stats(samples):
        if not samples:
            return 0.0, 0
        dup_count = sum(1 for sample in samples if sample[1])
        bursts = 0
        prev_dup = False
        for sample in samples:
            dup = sample[1]
            if dup and not prev_dup:
                bursts += 1
            prev_dup = dup
        return dup_count / len(samples), bursts

    def _reset_analysis_state(self):
        self.previous = None
        self.prev_analysis_idx = None
        self.tracker.reset()
        self.window.clear()
        self.dup_start = None
        self.dup_last = None
        self._reset_event_state()

    def reset_for_new_dataset(self):
        self._reset_analysis_state()
        self.last_input_time = None
        self.max_read_interval = 0.0

    # ---------- 事件生命周期 ----------
    def _start_event(self, suggested_start, trigger, dup_ratio, bursts):
        self.in_event = True
        self.event_start = suggested_start
        self.last_confirmed = suggested_start
        self.active_samples.clear()
        self.trigger_type = trigger
        self.max_dup_ratio = dup_ratio
        self.max_bursts = bursts

    def _finish_event(self, end_frame, note_close="recovered"):
        """结束时间=最后一次明确卡顿帧；持续=结束-开始。"""
        start = self.event_start if self.event_start is not None else end_frame
        end = max(end_frame, start)
        duration = end - start + 1
        level = "SERIOUS" if duration >= self.serious_frames else "SUSPECT"
        event = self.make_event(event_type="stutter", start_frame=start, end_frame=end,
                                start_t=start / self._fps, end_t=(end + 1) / self._fps,
                                level=level, trigger=self.trigger_type, status="confirmed",
                                max_duplicate_ratio=round(self.max_dup_ratio, 4),
                                max_duplicate_bursts=self.max_bursts,
                                max_read_interval_ms=round(self.max_read_interval * 1000, 1),
                                close_reason=note_close)
        self._reset_event_state()
        self.active_samples.clear()
        return event

    def flush(self):
        if self.in_event and self.event_start is not None:
            end = self.last_confirmed if self.last_confirmed is not None else self._last_packet_idx
            return [self._finish_event(end, note_close="stream_end")]
        return []
