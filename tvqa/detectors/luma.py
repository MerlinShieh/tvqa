# -*- coding: utf-8 -*-
"""黑屏 / 白屏 / 闪烁统一检测器。

统一在帧号空间工作（frame_idx 差值），t 用最近一次 packet.fps 换算，
因此评测帧目录与现场采集对检测器完全同构。

- 黑/白屏：镜像状态机（候选 → 连续 confirm_frames 帧确认 → 跟踪 →
  连续 end_tolerance 帧不满足即闭合）。逐帧发现，比原脚本 1s 采样更灵敏。
- 闪烁：滑窗亮度中轴符号翻转率。活动期持续满足「窗口内翻转数 ≥ 阈值」，
  回落超过 end_tolerance 帧即闭合。闪烁存续期间抑制黑/白候选（同源合并，
  避免黑闪同时报 black + flicker 双事件）。
- process_frame 每帧返回本次「闭合」的事件列表；flush 收尾闭合悬挂事件。
"""

from collections import deque

from .base import Detector


class _PolarityState:
    def __init__(self, kind, confirm_frames, end_tolerance, serious_frames):
        self.kind = kind
        self.confirm_frames = confirm_frames
        self.end_tolerance = end_tolerance
        self.serious_frames = serious_frames
        self.reset()

    def reset(self):
        self.candidate_start = None
        self.confirmed = False
        self.last_seen = None
        self.consecutive = 0

    @property
    def active(self):
        return self.candidate_start is not None


class LumaDetector(Detector):
    event_type = "luma"

    def __init__(self, cfg, session=None, dataset=""):
        super().__init__(cfg, session, dataset)
        self.thumb_w = int(self.conf("thumbnail_width", 85))
        self.thumb_h = int(self.conf("thumbnail_height", 48))
        self.black = _PolarityState("black", int(self.conf("black_confirm_frames", 4)),
                                    int(self.conf("black_end_tolerance_frames", 2)),
                                    int(self.conf("black_serious_frames", 600)))
        self.white = _PolarityState("white", int(self.conf("white_confirm_frames", 4)),
                                    int(self.conf("white_end_tolerance_frames", 2)),
                                    int(self.conf("white_serious_frames", 600)))
        self.black_pixel = int(self.conf("black_pixel_threshold", 30))
        self.black_brightness = float(self.conf("black_brightness_threshold", 25.0))
        self.black_ratio = float(self.conf("black_ratio_threshold", 0.95))
        self.white_pixel = int(self.conf("white_pixel_threshold", 225))
        self.white_brightness = float(self.conf("white_brightness_threshold", 235.0))
        self.white_ratio = float(self.conf("white_ratio_threshold", 0.95))
        self.flicker_window = int(self.conf("flicker_window_frames", 30))
        self.flicker_amp = float(self.conf("flicker_min_amplitude", 8.0))
        self.flicker_min_rate = float(self.conf("flicker_min_rate", 2.0))
        self.flicker_end_tolerance = int(self.conf("flicker_end_tolerance_frames", 9))
        self._reset_flicker_state()
        self._fps = 30.0

    def _reset_flicker_state(self):
        self._jump_frames = deque()
        self._prev_brightness = None
        self._flicker_start = None
        self._flicker_last_active = None
        self._flicker_peak_rate = 0.0

    # ---------- 度量 ----------
    def _measure(self, frame):
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (self.thumb_w, self.thumb_h), interpolation=cv2.INTER_AREA)
        brightness = float(thumb.mean())
        is_black = (brightness <= self.black_brightness
                    and float((thumb < self.black_pixel).mean()) >= self.black_ratio)
        is_white = (brightness >= self.white_brightness
                    and float((thumb > self.white_pixel).mean()) >= self.white_ratio)
        return brightness, is_black, is_white

    # ---------- 主入口 ----------
    def process_frame(self, packet):
        frame_idx = packet.frame_idx
        self._fps = packet.fps or 30.0
        brightness, is_black, is_white = self._measure(packet.frame)

        flicker_new = self._update_flicker(frame_idx, brightness, is_black, is_white)
        flicker_active = self._flicker_start is not None

        events = list(flicker_new)
        suppress = flicker_active
        if suppress:
            if self.black.active and not self.black.confirmed:
                self.black.reset()
            if self.white.active and not self.white.confirmed:
                self.white.reset()

        for state, detected in ((self.black, is_black), (self.white, is_white)):
            events.extend(self._step_polarity(state, detected, frame_idx, suppress))
        return events

    def _step_polarity(self, state, detected, frame_idx, suppressed):
        events = []
        if detected and not suppressed:
            if not state.active:
                state.candidate_start = frame_idx
                state.last_seen = frame_idx
                state.consecutive = 1
            else:
                # 确认要求「连续」满足；容忍间隙只用于已确认事件的闭合判定，
                # 防止黑闪（隔帧交替）靠累计跨度抢先确认为黑屏。
                state.consecutive = state.consecutive + 1 if frame_idx == state.last_seen + 1 else 1
                state.last_seen = frame_idx
            if not state.confirmed and state.consecutive >= state.confirm_frames:
                state.confirmed = True
        elif state.active:
            if frame_idx - state.last_seen > state.end_tolerance:
                if state.confirmed:
                    events.append(self._close_polarity(state, state.last_seen))
                state.reset()
        return events

    def _close_polarity(self, state, end_frame):
        start = state.candidate_start
        duration = end_frame - start + 1
        level = "SERIOUS" if duration >= state.serious_frames else "SUSPECT"
        event = self.make_event(event_type=state.kind, start_frame=start, end_frame=end_frame,
                                start_t=start / self._fps, end_t=(end_frame + 1) / self._fps,
                                level=level, trigger="CONTINUOUS_POLARITY", status="confirmed")
        state.reset()
        return event

    # ---------- 闪烁 ----------
    def _update_flicker(self, frame_idx, brightness, is_black=False, is_white=False):
        """亮度跳变率闪烁检测；返回本次闭合的闪烁事件（0 或 1 条）。

        一次「跳变」= 相邻帧平均亮度差 ≥ flicker_min_amplitude（明→暗或暗→明
        各计一次，与注入规则 flip_rate_per_sec 的口径一致：一个明暗周期 2 次跳变）。
        窗口内跳变数 / 窗口秒数 = 跳变率；≥ flicker_min_rate 且 ≥3 次判为活动。
        黑/白屏帧不计跳变（与极性检测同源合并；黑屏进出只有 1~2 次孤立跳变，
        达不到 ≥3 的门槛，天然不误报）。
        """
        closed = []
        jump = 0.0
        if self._prev_brightness is not None:
            jump = abs(brightness - self._prev_brightness)
        self._prev_brightness = brightness
        if jump >= self.flicker_amp and not is_black and not is_white:
            self._jump_frames.append(frame_idx)
        window_start = frame_idx - self.flicker_window
        while self._jump_frames and self._jump_frames[0] < window_start:
            self._jump_frames.popleft()

        n_jumps = len(self._jump_frames)
        rate = n_jumps / (self.flicker_window / self._fps)
        active_now = n_jumps >= 3 and rate >= self.flicker_min_rate

        if active_now:
            self._flicker_last_active = frame_idx
            self._flicker_peak_rate = max(self._flicker_peak_rate, rate)
            if self._flicker_start is None:
                self._flicker_start = max(window_start, self._jump_frames[0])
        elif self._flicker_start is not None and frame_idx - self._flicker_last_active > self.flicker_end_tolerance:
            closed.append(self._close_flicker(self._flicker_last_active))
        return closed

    def _close_flicker(self, end_frame):
        start = self._flicker_start
        event = self.make_event(event_type="flicker", start_frame=start, end_frame=end_frame,
                                start_t=start / self._fps, end_t=(end_frame + 1) / self._fps,
                                level="SUSPECT", trigger="HIGH_FLIP_RATE", status="confirmed",
                                measured_flip_rate_per_sec=round(self._flicker_peak_rate, 2))
        self._flicker_start = None
        self._flicker_peak_rate = 0.0
        return event

    # ---------- 收尾 ----------
    def flush(self):
        events = []
        for state in (self.black, self.white):
            if state.confirmed:
                events.append(self._close_polarity(state, state.last_seen))
            state.reset()
        if self._flicker_start is not None:
            events.append(self._close_flicker(self._flicker_last_active))
        return events
