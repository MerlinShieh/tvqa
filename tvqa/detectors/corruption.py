# -*- coding: utf-8 -*-
"""花屏 / 撕裂检测器（规则层）。

三项检测共享同一降采样灰度与分析网格：
1. tear（撕裂）：帧内水平断层 → 把画面按候选横线分成上/下两行带，
   分别估计两带相对上一帧的水平位移（列边缘剖面相关）；
   正常帧两带位移≈一致（同一内容运动），撕裂帧位移差显著 ≥ 阈。
   输出撕裂线位置（全分辨率像素）与最大位移差估计。
2. local_freeze（局部冻结/局部损坏矛盾）：块网格 diff 图上，
   全局运动活跃但存在连通「零变化」块区域 ≥ N 块且持续 ≥ M 帧。
3. blocking（马赛克块效应）：块网格边界梯度能量 / 块内梯度能量比值
   相对滑动基线的突增。
阈值全部来自配置；本阶段用注入数据集 + 合成样本验证功能链路。
"""

from collections import deque

import cv2
import numpy as np

from ..logging_setup import get_logger
from .base import Detector

log = get_logger("detectors.corruption")


class CorruptionDetector(Detector):
    event_type = "corruption"

    def __init__(self, cfg, session=None, dataset=""):
        super().__init__(cfg, session, dataset)
        self.small_w = int(self.conf("tear_downscale_width", 256))
        self.band_step = int(self.conf("tear_boundary_step", 16))
        self.shift_search = int(self.conf("tear_shift_search", 24))
        self.tear_min_diff = int(self.conf("tear_min_shift_diff", 4))
        self.tear_band_frames = int(self.conf("tear_band_height", 30))
        self.tear_min_event = int(self.conf("tear_min_event_frames", 3))
        self.tear_end_tol = int(self.conf("tear_end_tolerance_frames", 3))
        self.tear_max_y_spread = int(self.conf("tear_max_y_spread", 24))
        self.grid_cols = int(self.conf("block_grid_cols", 16))
        self.grid_rows = int(self.conf("block_grid_rows", 9))
        self.frozen_max_diff = float(self.conf("frozen_block_max_diff", 1.0))
        self.frozen_min_texture = float(self.conf("frozen_min_texture", 6.0))
        self.global_motion_min = float(self.conf("global_motion_min_ratio", 0.5))
        self.frozen_min_blocks = int(self.conf("frozen_min_blocks", 4))
        self.frozen_min_event = int(self.conf("frozen_min_event_frames", 3))
        self.blocking_grid = int(self.conf("blocking_grid", 8))
        self.blocking_min_span = float(self.conf("blocking_min_span", 0.25))

        self._prev_small = None
        self._prev_blocks = None
        self._frame_size = None
        # tear 事件跟踪
        self._tear_open = None
        self._tear_last = None
        # frozen 事件跟踪
        self._frozen_open = None
        self._frozen_last = None
        # blocking 基线
        self._blocking_history = deque(maxlen=150)
        self._blocking_open = None
        self._blocking_last = None
        self._last_frame_idx = 0
        self._fps = 30.0

    # ================= 主入口 =================
    def process_frame(self, packet):
        frame = packet.frame
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        self._last_frame_idx = packet.frame_idx
        self._fps = packet.fps or 30.0
        scale_h = int(round(h * self.small_w / w))
        small = cv2.cvtColor(cv2.resize(frame, (self.small_w, scale_h), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)

        events = []
        if self._prev_small is not None and self._prev_small.shape == small.shape:
            events.extend(self._detect_tear(small, packet))
            events.extend(self._detect_local_freeze(small, packet))
        else:
            # 分辨率变化：闭合悬挂事件
            events.extend(self._close_all())
        events.extend(self._detect_blocking(packet))
        self._prev_small = small
        return events

    def flush(self):
        return self._close_all()

    def _close_all(self):
        events = []
        if self._tear_open is not None:
            events.append(self._emit_tear(self._tear_open["start"], self._tear_last,
                                          self._tear_open["peak"], self._tear_open["y"]))
        if self._frozen_open is not None:
            events.append(self._emit_frozen(self._frozen_open["start"], self._frozen_last,
                                            self._frozen_open["blocks"]))
        if self._blocking_open is not None:
            events.append(self._emit_blocking(self._blocking_open["start"], self._blocking_last,
                                              self._blocking_open["peak"]))
        self._tear_open = self._frozen_open = self._blocking_open = None
        return events

    def reset_for_new_dataset(self):
        self._prev_small = None
        self._prev_blocks = None
        self._close_all()
        self._tear_spans.clear()
        self._blocking_history.clear()

    # ================= 撕裂 =================
    def _row_edge_profile(self, small, y0, y1):
        band = small[y0:y1, :]
        gx = np.abs(np.diff(band, axis=1, prepend=band[:, :1]))
        return gx.mean(axis=0)

    @staticmethod
    def _best_shift(profile_a, profile_b, search, min_cos=0.55):
        """profile_b 相对 profile_a 的最优水平位移（像素，降采样尺度）。"""
        a = profile_a - profile_a.mean()
        norm_a = np.linalg.norm(a)
        if norm_a < 1e-6:
            return 0, 0.0
        best_shift, best_cos = 0, -1.0
        for shift in range(-search, search + 1):
            b = np.roll(profile_b, shift)
            b = b - b.mean()
            denom = np.linalg.norm(b)
            if denom < 1e-6:
                continue
            cos = float(np.dot(a, b) / (norm_a * denom))
            if cos > best_cos:
                best_cos, best_shift = cos, shift
        return best_shift, best_cos

    def _detect_tear(self, small, packet):
        _, h = small.shape
        frame_idx = packet.frame_idx
        best = None  # (abs_shift_diff, y_small)
        for y in range(h // 6, h - h // 6, self.band_step):
            band_h = max(6, int(self.tear_band_frames * h / (self._frame_size[1] or h)))
            y_top0, y_top1 = max(0, y - band_h), y
            y_bot0, y_bot1 = y, min(h, y + band_h)
            if y_top1 - y_top0 < band_h - 1 or y_bot1 - y_bot0 < band_h - 1:
                continue
            m_top, c_top = self._best_shift(self._row_edge_profile(self._prev_small, y_top0, y_top1),
                                            self._row_edge_profile(small, y_top0, y_top1), self.shift_search)
            m_bot, c_bot = self._best_shift(self._row_edge_profile(self._prev_small, y_bot0, y_bot1),
                                            self._row_edge_profile(small, y_bot0, y_bot1), self.shift_search)
            if c_top < 0.65 or c_bot < 0.65:
                continue
            diff = abs(m_top - m_bot)
            if diff >= self.tear_min_diff and (best is None or diff > best[0]):
                best = (diff, y)
        events = []
        if best is not None:
            if self._tear_open is None:
                self._tear_open = {"start": frame_idx, "peak": best[0],
                                   "y": best[1], "y_min": best[1], "y_max": best[1]}
            else:
                self._tear_open["peak"] = max(self._tear_open["peak"], best[0])
                self._tear_open["y"] = best[1]
                self._tear_open["y_min"] = min(self._tear_open["y_min"], best[1])
                self._tear_open["y_max"] = max(self._tear_open["y_max"], best[1])
            self._tear_last = frame_idx
        elif self._tear_open is not None and frame_idx - self._tear_last > self.tear_end_tol:
            span = self._tear_last - self._tear_open["start"] + 1
            y_spread = self._tear_open["y_max"] - self._tear_open["y_min"]
            if span >= self.tear_min_event and y_spread <= self.tear_max_y_spread:
                events.append(self._emit_tear(self._tear_open["start"], self._tear_last,
                                              self._tear_open["peak"], self._tear_open["y"]))
            elif span >= self.tear_min_event:
                log.debug(f"撕裂候选被位置稳定性否决：span={span} y_spread={y_spread}")
            self._tear_open = None
        return events

    def _emit_tear(self, start, end, peak=0, y_small=None):
        frame_w, frame_h = self._frame_size
        scale = frame_w / self.small_w
        tear_y = int(y_small * scale) if y_small is not None else -1
        event = self.make_event(event_type="tear", start_frame=start, end_frame=end,
                                start_t=start / self._fps, end_t=(end + 1) / self._fps,
                                level="SUSPECT", trigger="ROW_BAND_SHIFT_DISCONTINUITY", status="confirmed",
                                max_shift_diff_px=round(peak * scale, 1), tear_line_y=tear_y)
        self._tear_open = None
        return event

    # ================= 局部冻结 =================
    def _block_diffs(self, small):
        if self._prev_blocks is None:
            return None
        return np.abs(small - self._prev_blocks)

    def _detect_local_freeze(self, small, packet):
        frame_idx = packet.frame_idx
        events = []
        diff_map = self._block_diffs(small)
        prev_small = self._prev_blocks
        self._prev_blocks = small
        if diff_map is None:
            return events
        h, w = diff_map.shape
        bh, bw = max(1, h // self.grid_rows), max(1, w // self.grid_cols)
        grid = np.zeros((self.grid_rows, self.grid_cols))
        texture = np.zeros((self.grid_rows, self.grid_cols))
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                block = diff_map[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                grid[r, c] = block.mean()
                texture[r, c] = prev_small[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw].std()
        global_motion_min = self.global_motion_min
        # 冻结块要求：与上一帧几乎零变化，且块内有真实纹理（排除黑场/纯色天空）
        frozen = (grid <= self.frozen_max_diff) & (texture >= self.frozen_min_texture)
        biggest = self._largest_component(frozen)
        # 全局运动判据（内容自适应）：非冻结块的平均像素差——画面确实在动才算矛盾；
        # 绝对阈值对慢速运动段失效，故用非冻结区域的整体差水平。
        non_frozen = grid[~frozen] if (~frozen).any() else grid.reshape(-1)
        motion_metric = float(non_frozen.mean())
        motion_ok = (motion_metric >= float(self.conf("motion_min_mean_diff", 1.5))
                     and (grid > 2.0).mean() >= global_motion_min * 0.5)

        if motion_ok and biggest >= self.frozen_min_blocks:
            if self._frozen_open is None:
                self._frozen_open = {"start": frame_idx, "blocks": biggest}
            else:
                self._frozen_open["blocks"] = max(self._frozen_open["blocks"], biggest)
            self._frozen_last = frame_idx
        elif self._frozen_open is not None and frame_idx - self._frozen_last > 2:
            span = self._frozen_last - self._frozen_open["start"] + 1
            if span >= self.frozen_min_event:
                events.append(self._emit_frozen(self._frozen_open["start"], self._frozen_last,
                                                self._frozen_open["blocks"]))
            self._frozen_open = None
        return events

    @staticmethod
    def _largest_component(mask):
        """4 邻域连通块最大面积（网格很小，BFS 足够）。"""
        seen = np.zeros_like(mask, dtype=bool)
        best = 0
        rows, cols = mask.shape
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c] or seen[r, c]:
                    continue
                size = 0
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    size += 1
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                best = max(best, size)
        return best

    def _emit_frozen(self, start, end, biggest):
        event = self.make_event(event_type="local_freeze", start_frame=start, end_frame=end,
                                start_t=start / self._fps, end_t=(end + 1) / self._fps,
                                level="SUSPECT", trigger="FROZEN_REGION_WITH_GLOBAL_MOTION",
                                status="confirmed", frozen_blocks=biggest)
        self._frozen_open = None
        return event

    # ================= 块效应 =================
    def _detect_blocking(self, packet):
        """在原生尺度（上限 blocking_max_width）计算网格边界/块内梯度比——
        编码块（8/16px）在原生分辨率上与网格对齐，降采样会错位。帧内指标。"""
        frame_idx = packet.frame_idx
        events = []
        frame = packet.frame
        h, w = frame.shape[:2]
        max_w = int(self.conf("blocking_max_width", 852))
        if w > max_w:
            scale = max_w / w
            gray = cv2.cvtColor(cv2.resize(frame, (max_w, int(h * scale)),
                                           interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        grid = self.blocking_grid
        cols = np.arange(gx.shape[1])
        rows = np.arange(gy.shape[0])
        edge_on = gx[:, (cols % grid == 0)].mean()
        edge_off = gx[:, (cols % grid != 0)].mean()
        row_on = gy[(rows % grid == 0), :].mean()
        row_off = gy[(rows % grid != 0), :].mean()
        denom = max(edge_off + row_off, 1e-6)
        ratio = (edge_on + row_on) / denom
        self._blocking_history.append(ratio)
        baseline = float(np.median(self._blocking_history)) if len(self._blocking_history) > 30 else ratio
        spike = ratio - max(baseline, 1e-6)
        if spike >= self.blocking_min_span:
            if self._blocking_open is None:
                self._blocking_open = {"start": frame_idx, "peak": spike}
            else:
                self._blocking_open["peak"] = max(self._blocking_open["peak"], spike)
            self._blocking_last = frame_idx
        elif self._blocking_open is not None and frame_idx - self._blocking_last > 2:
            span = self._blocking_last - self._blocking_open["start"] + 1
            if span >= 3:
                events.append(self._emit_blocking(self._blocking_open["start"], self._blocking_last,
                                                  self._blocking_open["peak"]))
            self._blocking_open = None
        return events

    def _emit_blocking(self, start, end, peak):
        event = self.make_event(event_type="blocking", start_frame=start, end_frame=end,
                                start_t=start / self._fps, end_t=(end + 1) / self._fps,
                                level="SUSPECT", trigger="BLOCKING_ARTIFACT_SPIKE", status="confirmed",
                                blocking_spike=round(peak, 3))
        self._blocking_open = None
        return event
