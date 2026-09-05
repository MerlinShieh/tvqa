# -*- coding: utf-8 -*-
"""音画同步测量（L1 统计层）：镜头切换流 × 音频起始流互相关。

方法（对应人工「挑声画锚点 + 切帧比对」的自动化等价物）：
- 视频锚点：镜头切换（相邻帧分析尺度平均像素差突增，时间精度=1 帧）；
- 音频锚点：RMS 通量 onset（能量突起，精度=音频块长 ≈21ms @48k）；
- 对每个视频切换找 ±search 窗内最近音频 onset，差值序列取中位数 = offset。
符号约定与 manifest 一致：offset > 0 = 声音滞后画面（视频提前/video_lead）。
"""

from statistics import median

import cv2
import numpy as np

from ..logging_setup import get_logger
from ..sources.audio_file import AudioFileSource

log = get_logger("detectors.avsync")

_ANALYSIS_WIDTH = 160


# ============================================================
# L3 主动标定：闪光 + 蜂鸣测试片（方案 §3.6）
# 内容本身无语义声画锚点时（如纯 BGM 视频），被动估计不可测；
# 现场标定与功能验证统一走 L3：生成已知偏移的测试媒体 → 测量 → 对账。
# ============================================================

def generate_test_clip(path, offset_ms, seconds=30.0, fps=30.0, width=640, height=360,
                       samplerate=48000, beep_hz=880.0):
    """生成一段「每 2 秒白闪 3 帧 + 同刻蜂鸣」的测试媒体，音轨整体平移 offset_ms。

    offset_ms > 0：视频提前（音频头部补静音，声音滞后画面，与数据集 lead 口径一致）；
    offset_ms < 0：视频延后（裁掉音频头部，声音超前画面）。
    真值 offset = 蜂鸣时刻 - 闪光时刻（注入后）。
    """
    import subprocess
    import tempfile
    import wave
    from pathlib import Path

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("缺少 ffmpeg 支持：pip install imageio-ffmpeg") from error

    total_frames = int(seconds * fps)
    flash_starts = [int(t * fps) for t in np.arange(3.0, seconds - 0.1, 3.0)]
    flash_len = 3
    base = 120.0
    frames = np.full((total_frames, height, width, 3), base, dtype=np.uint8)
    # 移动斜条（模拟内容运动，避免纯静态）
    for i in range(total_frames):
        y0 = int((i * 3) % height)
        frames[i, max(0, y0 - 20):y0, :, :] = 200
    for start in flash_starts:
        frames[start:start + flash_len, :, :, :] = 250

    beep_samples = int(0.12 * samplerate)
    audio = np.zeros(int(seconds * samplerate), dtype=np.float32)
    beep_t = np.arange(beep_samples) / samplerate
    envelope = np.minimum(1.0, np.minimum(beep_t / 0.005, (0.12 - beep_t) / 0.01))
    beep = (0.5 * np.sin(2 * np.pi * beep_hz * beep_t) * envelope).astype(np.float32)
    for t in np.arange(3.0, seconds - 0.1, 3.0):
        start = int(t * samplerate)
        audio[start:start + beep_samples] = np.maximum(audio[start:start + beep_samples], beep)

    shift = int(round(offset_ms / 1000 * samplerate))
    if shift > 0:    # 音频滞后：头部补静音
        audio = np.concatenate([np.zeros(shift, dtype=np.float32), audio[:-shift or None]])
    elif shift < 0:  # 音频超前：裁头部，尾部补零
        audio = np.concatenate([audio[-shift:], np.zeros(-shift, dtype=np.float32)])

    wav_path = Path(path).with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(samplerate)
        wav_file.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())

    command = [ffmpeg, "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-",
               "-i", str(wav_path),
               "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        process.stdin.write(frames.tobytes())
        process.stdin.close()
        process.wait(timeout=120)
    finally:
        wav_path.unlink(missing_ok=True)
    if process.returncode != 0:
        raise RuntimeError(f"测试片生成失败：{path}")
    return {"flash_count": len(flash_starts), "offset_ms": offset_ms,
            "video_seconds": seconds, "fps": fps}


class L3Analyzer:
    """闪光+蜂鸣测试片偏移测量（L1 事件机制在确定性信号上的特化版）。"""

    def __init__(self, cfg=None):
        cfg = cfg if isinstance(cfg, dict) else {}
        self.flash_brightness = float(cfg.get("flash_brightness", 200.0))
        self.search_s = float(cfg.get("l3_search_window_ms", 1400)) / 1000.0

    def measure(self, video_path, fps_hint=30.0):
        cap = cv2.VideoCapture(str(video_path))
        fps = fps_hint or float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        flashes, prev_bright, in_flash = [], None, False
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bright = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            is_flash = bright >= self.flash_brightness
            if is_flash and not in_flash and (prev_bright is None or prev_bright < self.flash_brightness):
                flashes.append(frame_idx / fps)
            in_flash = is_flash
            prev_bright = bright
            frame_idx += 1
        cap.release()

        source = AudioFileSource({"path": str(video_path), "block_samples": 1024})
        samples = source.decode_all()
        samplerate = source.samplerate
        n_blocks = len(samples) // 1024
        blocks = samples[:n_blocks * 1024].reshape(n_blocks, 1024)
        rms = np.sqrt((blocks ** 2).mean(axis=1))
        floor = float(np.percentile(rms, 50)) * 3 + 1e-4
        beeps = []
        last = -10 ** 9
        for i, value in enumerate(rms):
            if value >= floor and i - last >= 40:  # ≥1.2s 间隔，每 3s 一个蜂鸣
                beeps.append((i + 1) * 1024 / samplerate)
                last = i
        deltas = []
        if flashes and beeps:
            beep_arr = np.asarray(beeps)
            for flash_t in flashes:
                window = beep_arr[(beep_arr >= flash_t - self.search_s) & (beep_arr <= flash_t + self.search_s)]
                if len(window):
                    deltas.append(float(window[np.argmin(np.abs(window - flash_t))]) - flash_t)
        result = {"flash_count": len(flashes), "beep_count": len(beeps), "pairs": len(deltas)}
        if deltas:
            offset = median(deltas)
            mad = median(abs(d - offset) for d in deltas)
            result.update({"offset_ms": round(offset * 1000, 1), "mad_ms": round(mad * 1000, 1),
                           "confidence": "high" if len(deltas) >= 3 else "low"})
        else:
            result.update({"offset_ms": None, "mad_ms": None, "confidence": "none"})
        return result


class AVSyncAnalyzer:
    """对一个 (视频, 音频) 对做离线偏移测量。"""

    def __init__(self, cfg):
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.search_ms = float(self.cfg.get("search_window_ms", 1500))
        self.cut_threshold = float(self.cfg.get("cut_mean_diff_threshold", 14.0))
        self.flux_factor = float(self.cfg.get("onset_flux_factor", 2.0))
        self.rms_floor = float(self.cfg.get("block_rms_threshold", 0.001))
        self.min_onset_gap_blocks = int(self.cfg.get("onset_min_interval_frames", 8))
        self.min_pairs = int(self.cfg.get("min_matched_pairs", 5))
        self.audio_block = int(self.cfg.get("audio_block_samples", 1024))

    # ---------- 视频侧 ----------
    def detect_cuts(self, video_path, fps_hint=None):
        """镜头切换：帧间大差 + 局部峰值（区分真实切换与高速运动的连续大差）。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频：{video_path}")
        fps = fps_hint or float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        diffs, prev_small = [], None
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (_ANALYSIS_WIDTH, 90),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev_small is not None:
                diffs.append((frame_idx, float(np.abs(small - prev_small).mean())))
            prev_small = small
            frame_idx += 1
        cap.release()
        diff_values = np.array([d for _, d in diffs]) if diffs else np.zeros(1)
        local_factor = float(self.cfg.get("cut_local_factor", 1.6))
        cuts = []
        for position, (idx, value) in enumerate(diffs):
            if value < self.cut_threshold:
                continue
            lo, hi = max(0, position - 8), min(len(diffs), position + 9)
            local_med = float(np.median(diff_values[lo:hi]))
            if value >= local_factor * max(local_med, 1e-6):
                cuts.append(idx / fps)
        return cuts, fps, frame_idx

    # ---------- 音频侧 ----------
    def detect_onsets(self, audio_path):
        """强瞬态 onset：局部通量峰值 + 全局分位数门槛（滤掉密集音乐节拍沿，
        保留打击/撞击类强瞬态——那是与画面动作对齐的锚点）。"""
        source = AudioFileSource({"path": str(audio_path), "block_samples": self.audio_block})
        samples = source.decode_all()
        samplerate = source.samplerate
        n_blocks = len(samples) // self.audio_block
        blocks = samples[:n_blocks * self.audio_block].reshape(n_blocks, self.audio_block)
        rms = np.sqrt((blocks ** 2).mean(axis=1))
        flux = np.maximum(0.0, rms[1:] - rms[:-1])
        percentile = float(self.cfg.get("onset_percentile", 90))
        global_floor = float(np.percentile(flux, percentile)) if len(flux) else 0.0
        local_factor = float(self.cfg.get("onset_local_factor", 3.0))
        min_gap = int(self.cfg.get("onset_min_interval_blocks", 20))
        onsets = []
        last = -10 ** 9
        for i, value in enumerate(flux):
            lo, hi = max(0, i - 25), min(len(flux), i + 26)
            local_med = float(np.median(flux[lo:hi]))
            if (value > local_factor * max(local_med, 1e-6) and value >= global_floor
                    and value > self.rms_floor and rms[i] > self.rms_floor
                    and i - last >= min_gap):
                onsets.append((i + 1) * self.audio_block / samplerate)
                last = i
        return onsets, samplerate, len(samples) / samplerate

    # ---------- 对齐 ----------
    def measure(self, video_path, audio_path=None, fps_hint=None):
        """返回 {offset_ms, mad_ms, pairs, confidence, ...}。

        偏移用聚类取模：delta 序列按 80ms 桶聚类，取最大簇的中位数——
        真实对齐占主导，散点噪声不拉偏结果（对应人工「多数锚点一致才下结论」）。
        """
        cuts, fps, total_frames = self.detect_cuts(video_path, fps_hint)
        onsets, samplerate, audio_seconds = self.detect_onsets(audio_path or video_path)
        search = self.search_ms / 1000.0
        deltas = []
        if onsets:
            onset_arr = np.asarray(onsets)
            for cut_t in cuts:
                candidates = onset_arr[(onset_arr >= cut_t - search) & (onset_arr <= cut_t + search)]
                if len(candidates):
                    deltas.append(float(candidates[np.argmin(np.abs(candidates - cut_t))]) - cut_t)
        result = {
            "video_seconds": round(total_frames / fps, 2) if fps else 0.0,
            "audio_seconds": round(audio_seconds, 3),
            "cut_count": len(cuts),
            "onset_count": len(onsets),
            "pairs": len(deltas),
        }
        if deltas:
            offset, mad, cluster_size = self._dominant_cluster(deltas)
            mass_fraction = cluster_size / len(deltas)
            result.update({
                "offset_ms": round(offset * 1000, 1),
                "mad_ms": round(mad * 1000, 1),
                "cluster_pairs": cluster_size,
                "mass_fraction": round(mass_fraction, 3),
                # 置信度要求主簇既有足够配对又有足够集中度（防周期节拍简并假峰）
                "confidence": ("high" if cluster_size >= self.min_pairs and mass_fraction >= 0.25
                               else "low"),
                "direction": "audio_lags_video(video_lead)" if offset > 0 else "audio_leads_video(video_lag)",
            })
        else:
            result.update({"offset_ms": None, "mad_ms": None, "cluster_pairs": 0,
                           "confidence": "none", "direction": "unknown"})
        return result

    @staticmethod
    def _dominant_cluster(deltas, bin_seconds=0.08):
        """按 80ms 桶聚类 delta，返回最大簇的 (中位数, MAD, 簇大小)。"""
        deltas = sorted(deltas)
        best_cluster, best_key = [], None
        anchor = deltas[0]
        cluster = []
        for value in deltas + [float("inf")]:
            if value - anchor <= bin_seconds:
                cluster.append(value)
                continue
            key = len(cluster)
            if best_key is None or key > best_key:
                best_key, best_cluster = key, cluster
            anchor = value
            cluster = [value] if value != float("inf") else []
        cluster = best_cluster
        offset = median(cluster)
        mad = median(abs(d - offset) for d in cluster)
        return offset, mad, len(cluster)
