# -*- coding: utf-8 -*-
"""六类电视故障样本数据生成器。

以 input/frames（2936 张 PNG，852x480@30fps）和 input/example.mp4 为素材，
生成 6 类故障样本，输出到 input/<故障类型>/ 目录（真值写入各目录 manifest.json）：

  1. 卡顿丢帧   冻结窗口内持续显示同一帧，窗口结束后跳到对应时间的正常帧（丢帧）
  2. 黑屏       窗口内全部替换为纯黑帧
  3. 白屏       窗口内全部替换为纯白帧
  4. 画面撕裂   窗口内帧的上下两段水平错位、错位露出的边缘填黑，帧内容正常推进
  5. 画面闪烁   每次事件 2 秒（60 帧），原帧与 20% 亮度暗帧按 N 次/秒往复
  6. 音画不同步 前 30 秒内容，视频提前/延后 5/10/15/20/30 帧，各生成 1 个 mp4

约束：类型 1-4 事件时长按序 5/10/15/20/30 帧，相邻事件间隔 >= 600 帧（20 秒）；
事件起点优先落在源视频有明显运动的区段（满足卡顿检测器运动 armed 条件）。
未修改的帧一律硬链接指向源帧文件（NTFS 去重，字节内容与独立拷贝完全一致）。
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
FRAMES = INPUT / "frames"
MOVIE = INPUT / "example.mp4"
FFMPEG = r"D:\tools\ffmpeg\ffmpeg.exe"
FFPROBE = r"D:\tools\ffmpeg\ffprobe.exe"

FPS = 30
W, H = 852, 480
N = 2936
DURATIONS = [5, 10, 15, 20, 30]      # 类型 1-4 五次事件的持续帧数
FLICKER_FREQS = [5, 10, 15, 20, 30]  # 类型 5 五次事件的翻转频率（次/秒）
FLICKER_LEN = 60                     # 类型 5 每次事件 60 帧 = 2 秒
MIN_GAP = 600                        # 相邻事件最小间隔（20 秒）
BASE_STARTS = [100, 760, 1420, 2080, 2760]
DESYNC_KS = [5, 10, 15, 20, 30]      # 类型 6 偏移帧数（提前/延后各一套）

MOTION_CACHE = ROOT / ".motion_profile.npy"


def frame_path(i: int) -> Path:
    """第 i 帧（1 基）的源文件路径。"""
    return FRAMES / f"frame_{i:05d}.png"


# ---------------------------------------------------------------- 运动量剖面

def motion_profile() -> np.ndarray:
    """逐帧间平均绝对差（缩小灰度域），len=N-1，scores[t] 为第 t+1 与 t+2 帧之差。

    首次计算后缓存到 .motion_profile.npy。
    """
    if MOTION_CACHE.exists():
        return np.load(MOTION_CACHE)
    scores = np.zeros(N - 1, dtype=np.float32)
    prev = None
    for idx in range(1, N + 1):
        with Image.open(frame_path(idx)) as im:
            g = np.asarray(im.convert("L").resize((86, 48)), dtype=np.int16)
        if prev is not None:
            scores[idx - 2] = np.abs(g - prev).mean()
        prev = g
        if idx % 500 == 0:
            print(f"  运动剖面进度 {idx}/{N}")
    np.save(MOTION_CACHE, scores)
    return scores


# ---------------------------------------------------------------- 事件布局

def episode_layout(type_key: int, durations) -> list:
    """返回 [(start, end, duration), ...]，1 基帧号，保证间隔 >= MIN_GAP。"""
    motion = motion_profile()
    rng = random.Random(9000 + type_key)
    starts = []
    for base in BASE_STARTS:
        starts.append(min(max(base + rng.randint(-100, 100), 40), N - 60))

    episodes = []
    prev_end = 0
    for start, dur in zip(starts, durations):
        # 在 ±100 帧窗口内选"事件前 15 帧平均运动量最大"的起点
        gap_floor = prev_end + MIN_GAP if episodes else 31
        lo = max(start - 100, gap_floor, 31)
        hi = min(start + 100, N - dur)
        if hi < lo:                 # 抖动后窗口空档，回退到边界
            hi = lo
        best, best_score = lo, -1.0
        for cand in range(lo, hi + 1):
            win = motion[max(cand - 16, 0):cand - 1]
            s = win.mean() if win.size else 0.0
            if s > best_score:
                best, best_score = cand, s
        start = min(best, N - dur)  # 夹到不超过末尾
        assert start >= gap_floor, f"{type_key} 间隔不足: {start} {gap_floor}"
        episodes.append((start, start + dur - 1, dur))
        prev_end = start + dur - 1
    assert prev_end <= N, f"{type_key} 末事件越界: {prev_end}"
    return episodes


# ---------------------------------------------------------------- 输出工具

def fresh_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)  # 删除的是链接，不伤及源帧数据
    p.mkdir(parents=True)


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def save_manifest(folder: Path, payload: dict):
    (folder / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def manifest_header(type_key, type_cn, episodes, note):
    return {
        "type": type_key,
        "type_cn": type_cn,
        "fps": FPS, "width": W, "height": H, "total_frames": N,
        "frame_pattern": "frame_%05d.png",
        "layout_rule": "5 次事件，持续 5/10/15/20/30 帧，间隔>=600 帧（20s）；起点落在运动区段",
        "episodes": [
            {"seq": i + 1, "start_frame": s, "end_frame": e, "duration_frames": d}
            for i, (s, e, d) in enumerate(episodes)
        ],
        "note": note,
    }


def in_episodes(i: int, episodes: list):
    for s, e, d in episodes:
        if s <= i <= e:
            return (s, e, d)
    return None


def episode_ranges(episodes: list) -> set:
    hit = set()
    for s, e, _ in episodes:
        hit.update(range(s, e + 1))
    return hit


# ---------------------------------------------------------------- 类型 1-3：链接复用型

def gen_freeze():
    """卡顿丢帧：[s,e] 槽位全部显示源帧 s，其后恢复"槽位 i = 源帧 i"（内容跳帧）。"""
    out = INPUT / "卡顿丢帧"
    episodes = episode_layout(1, DURATIONS)
    fresh_dir(out)
    for i in range(1, N + 1):
        ep = in_episodes(i, episodes)
        src_idx = ep[0] if ep else i
        link_or_copy(frame_path(src_idx), out / f"frame_{i:05d}.png")
    payload = manifest_header(
        "freeze_drop", "卡顿丢帧", episodes,
        "冻结期间重复显示 start_frame 对应源帧；事件结束后显示对应时间正常帧，"
        "源帧 start+1..end 未出现（丢帧），时间轴长度不变")
    for ep, m in zip(episodes, payload["episodes"]):
        m["freeze_source_frame"] = ep[0]
        m["dropped_source_frames"] = [ep[0] + 1, ep[1]] if ep[1] > ep[0] else []
    save_manifest(out, payload)
    return out, episodes


def gen_solid(name: str, color, key: str, note: str):
    """黑屏 / 白屏：事件槽位替换为纯色帧（预生成一张，逐槽硬链接）。"""
    out = INPUT / name
    episodes = episode_layout({"黑屏": 2, "白屏": 3}[name], DURATIONS)
    fresh_dir(out)
    asset = out / "_asset.png"
    Image.new("RGB", (W, H), color).save(asset)
    try:
        for i in range(1, N + 1):
            dst = out / f"frame_{i:05d}.png"
            if in_episodes(i, episodes):
                link_or_copy(asset, dst)
            else:
                link_or_copy(frame_path(i), dst)
    finally:
        asset.unlink()  # 链接已建，删除文件名不影响数据
    save_manifest(out, manifest_header(key, name, episodes, note))
    return out, episodes


# ---------------------------------------------------------------- 类型 4：画面撕裂

def render_torn(src_idx: int, y_t: int, delta: int, dst: Path):
    """下半段水平平移 delta 像素，露出的边缘填黑。"""
    img = np.asarray(Image.open(frame_path(src_idx)).convert("RGB"))
    out = np.zeros_like(img)
    out[:y_t] = img[:y_t]
    bot = img[y_t:]
    if delta >= 0:
        out[y_t:, delta:] = bot[:, :W - delta]
    else:
        out[y_t:, :W + delta] = bot[:, -delta:]
    Image.fromarray(out).save(dst)


def gen_tearing():
    out = INPUT / "画面撕裂"
    episodes = episode_layout(4, DURATIONS)
    fresh_dir(out)
    rng = random.Random(4000)
    per_frame_params = []
    for s, e, d in episodes:
        y_t0 = rng.randint(int(0.42 * H), int(0.58 * H))
        params = []
        for i in range(s, e + 1):
            y_t = int(min(max(y_t0 + rng.randint(-8, 8), 1), H - 1))
            delta = rng.choice([-1, 1]) * rng.randint(15, 60)
            render_torn(i, y_t, delta, out / f"frame_{i:05d}.png")
            params.append({"frame": i, "tear_y": y_t, "shift_x": delta})
        per_frame_params.append(params)
    for i in range(1, N + 1):
        if not in_episodes(i, episodes):
            link_or_copy(frame_path(i), out / f"frame_{i:05d}.png")
    payload = manifest_header(
        "screen_tearing", "画面撕裂", episodes,
        "撕裂窗口内帧的 [tear_y:] 下半段水平平移 shift_x 像素、露边填黑；"
        "帧内容正常推进不丢帧；5 次事件位于不同画面场景")
    for m, params in zip(payload["episodes"], per_frame_params):
        m["frames"] = params
    save_manifest(out, payload)
    return out, episodes


# ---------------------------------------------------------------- 类型 5：画面闪烁

def gen_flicker():
    out = INPUT / "画面闪烁"
    episodes = episode_layout(5, [FLICKER_LEN] * 5)
    fresh_dir(out)
    for (s, e, d), freq in zip(episodes, FLICKER_FREQS):
        for i in range(s, e + 1):
            t = i - s
            dst = out / f"frame_{i:05d}.png"
            if (t * freq) // FPS % 2 == 1:  # 暗帧：每秒 freq 次翻转
                with Image.open(frame_path(i)) as im:
                    ImageEnhance.Brightness(im.convert("RGB")).enhance(0.2).save(dst)
            else:
                link_or_copy(frame_path(i), dst)
    for i in range(1, N + 1):
        if not in_episodes(i, episodes):
            link_or_copy(frame_path(i), out / f"frame_{i:05d}.png")
    payload = manifest_header(
        "flicker", "画面闪烁", episodes,
        "每次事件 60 帧（2 秒），原帧与 20% 亮度暗帧往复；"
        "暗帧判定 (t*freq)//30 % 2 == 1，t 为窗口内相对帧号")
    for m, freq in zip(payload["episodes"], FLICKER_FREQS):
        m["flip_rate_per_sec"] = freq
    save_manifest(out, payload)
    return out, episodes


# ---------------------------------------------------------------- 类型 6：音画不同步

def audio_channels() -> int:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", str(MOVIE)],
        capture_output=True, text=True)
    return int(r.stdout.strip() or 2)


def gen_desync():
    out = INPUT / "音画不同步"
    fresh_dir(out)
    ch = audio_channels()
    records = []
    for k in DESYNC_KS:
        ms = round(k * 1000 / FPS)
        sec = k / FPS
        delay = "|".join([str(ms)] * ch)
        common = ["-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
                  "-t", "30"]
        jobs = [
            (f"video_lead_{k}f.mp4",
             ["-t", "30", "-i", str(MOVIE),
              "-filter_complex", f"[0:a]adelay={delay}[a]"] + common,
             k, ms, "视频提前（音频头部补静音，声音滞后画面）"),
            (f"video_lag_{k}f.mp4",
             ["-t", "31", "-i", str(MOVIE),
              "-filter_complex", f"[0:a]atrim=start={sec:.6f},asetpts=PTS-STARTPTS[a]"] + common,
             -k, ms, "视频延后（裁掉音频头部，声音超前画面）"),
        ]
        for fname, args, off_f, off_ms, desc in jobs:
            cmd = [FFMPEG, "-y", "-v", "error"] + args + [str(out / fname)]
            subprocess.run(cmd, check=True)
            records.append({"file": fname, "offset_frames": off_f,
                            "offset_ms": off_ms if off_f > 0 else -off_ms,
                            "desc": desc})
            print(f"  生成 {fname}")
    save_manifest(out, {
        "type": "av_desync", "type_cn": "音画不同步", "fps": FPS,
        "source": "input/example.mp4 前 30 秒（10 个视频内容相同，仅偏移不同）",
        "clip_seconds": 30,
        "video_codec": "h264 (copy，零损失)", "audio_codec": "aac",
        "cases": records,
        "reference": "ITU-R BT.1359: 音频滞后视频 >45ms 可察觉、>125ms 严重",
    })
    return out, records


# ---------------------------------------------------------------- 主流程

def main():
    print(f"素材: {FRAMES} ({N} 帧, {W}x{H}@{FPS})")
    print("[0] 计算运动量剖面 ...")
    motion_profile()
    print("[1] 卡顿丢帧 ...")
    gen_freeze()
    print("[2] 黑屏 ...")
    gen_solid("黑屏", (0, 0, 0), "black_screen",
              "事件窗口内为纯黑帧（灰度全 0，满足检测器 均亮<=25 且黑占比>=95%）")
    print("[3] 白屏 ...")
    gen_solid("白屏", (255, 255, 255), "white_screen",
              "事件窗口内为纯白帧（灰度全 255，满足检测器 均亮>=235 且白占比>=95%）")
    print("[4] 画面撕裂 ...")
    gen_tearing()
    print("[5] 画面闪烁 ...")
    gen_flicker()
    print("[6] 音画不同步 ...")
    gen_desync()
    print("全部生成完成。")


if __name__ == "__main__":
    sys.exit(main())
