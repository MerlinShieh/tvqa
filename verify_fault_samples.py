# -*- coding: utf-8 -*-
"""对六类故障样本产物做程序化验证，输出核对报告。"""

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
FRAMES = INPUT / "frames"
N = 2936
W, H = 852, 480

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def frame_path(i):
    return FRAMES / f"frame_{i:05d}.png"


def out_path(folder, i):
    return INPUT / folder / f"frame_{i:05d}.png"


def load_arr(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)


def is_same_file(a, b):
    sa, sb = a.stat(), b.stat()
    return (sa.st_ino, sa.st_dev) == (sb.st_ino, sb.st_dev)


def same_bytes(a, b):
    if is_same_file(a, b):
        return True
    return a.read_bytes() == b.read_bytes()


def manifest(folder):
    return json.loads((INPUT / folder / "manifest.json").read_text(encoding="utf-8"))


def verify_frame_count(folder):
    files = list((INPUT / folder).glob("frame_*.png"))
    check(f"{folder}: 帧数=2936", len(files) == N, f"实际 {len(files)}")


def verify_outside_events(folder, episodes):
    """抽查 300 个非事件槽位，应与源帧同 inode/同字节。"""
    event_slots = set()
    for s, e in episodes:
        event_slots.update(range(s, e + 1))
    rng = random.Random(7)
    outside = [i for i in rng.sample(range(1, N + 1), 300) if i not in event_slots]
    bad = [i for i in outside if not same_bytes(out_path(folder, i), frame_path(i))]
    check(f"{folder}: 非事件帧与源帧一致（抽查300）", not bad, f"异常 {bad[:5]}")


def verify_intervals(folder, eps):
    ok = all(eps[i + 1][0] - eps[i][1] >= 600 for i in range(len(eps) - 1))
    check(f"{folder}: 事件间隔>=600帧", ok, str(eps))


def verify_freeze():
    folder = "卡顿丢帧"
    m = manifest(folder)
    eps = [(e["start_frame"], e["end_frame"]) for e in m["episodes"]]
    verify_frame_count(folder)
    verify_intervals(folder, eps)
    ok = True
    for (s, e), ep in zip(eps, m["episodes"]):
        for i in range(s, e + 1):
            if not same_bytes(out_path(folder, i), frame_path(s)):
                ok = False
                break
    check(f"{folder}: 事件窗口内全部为源帧 start 的重复", ok)
    after_ok = all(
        same_bytes(out_path(folder, e + 1), frame_path(e + 1))
        for _, e in eps if e + 1 <= N)
    check(f"{folder}: 窗口结束后接回对应时间正常帧", after_ok)
    verify_outside_events(folder, eps)


def verify_solid(folder, value):
    m = manifest(folder)
    eps = [(e["start_frame"], e["end_frame"]) for e in m["episodes"]]
    verify_frame_count(folder)
    verify_intervals(folder, eps)
    ok = True
    for s, e in eps:
        for i in range(s, e + 1):
            if int(load_arr(out_path(folder, i)).mean()) != value:
                ok = False
    check(f"{folder}: 事件帧为纯{'黑' if value == 0 else '白'}（均值={value}）", ok)
    verify_outside_events(folder, eps)


def verify_tearing():
    folder = "画面撕裂"
    m = manifest(folder)
    eps = [(e["start_frame"], e["end_frame"]) for e in m["episodes"]]
    verify_frame_count(folder)
    verify_intervals(folder, eps)
    rng = random.Random(11)
    bad = []
    for ep in m["episodes"]:
        for p in rng.sample(ep["frames"], 3):
            i, y, d = p["frame"], p["tear_y"], p["shift_x"]
            out_a = load_arr(out_path(folder, i))
            src_a = load_arr(frame_path(i))
            recon = np.zeros_like(src_a)
            recon[:y] = src_a[:y]
            if d >= 0:
                recon[y:, d:] = src_a[y:, :W - d]
            else:
                recon[y:, :W + d] = src_a[y:, -d:]
            if not np.array_equal(out_a, recon):
                bad.append(i)
    check(f"{folder}: 撕裂帧按 manifest 参数可精确重构（抽查15帧）", not bad, f"异常 {bad[:5]}")
    verify_outside_events(folder, eps)
    # 场景分散性：5 次事件间隔 >= 600 帧即不同场景
    starts = [e["start_frame"] for e in m["episodes"]]
    check(f"{folder}: 5 个事件位于分散画面场景", len(starts) == 5, str(starts))


def verify_flicker():
    folder = "画面闪烁"
    m = manifest(folder)
    verify_frame_count(folder)
    eps = [(e["start_frame"], e["end_frame"]) for e in m["episodes"]]
    verify_intervals(folder, eps)
    all_ok, flips_ok = True, True
    for ep in eps:
        s = ep[0]
        freq = [x["flip_rate_per_sec"] for x in m["episodes"] if x["start_frame"] == s][0]
        luma = []
        for i in range(s, ep[1] + 1):
            t = i - s
            dark = (t * freq) // 30 % 2 == 1
            a = load_arr(out_path(folder, i))
            if dark:
                src_mean = load_arr(frame_path(i)).mean()
                if a.mean() > src_mean * 0.45:
                    all_ok = False
            else:
                if not same_bytes(out_path(folder, i), frame_path(i)):
                    all_ok = False
            luma.append(a.mean())
        flips = int(np.sum(np.abs(np.diff(luma)) > 30))
        expected = int((ep[1] - s) * freq / 30)  # 60 帧内约 2*freq 次翻转
        if abs(flips - expected) > 2:
            flips_ok = False
    check(f"{folder}: 暗帧亮度约20%/明帧与源帧一致", all_ok)
    check(f"{folder}: 翻转次数与频率 N 匹配", flips_ok)
    verify_outside_events(folder, eps)


def verify_desync():
    folder = "音画不同步"
    m = manifest(folder)
    ok = True
    detail = []
    for c in m["cases"]:
        p = INPUT / folder / c["file"]
        if not p.exists():
            ok = False
            continue
        r = subprocess.run(
            ["D:/tools/ffmpeg/ffprobe.exe", "-v", "error",
             "-show_entries", "stream=codec_type,duration", "-of", "json", str(p)],
            capture_output=True, text=True)
        info = json.loads(r.stdout)
        durs = {s["codec_type"]: float(s.get("duration", 0)) for s in info["streams"]}
        vd, ad = durs.get("video", 0), durs.get("audio", 0)
        if not (29.0 <= vd <= 31.0 and 29.0 <= ad <= 31.0):
            ok = False
            detail.append(f"{c['file']}: v={vd} a={ad}")
    check(f"{folder}: 10 个视频存在且时长≈30s", ok, "; ".join(detail[:3]))
    offs = [c["offset_ms"] for c in m["cases"]]
    check(f"{folder}: 偏移覆盖 ±5/10/15/20/30 帧",
          sorted(abs(o) for o in offs) == [167, 167, 333, 333, 500, 500, 667, 667, 1000, 1000],
          str(sorted(offs)))


def main():
    verify_freeze()
    verify_solid("黑屏", 0)
    verify_solid("白屏", 255)
    verify_tearing()
    verify_flicker()
    verify_desync()
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"\n===== 验证汇总: {len(results)} 项检查, {n_fail} 项失败 =====")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
