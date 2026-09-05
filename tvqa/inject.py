# -*- coding: utf-8 -*-
"""花屏合成样本注入器：从基准帧序列生成带真值 manifest 的花屏数据集。

当前 input/ 数据集缺花屏类型；本注入器用与既有数据集相同的 manifest 规范补造：
- frozen_region 样式：矩形区域内冻结为首帧内容（局部冻结/花屏典型形态）；
- mosaic 样式：矩形区域做 8 像素块量化（马赛克/压缩块效应）。
事件时长 10/15/20/30/40 帧（局部损坏类事件比黑屏更短时与内容歧义难分，取更长下限）。

用法：python -m tvqa inject-corruption --base input/frames --out input/花屏
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .logging_setup import get_logger

log = get_logger("inject")

# (seq, start_frame, duration_frames, style, region)
# start 沿用既有数据集布局规则：间隔 ≥600 帧且落在运动区段（与黑屏 manifest 同源）
EPISODE_PLAN = [
    (1, 63, 10, "frozen_region", (0.30, 0.25, 0.36, 0.45)),   # (x0,y0,w,h) 比例
    (2, 858, 15, "mosaic", (0.10, 0.15, 0.30, 0.35)),
    (3, 1503, 20, "frozen_region", (0.45, 0.40, 0.30, 0.40)),
    (4, 2203, 30, "mosaic", (0.25, 0.30, 0.38, 0.45)),
    (5, 2822, 40, "frozen_region", (0.15, 0.20, 0.34, 0.50)),
]


def apply_frozen_region(frame, anchor_frame, region, size):
    h, w = size[:2]
    x0, y0, rw, rh = region
    xa, ya, wa, ha = int(x0 * w), int(y0 * h), int(rw * w), int(rh * h)
    out = frame.copy()
    out[ya:ya + ha, xa:xa + wa] = anchor_frame[ya:ya + ha, xa:xa + wa]
    return out


def apply_mosaic(frame, region, size, block=8):
    h, w = size[:2]
    x0, y0, rw, rh = region
    xa, ya, wa, ha = int(x0 * w), int(y0 * h), int(rw * w), int(rh * h)
    out = frame.copy()
    patch = frame[ya:ya + ha, xa:xa + wa]
    small = cv2.resize(patch, (max(1, wa // block), max(1, ha // block)), interpolation=cv2.INTER_LINEAR)
    out[ya:ya + ha, xa:xa + wa] = cv2.resize(small, (wa, ha), interpolation=cv2.INTER_NEAREST)
    return out


def build_corruption_dataset(base_dir, out_dir, seed=7):
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    frame_paths = sorted(p for p in base_dir.glob("frame_*") if p.suffix.lower() in (".png", ".jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"基准帧目录无帧：{base_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    total = len(frame_paths)
    boundaries = sorted(
        [(seq, start, start + duration - 1, duration, style, region)
         for seq, start, duration, style, region in EPISODE_PLAN],
        key=lambda item: item[1])
    episode_at = {item[1]: item for item in boundaries}
    # 锚点帧（冻结区域复制源）单独读取，避免整段驻留内存
    anchor_frames = {}
    for seq, start, end, duration, style, region in boundaries:
        raw = np.fromfile(str(frame_paths[start]), dtype=np.uint8)
        anchor_frames[start] = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    manifest_episodes = []
    current_episode = None
    for idx, path in enumerate(frame_paths):
        raw = np.fromfile(str(path), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        episode = episode_at.get(idx)
        if episode:
            seq, start, end, duration, style, region = episode
            current_episode = {"seq": seq, "start": start, "end": end,
                               "style": style, "region": region,
                               "anchor": anchor_frames[start]}
            manifest_episodes.append({
                "seq": seq, "start_frame": start, "end_frame": end,
                "duration_frames": duration, "style": style,
                "region_ratio": list(region),
            })
        if current_episode and current_episode["start"] <= idx <= current_episode["end"]:
            if current_episode["style"] == "frozen_region":
                frame = apply_frozen_region(frame, current_episode["anchor"],
                                            current_episode["region"], frame.shape)
            else:
                frame = apply_mosaic(frame, current_episode["region"], frame.shape)
        if current_episode and idx > current_episode["end"]:
            current_episode = None
        # PNG 输出（与既有数据集一致）
        success, data = cv2.imencode(".png", frame)
        data.tofile(str(out_dir / f"frame_{idx:05d}.png"))
    manifest = {
        "type": "corruption",
        "type_cn": "花屏（合成）",
        "fps": 30.0,
        "width": int(cv2.imdecode(np.fromfile(str(frame_paths[0]), dtype=np.uint8), cv2.IMREAD_COLOR).shape[1]),
        "height": int(cv2.imdecode(np.fromfile(str(frame_paths[0]), dtype=np.uint8), cv2.IMREAD_COLOR).shape[0]),
        "total_frames": total,
        "frame_pattern": "frame_%05d.png",
        "layout_rule": "5 次事件（10/15/20/30/40 帧），间隔>=600 帧；起点沿用黑屏数据集运动区段",
        "episodes": manifest_episodes,
        "note": "frozen_region=区域冻结(局部花屏)；mosaic=区域8px块量化(块效应)；由 tvqa inject-corruption 生成",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"花屏数据集已生成：{out_dir}（{total} 帧，{len(manifest_episodes)} 个事件）")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tvqa inject-corruption")
    parser.add_argument("--base", default="input/frames")
    parser.add_argument("--out", default="input/花屏")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    build_corruption_dataset(args.base, args.out, args.seed)
    return 0


if __name__ == "__main__":
    main()
