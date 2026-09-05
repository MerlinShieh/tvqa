# -*- coding: utf-8 -*-
"""manifest.json 解析与数据集发现。

约定：input/<名称>/manifest.json 描述该数据集。type 决定评测走哪条链路：
- 视觉类（black_screen/white_screen/flicker/freeze_drop/screen_tearing）：
  episodes 提供真值帧区间，跑检测器后区间匹配。
- av_desync：cases 提供每个 mp4 的真值 offset_frames/offset_ms，走 avsync。
- frames（基准正常内容）：无 manifest.json，作误报基线（期望 0 事件）。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

# manifest.type → 期望检测器事件类型（用于记分卡归类；一型可对多事件类型）
TYPE_TO_EVENT = {
    "black_screen": "black",
    "white_screen": "white",
    "flicker": "flicker",
    "freeze_drop": "stutter",
    "screen_tearing": "tear",
    "corruption": "local_freeze",
    "blocking": "blocking",
}

AV_TYPE = "av_desync"


@dataclass
class DatasetSpec:
    name: str
    path: Path
    kind: str                       # visual | av | baseline
    manifest_type: str = ""
    fps: float = 30.0
    total_frames: int = -1
    episodes: list = field(default_factory=list)   # [{start_frame,end_frame,...}]
    cases: list = field(default_factory=list)      # av: [{file, offset_frames, offset_ms}]
    expected_event: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def gt_count(self):
        return len(self.episodes)


def load_dataset_spec(path):
    path = Path(path)
    manifest_path = path / "manifest.json"
    name = path.name
    if not manifest_path.exists():
        # 无 manifest → baseline（正常内容，期望无事件）
        return DatasetSpec(name=name, path=path, kind="baseline", expected_event="")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mtype = manifest.get("type", "")
    fps = float(manifest.get("fps", 30.0))
    if mtype == AV_TYPE:
        return DatasetSpec(name=name, path=path, kind="av", manifest_type=mtype, fps=fps,
                           cases=manifest.get("cases", []), raw=manifest)
    episodes = _normalize_episodes(manifest)
    return DatasetSpec(name=name, path=path, kind="visual", manifest_type=mtype, fps=fps,
                       total_frames=int(manifest.get("total_frames", -1)), episodes=episodes,
                       expected_event=TYPE_TO_EVENT.get(mtype, ""), raw=manifest)


def _normalize_episodes(manifest):
    episodes = []
    for ep in manifest.get("episodes", []):
        start = ep.get("start_frame")
        end = ep.get("end_frame")
        if start is None or end is None:
            continue
        entry = {"start_frame": int(start), "end_frame": int(end), "seq": ep.get("seq")}
        # 透传参数（tear 的逐帧 tear_y/shift_x、flicker 的 flip_rate 等）供精细核对
        for key, value in ep.items():
            if key not in ("start_frame", "end_frame", "seq"):
                entry[key] = value
        episodes.append(entry)
    return episodes


def discover_datasets(root, only=None):
    root = Path(root)
    specs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if only and child.name not in only:
            continue
        has_frames = any(child.glob("frame_*"))
        has_manifest = (child / "manifest.json").exists()
        has_av = has_manifest and json.loads((child / "manifest.json").read_text(encoding="utf-8")).get("type") == AV_TYPE
        if not has_av and not has_frames:
            continue
        specs.append(load_dataset_spec(child))
    return specs
