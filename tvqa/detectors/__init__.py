# -*- coding: utf-8 -*-
"""检测器注册表：按配置装配启用的视觉检测器。"""

from .luma import LumaDetector
from .stutter import StutterDetector
from .corruption import CorruptionDetector

VISUAL_REGISTRY = {
    "luma": LumaDetector,
    "stutter": StutterDetector,
    "corruption": CorruptionDetector,
}


def build_visual_detectors(cfg, session, dataset):
    """返回按配置启用的检测器实例列表。avsync 不消费逐帧流，单独处理。"""
    detectors = []
    for name, cls in VISUAL_REGISTRY.items():
        section = cfg.section(f"detectors.{name}")
        params = cfg.section(name)
        enabled = section.get("enabled", params.get("enabled", True))
        if not enabled:
            continue
        merged = {**params}
        merged.setdefault("enabled", True)
        detectors.append(cls(merged, session=session, dataset=dataset))
    return detectors
