# -*- coding: utf-8 -*-
"""配置系统：default.yaml + profile 档 + CLI 逐项覆盖 → 嵌套 dict。

规则：
- 深度合并，后者覆盖前者；检测阈值全部在 profiles 里，代码不硬编码业务阈值。
- Cfg.get("a.b.c", default) 点号路径读取。
- --set a.b.c=value 覆盖（自动解析数字/布尔）。
"""

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(text):
    lowered = text.strip().lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


class Cfg:
    """只读点号路径访问的配置包装。"""

    def __init__(self, data):
        self._data = data

    def get(self, dotted_key, default=None):
        node = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, dotted_key):
        value = self.get(dotted_key, {})
        return value if isinstance(value, dict) else {}

    def as_dict(self):
        return copy.deepcopy(self._data)


def load_config(profile=None, overrides=None, config_dir=None):
    """装配配置。profile 为 None 时只取 default；字符串名到 config/profiles/ 查找。"""
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    data = {}
    default_path = config_dir / "default.yaml"
    if default_path.exists():
        data = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}

    if profile:
        profile_path = Path(profile)
        if not profile_path.is_absolute():
            profile_path = config_dir / "profiles" / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"配置档不存在：{profile_path}")
        layer = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        data = _deep_merge(data, layer)

    if overrides:
        patch = {}
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"--set 需要 key=value 形式，收到：{item}")
            key, value = item.split("=", 1)
            node = patch
            parts = key.strip().split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = _parse_scalar(value)
        data = _deep_merge(data, patch)

    return Cfg(data)
