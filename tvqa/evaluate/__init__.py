# -*- coding: utf-8 -*-
"""数据集评测：manifest 真值解析 + 帧区间匹配记分卡。"""

from .manifest_loader import DatasetSpec, discover_datasets, load_dataset_spec
from .matcher import match_events, score_dataset

__all__ = ["DatasetSpec", "discover_datasets", "load_dataset_spec",
           "match_events", "score_dataset"]
