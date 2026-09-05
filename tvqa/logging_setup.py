# -*- coding: utf-8 -*-
"""统一日志系统。

- 北京时间毫秒精度，控制台 + 会话日志文件双写。
- CountingHandler 记录已写行数：events.jsonl 里每条事件带 log_line 引用，
  实现「事件 → 日志上下文」的可追溯链（配合归档日志切片使用）。
"""

import logging
import sys
from datetime import datetime

from .utils import BEIJING_TIMEZONE

_CONSOLE_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


class BeijingFormatter(logging.Formatter):
    """把 asctime 换成北京时间（毫秒）。"""

    def formatTime(self, record, datefmt=None):  # noqa: N802
        dt = datetime.fromtimestamp(record.created, BEIJING_TIMEZONE)
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


class CountingHandler(logging.StreamHandler):
    """文件 handler，统计已输出行数供事件追溯引用。"""

    def __init__(self, stream):
        super().__init__(stream)
        self.line_count = 0

    def emit(self, record):
        super().emit(record)
        self.line_count += 1


def setup_logging(log_file=None, console_level="INFO", file_level="DEBUG"):
    """配置根 logger（幂等：重复调用会重建 handler）。返回 CountingHandler 或 None。"""
    root = logging.getLogger("tvqa")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    formatter = BeijingFormatter(_CONSOLE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, str(console_level).upper(), logging.INFO))
    console.setFormatter(formatter)
    root.addHandler(console)

    counting = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stream = log_file.open("a", encoding="utf-8")
        counting = CountingHandler(stream)
        counting.setLevel(getattr(logging, str(file_level).upper(), logging.DEBUG))
        counting.setFormatter(formatter)
        root.addHandler(counting)
    return counting


def get_logger(name):
    """模块 logger 工厂：统一挂在 tvqa.* 命名空间下。"""
    return logging.getLogger(f"tvqa.{name}")
