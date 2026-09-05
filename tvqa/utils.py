# -*- coding: utf-8 -*-
"""通用工具：北京时间格式化、中文路径安全存图、唯一目录、数值小工具。

移植自原《黑屏和卡顿检测.py》，行为保持一致。
"""

import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def get_beijing_datetime():
    return datetime.now(BEIJING_TIMEZONE)


def format_time_for_filename(value=None):
    if value is None:
        value = get_beijing_datetime()
    return value.strftime("%Y-%m-%d_%H-%M-%S-") + value.strftime("%f")[:3]


def format_time_for_log(value=None):
    if value is None:
        value = get_beijing_datetime()
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_time_for_csv(value):
    if value is None:
        return ""
    if isinstance(value, datetime):
        return f"北京时间 {format_time_for_log(value)}"
    return f"北京时间 {value}"


def save_image(path, image, jpg_quality=95):
    """编码后写文件，兼容中文路径；返回是否成功。"""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if ext in (".jpg", ".jpeg") else []
        success, data = cv2.imencode(ext, image, params)
        if not success:
            return False
        data.tofile(str(path))
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


def create_unique_directory(parent_dir, base_name):
    event_dir = Path(parent_dir) / base_name
    dup = 1
    while event_dir.exists():
        event_dir = Path(parent_dir) / f"{base_name}_dup{dup:03d}"
        dup += 1
    event_dir.mkdir(parents=True, exist_ok=False)
    return event_dir


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def mean_abs_diff(gray_a, gray_b):
    """两张灰度图的平均绝对差与明显变化像素比例（帧差统计统一入口）。"""
    difference = cv2.absdiff(gray_a, gray_b)
    return float(np.mean(difference)), difference


def downscale_gray(frame, width, height):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
