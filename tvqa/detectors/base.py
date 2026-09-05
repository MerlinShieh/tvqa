# -*- coding: utf-8 -*-
"""检测器基类与事件协议。

约定：
- 检测器只消费 FramePacket（+自身阈值配置），产出「事件 dict」列表；
  不直接写文件——归档、截图、CSV 全由 pipeline 统一处理（关注点分离）。
- close 语义：事件闭环时返回 open=False，随后 flush 出 finalized 事件。
- safe_process：任何内部异常只记日志，不打断主循环（沿用原脚本隔离策略）。
"""

from ..logging_setup import get_logger
from ..utils import get_beijing_datetime, format_time_for_filename

log = get_logger("detectors.base")


class Detector:
    """所有视觉检测器的基类。子类需实现 process_frame / reset。"""

    #: 事件类型标签（luma 会按极性覆盖为 black/white/flicker）
    event_type = "base"

    def __init__(self, cfg, session=None, dataset=""):
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.session = session
        self.dataset = dataset

    # ---- 生命周期 ----
    def process_frame(self, packet):
        """输入 FramePacket，返回本次闭合的事件 dict 列表。"""
        raise NotImplementedError

    def safe_process(self, packet):
        try:
            return self.process_frame(packet) or []
        except Exception as error:  # noqa: BLE001
            log.warning(f"[{self.name()}] 检测异常已隔离：{type(error).__name__}: {error}")
            return []

    def flush(self):
        """数据流结束时调用：闭合悬挂事件（现场模式/截断评测）。返回事件列表。"""
        return []

    def reset_for_new_dataset(self):
        """评测模式切数据集时重置内部状态。"""
        pass

    # ---- 工具 ----
    def name(self):
        return self.__class__.__name__

    @property
    def enabled(self):
        return bool(self.cfg.get("enabled", True))

    def conf(self, key, default=None):
        return self.cfg.get(key, default)

    def make_event(self, event_type, start_frame, end_frame, start_t, end_t, level="SUSPECT",
                   trigger="", status="recorded", **metrics):
        return {
            "type": event_type,
            "dataset": self.dataset,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_t_sec": round(float(start_t), 4),
            "end_t_sec": round(float(end_t), 4),
            "duration_sec": round(float(end_t - start_t), 4),
            "duration_frames": int(end_frame - start_frame + 1),
            "level": level,
            "trigger": trigger,
            "status": status,
            "metrics": metrics,
            "captured_bj_time": format_time_for_filename(get_beijing_datetime()),
        }
