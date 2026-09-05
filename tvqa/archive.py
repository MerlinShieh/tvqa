# -*- coding: utf-8 -*-
"""归档与追溯层（可归档/可追溯的落地实现）。

一次会话（session）的产物结构：
  output/session_bjtime_.../
    run_meta.json            时间、主机、代码版本、完整配置快照
    events.jsonl             事件流（每行一个 JSON 事件，含证据相对路径与日志行引用）
    <type>_events_summary.csv 分类事件汇总（表头 utf-8-sig 一次，追加用 utf-8——修复原脚本重复 BOM bug）
    logs/tvqa.log            统一运行日志
    evidence/<dataset>/<type>_event_NNN/   事件证据（截图、标注图、数据文件）

trace 链：HTML 报告行 → events.jsonl 的 event_id → evidence 目录 + log_line。
"""

import json
import os
import platform
from pathlib import Path

from .utils import create_unique_directory, format_time_for_filename, get_beijing_datetime

CSV_CREATE_ENCODING = "utf-8-sig"  # 仅创建时写一次 BOM，Excel 友好
CSV_APPEND_ENCODING = "utf-8"      # 追加不再带 BOM（修复已知 bug）

EVENT_CSV_COLUMNS = [
    "event_id", "type", "dataset", "start_frame", "end_frame",
    "start_t_sec", "end_t_sec", "duration_sec", "level", "trigger",
    "status", "attribution", "metrics_json", "evidence_dir", "log_line",
    "start_bj_time", "end_bj_time",
]


class Session:
    """一次运行的归档会话。"""

    def __init__(self, output_root, mode, config_snapshot, log_counting_handler=None):
        self.started_at = get_beijing_datetime()
        self.mode = mode
        base_name = f"session_bjtime_{format_time_for_filename(self.started_at)}"
        self.root = create_unique_directory(Path(output_root), base_name)
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_root = self.root / "evidence"
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self._events_file = self.events_path.open("a", encoding="utf-8")
        self._log_handler = log_counting_handler
        self._csv_files = {}
        self._event_seq = 0
        self._write_run_meta(config_snapshot)

    def set_log_handler(self, counting_handler):
        """会话创建后才能挂日志 handler（日志写进会话目录），补注入。"""
        self._log_handler = counting_handler

    # ---------- 元信息 ----------
    def _write_run_meta(self, config_snapshot):
        meta = {
            "session": self.root.name,
            "start_bj_time": self.started_at.isoformat(),
            "mode": self.mode,
            "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version()},
            "code_version": _code_version(),
            "config": config_snapshot,
        }
        (self.root / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def finish(self, extra_meta=None):
        if extra_meta:
            meta_path = self.root / "run_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["finish"] = extra_meta
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._events_file.close()
        for handle in self._csv_files.values():
            handle.close()

    # ---------- 事件流 ----------
    @property
    def log_line(self):
        return self._log_handler.line_count if self._log_handler else 0

    def next_event_id(self, event_type, dataset):
        self._event_seq += 1
        return f"{event_type}@{dataset}#{self._event_seq:04d}"

    def record_event(self, event):
        """event: dict（detector 层构造，含 type/dataset/frame 区间等）。落 events.jsonl + 分类 CSV。"""
        event.setdefault("event_id", self.next_event_id(event.get("type", "unknown"), event.get("dataset", "-")))
        event.setdefault("log_line", self.log_line)
        line = json.dumps(event, ensure_ascii=False, default=str)
        self._events_file.write(line + "\n")
        self._events_file.flush()
        self._append_csv(event)

    def _append_csv(self, event):
        event_type = event.get("type", "unknown")
        csv_path = self.root / f"{event_type}_events_summary.csv"
        if event_type not in self._csv_files:
            handle = csv_path.open("a", encoding=CSV_APPEND_ENCODING, newline="")
            if csv_path.stat().st_size == 0:
                # 新建时补一次带 BOM 的表头写入（保持 Excel 中文不乱码）
                handle.close()
                with csv_path.open("w", encoding=CSV_CREATE_ENCODING, newline="") as writer_file:
                    import csv as csv_mod
                    csv_mod.writer(writer_file).writerow(EVENT_CSV_COLUMNS)
                handle = csv_path.open("a", encoding=CSV_APPEND_ENCODING, newline="")
            self._csv_files[event_type] = handle
        import csv as csv_mod
        row = []
        for column in EVENT_CSV_COLUMNS:
            value = event.get(column, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            row.append(value)
        csv_mod.writer(self._csv_files[event_type]).writerow(row)

    # ---------- 证据目录 ----------
    def evidence_dir(self, dataset, folder_name):
        """返回事件证据目录（自动创建）。folder_name 建议 <type>_event_NNN_startframe_..."""
        path = self.evidence_root / dataset / folder_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_evidence_json(self, evidence_dir, name, payload):
        path = Path(evidence_dir) / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def rel(self, path):
        """绝对路径 → 相对会话根目录（HTML 链接用）。"""
        return os.path.relpath(path, self.root).replace("\\", "/")


def _code_version():
    """代码版本标识：优先 git commit；无 git 时用包版本 + 主模块 mtime 摘要。"""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent), capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            return {"git": result.stdout.strip()}
    except Exception:
        pass
    from . import __version__
    return {"package_version": __version__}
