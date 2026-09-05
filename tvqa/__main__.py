# -*- coding: utf-8 -*-
"""tvqa 命令行入口。

  python -m tvqa eval  --dataset input --profile eval [--only 黑屏 ...] [--max-frames N]
  python -m tvqa run   --profile field                     # 现场模式（步骤8打通）
  python -m tvqa report --session output/session_...       # 重新生成 HTML（步骤6）
"""

import argparse
import json
import sys
import time

from .archive import Session
from .clock import VirtualClock
from .config import load_config
from .logging_setup import get_logger, setup_logging
from .pipeline import run_frames_dataset
from .sources.frames_dir import FramesDirSource


def build_parser():
    parser = argparse.ArgumentParser(prog="tvqa", description="电视音画质量自动化检测")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="在带 manifest 真值的数据集上评测")
    ev.add_argument("--dataset", default="input", help="数据集根目录")
    ev.add_argument("--profile", default="eval", help="阈值档位名（config/profiles/*.yaml）")
    ev.add_argument("--only", nargs="*", help="只评测指定数据集目录名")
    ev.add_argument("--max-frames", type=int, default=0, help="每数据集最多处理帧数（0=全部）")
    ev.add_argument("--no-evidence", action="store_true", help="不保存事件截图（快速回归）")
    _add_common(ev)

    run = sub.add_parser("run", help="现场/采集模式")
    run.add_argument("--profile", default="field")
    _add_common(run)

    rep = sub.add_parser("report", help="由已归档会话重新生成 HTML 报告")
    rep.add_argument("--session", required=True)

    inj = sub.add_parser("inject-corruption", help="从基准帧序列生成花屏合成数据集（带真值 manifest）")
    inj.add_argument("--base", default="input/frames")
    inj.add_argument("--out", default="input/花屏")
    inj.add_argument("--seed", type=int, default=7)
    return parser


def _add_common(parser):
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="配置覆盖（可重复），如 --set video.backend=capture_card")
    parser.add_argument("--output-root", default=None)


def cmd_eval(args):
    from .evaluate import discover_datasets, score_dataset
    cfg = load_config(args.profile, args.overrides)
    output_root = args.output_root or cfg.get("output_root", "output")
    session = Session(output_root, mode="eval", config_snapshot=cfg.as_dict())
    counting = setup_logging(session.logs_dir / "tvqa.log",
                             cfg.get("log.console_level"), cfg.get("log.file_level"))
    session.set_log_handler(counting)
    log = get_logger("cli")
    log.info(f"会话归档目录：{session.root}")

    if args.no_evidence:
        cfg._data.setdefault("evidence", {})["frames_per_event"] = None

    specs = discover_datasets(args.dataset, only=args.only)
    if not specs:
        log.error(f"未在 {args.dataset} 发现可评测数据集")
        return 2

    scorecards = []
    started = time.time()
    for spec in specs:
        if spec.kind == "av":
            scorecards.extend(run_av_dataset(cfg, session, spec, log))
            continue
        log.info(f"评测数据集：{spec.name}（type={spec.manifest_type or 'baseline'}，"
                 f"真值事件 {spec.gt_count} 个）")
        source = FramesDirSource({"path": str(spec.path), "fps": spec.fps}, VirtualClock(spec.fps))
        if args.max_frames and source.meta.total_frames > args.max_frames:
            source._frames = source._frames[:args.max_frames]
        events = run_frames_dataset(cfg, session, source, spec.name)
        scorecards.append(score_dataset(spec, events, evaluate_cfg=cfg.section("evaluate")))
        source.close()

    session.root.joinpath("scorecard.json").write_text(
        json.dumps(scorecards, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _print_summary(log, scorecards)
    session.finish({"elapsed_seconds": round(time.time() - started, 1), "scorecard": "scorecard.json"})
    try:
        from .report import generate_report
        report_path = generate_report(session.root)
        log.info(f"HTML 报告：{report_path}")
    except ImportError:
        log.info("HTML 报告模块尚未启用（步骤6），查看 scorecard.json / events.jsonl")
    log.info(f"会话归档：{session.root}")
    return 0


def run_av_dataset(cfg, session, spec, log):
    """音画不同步数据集：L1 被动测量 + L3 主动测试片自检双链路对账。

    内容本身没有语义声画锚点（纯 BGM）时，L1 会诚实地给出低置信/不可测结论——
    这是内容性质而非缺陷；测量链路正确性由 L3（已知偏移测试片）验证。
    """
    from pathlib import Path
    from .detectors.avsync import AVSyncAnalyzer, L3Analyzer, generate_test_clip
    analyzer = AVSyncAnalyzer(cfg.section("avsync"))
    log.info(f"评测数据集：{spec.name}（音画不同步，{len(spec.cases)} 个 case）")
    rows = []
    for case in spec.cases:
        video_path = spec.path / case["file"]
        result = analyzer.measure(video_path)
        expected = case.get("offset_ms")
        row = {"file": case["file"], "desc": case.get("desc", ""),
               "expected_offset_ms": expected, "measured_offset_ms": result.get("offset_ms"),
               "mad_ms": result.get("mad_ms"), "pairs": result.get("pairs"),
               "cluster_pairs": result.get("cluster_pairs"),
               "mass_fraction": result.get("mass_fraction"),
               "confidence": result.get("confidence"), "direction": result.get("direction")}
        if expected is not None and result.get("offset_ms") is not None:
            error_ms = round(result["offset_ms"] - expected, 1)
            row["error_ms"] = error_ms
            row["error_frames"] = round(abs(error_ms) / 1000 * spec.fps, 2)
            row["sign_correct"] = (result["offset_ms"] > 0) == (expected > 0)
        ev_dir = session.evidence_dir(spec.name, f"av_{Path(case['file']).stem}")
        session.write_evidence_json(ev_dir, "measurement.json", {**row, "raw": result})
        row["evidence_dir"] = session.rel(ev_dir)
        session.record_event({"type": "av_offset", "dataset": spec.name,
                              "start_frame": 0, "end_frame": 0,
                              "start_t_sec": 0, "end_t_sec": result.get("video_seconds", 0),
                              "duration_sec": result.get("video_seconds", 0),
                              "level": "INFO", "trigger": "AV_OFFSET_MEASURE",
                              "status": "confirmed", "metrics": row})
        rows.append(row)
        log.info(f"  {case['file']}: 期望 {expected}ms → 实测 {result.get('offset_ms')}ms "
                 f"(cluster={result.get('cluster_pairs')}, mass={result.get('mass_fraction')}, "
                 f"conf={result.get('confidence')})")
    matched_rows = [r for r in rows if "error_ms" in r]
    summary = {
        "dataset": spec.name, "kind": "av", "case_count": len(rows),
        "cases": rows,
        "sign_correct_count": sum(1 for r in rows if r.get("sign_correct")),
        "within_one_frame": sum(1 for r in matched_rows if r.get("error_frames", 99) <= 1.0),
        "max_abs_error_frames": round(max((abs(r["error_frames"]) for r in matched_rows), default=0), 2),
        "inconclusive_count": sum(1 for r in rows if r.get("confidence") in ("low", "none")),
        "note": "L1 被动估计依赖内容中的语义声画锚点；低置信表示该内容无锚点，链路正确性见 L3 自检",
    }
    log.info(f"[{spec.name}] L1 被动：方向正确 {summary['sign_correct_count']}/{len(rows)}，"
             f"低置信(内容无锚点) {summary['inconclusive_count']}/{len(rows)}")

    # L3 主动测试片自检（同一组偏移真值）
    l3_rows = []
    l3_dir = session.evidence_dir(spec.name, "l3_selftest")
    for case in spec.cases:
        offset = case["offset_ms"]
        clip_path = l3_dir / f"l3_{case['file']}"
        generate_test_clip(clip_path, offset)
        result = L3Analyzer(cfg.section("avsync")).measure(clip_path)
        error_frames = round(abs(result.get("offset_ms", 0) - offset) / 1000 * spec.fps, 2)
        l3_rows.append({"file": Path(case['file']).name, "expected_offset_ms": offset,
                        "measured_offset_ms": result.get("offset_ms"),
                        "error_frames": error_frames,
                        "pass": error_frames <= 1.0,
                        "evidence": session.rel(clip_path)})
        log.info(f"  L3 {case['file']}: 期望 {offset}ms → 实测 {result.get('offset_ms')}ms "
                 f"(误差 {error_frames} 帧)")
    session.write_evidence_json(l3_dir, "l3_results.json", l3_rows)
    l3_card = {"dataset": spec.name, "kind": "av_l3", "cases": l3_rows,
               "pass_count": sum(1 for r in l3_rows if r["pass"]),
               "case_count": len(l3_rows),
               "max_abs_error_frames": round(max(r["error_frames"] for r in l3_rows), 2)}
    log.info(f"[{spec.name}] L3 主动：通过 {l3_card['pass_count']}/{l3_card['case_count']}，"
             f"最大误差 {l3_card['max_abs_error_frames']} 帧")
    return [summary, l3_card]


def _print_summary(log, scorecards):
    log.info("=" * 78)
    for card in scorecards:
        if card.get("kind") == "av":
            log.info(f"[{card['dataset']}] 音画 L1 被动：方向正确 {card['sign_correct_count']}/{card['case_count']}"
                     f"，低置信 {card.get('inconclusive_count', 0)}/{card['case_count']}（内容无语义锚点时属预期）")
            continue
        if card.get("kind") == "av_l3":
            log.info(f"[{card['dataset']}] 音画 L3 主动：通过 {card['pass_count']}/{card['case_count']}"
                     f"，最大误差 {card['max_abs_error_frames']} 帧")
            continue
        if card.get("kind") == "baseline":
            log.info(f"[{card['dataset']}] 正常基线：检出事件 {card['detected_count']}（期望 0）")
            continue
        log.info(f"[{card['dataset']}] 真值 {card['gt_count']} → 匹配 {card['matched']} "
                 f"| recall={card['recall']} precision={card['precision']} "
                 f"| 平均IoU={card.get('mean_iou')} 起点误差={card.get('mean_abs_start_err_frames')}帧 "
                 f"| 漏检 {len(card['misses'])} 误报 {len(card['false_positives'])}")
    log.info("=" * 78)


def cmd_run(args):
    from .archive import Session
    from .runner import run_field
    cfg = load_config(args.profile, args.overrides)
    output_root = args.output_root or cfg.get("output_root", "output")
    session = Session(output_root, mode="field", config_snapshot=cfg.as_dict())
    run_field(cfg, session)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        from .report import generate_report
        print(generate_report(args.session))
        return 0
    if args.command == "inject-corruption":
        from .inject import build_corruption_dataset
        build_corruption_dataset(args.base, args.out, args.seed)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
