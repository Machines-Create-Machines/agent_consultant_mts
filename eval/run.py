"""CLI entry: python -m eval.run [options]"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eval.aggregate import aggregate
from eval.checkers import score_case
from eval.diff import compare_runs, render_diff_md
from eval.judge import JudgeCache, enrich_with_judge, load_baseline_answers
from eval.kb import load_chunks
from eval.loader import load_dataset
from eval.model_client import ChatClient, ModelConfig, model_available, resolve_model_env
from eval.report import (
    append_index,
    render_failures,
    render_summary,
    top_issues,
    write_cases_csv,
    write_json,
    write_jsonl,
)
from eval.runner import run_cases

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def _git_commit(root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _prompt_hashes() -> dict[str, str]:
    from eval import judge as judge_mod

    def h(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    return {
        "nugget_system": h(judge_mod.NUGGET_SYSTEM),
        "pairwise_system": h(judge_mod.PAIRWISE_SYSTEM),
        "prompt_version": judge_mod.PROMPT_VERSION_DEFAULT,
    }


def _make_run_id(label: str, runs_dir: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base = f"{stamp}__{label}"
    candidate = base
    n = 2
    while (runs_dir / candidate).exists():
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autonomous eval contour for solution.py")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    p.add_argument("--tag", default=None, help="Run label embedded in run_id (ablation name)")
    p.add_argument("--case-tag", default=None, help="Filter cases that contain this dataset tag")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-judge", action="store_true", help="Skip LLM judge (fast loop)")
    p.add_argument("--baseline-run", default=None, help="Prior run_id under eval/runs/")
    p.add_argument("--no-cache", action="store_true", help="Disable judge disk cache")
    p.add_argument("--label", default=None, help="Alias for --tag")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--solution-module", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config)
    project_root = ROOT
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    label = args.tag or args.label or ("smoke" if args.limit else args.split)
    # sanitize label for filesystem
    label = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label) or "run"
    run_id = _make_run_id(label, runs_dir)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=False)

    dataset_path = project_root / cfg["dataset_path"]
    ds = load_dataset(
        dataset_path,
        holdout_pct=int(cfg["holdout_pct"]),
        split_seed=int(cfg["split_seed"]),
    )
    cases = ds.select(split=args.split, tag=args.case_tag, limit=args.limit)
    if not cases:
        print("No cases selected.", file=sys.stderr)
        return 2

    kb_dir = project_root / cfg["knowledge_base_dir"]
    chunks = load_chunks(kb_dir)

    judge_enabled = not args.no_judge
    if judge_enabled and not model_available():
        print(
            "WARNING: LLM env not set; continuing with --no-judge. "
            "Set OPENAI_API_KEY + OPENAI_MODEL for full judge run.",
            file=sys.stderr,
        )
        judge_enabled = False

    module_name = args.solution_module or cfg.get("solution_module") or "solution"
    concurrency = args.concurrency or int(cfg["concurrency"])

    print(f"run_id={run_id} cases={len(cases)} split={args.split} judge={judge_enabled}")
    answers = run_cases(
        cases=cases,
        project_root=project_root,
        module_name=module_name,
        concurrency=concurrency,
        timeout_sec=float(cfg["timeout_sec"]),
        retries=int(cfg["retries"]),
        isolation=str(cfg.get("isolation") or "call"),
    )
    answers_by_id = {a["id"]: a for a in answers}
    write_jsonl(run_dir / "answers.jsonl", answers)

    rows = [
        score_case(
            case=c,
            answer_rec=answers_by_id[c.id],
            chunks=chunks,
            k=int(cfg["k"]),
            max_words=int(cfg["max_answer_words"]),
        )
        for c in cases
    ]

    judge_stats = {"enabled": judge_enabled, "cache_hits": 0, "cache_misses": 0, "llm_calls": 0}
    if judge_enabled:
        model_cfg = ModelConfig(
            temperature=float(cfg["model"]["temperature"]),
            max_tokens=int(cfg["model"]["max_tokens"]),
            timeout_sec=float(cfg["model"]["timeout_sec"]),
        )
        client = ChatClient(model_cfg)
        cache = JudgeCache(
            root=project_root / cfg["judge"]["cache_dir"],
            enabled=not args.no_cache,
        )
        baseline_path = project_root / cfg.get("baseline_answers_path", "demo_output.csv")
        baseline_answers = load_baseline_answers(baseline_path)
        rows = enrich_with_judge(
            cases=cases,
            answers_by_id=answers_by_id,
            rows=rows,
            baseline_by_id=baseline_answers,
            client=client,
            cache=cache,
            prompt_version=str(cfg["judge"].get("prompt_version") or "v1"),
            soft_violation_check=bool(cfg["judge"].get("soft_violation_check")),
        )
        judge_stats.update(
            {
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "llm_calls": client.calls,
            }
        )

    write_cases_csv(run_dir / "cases.csv", rows)
    metrics = aggregate(
        rows=rows,
        weights=cfg["composite_weights"],
        gates=cfg["gates"],
        speed_cfg=cfg["speed_score"],
    )
    metrics["judge"] = judge_stats
    write_json(run_dir / "metrics.json", metrics)

    env = resolve_model_env()
    manifest = {
        "run_id": run_id,
        "label": label,
        "split": args.split,
        "tag": label,
        "case_tag": args.case_tag,
        "limit": args.limit,
        "n_cases": len(cases),
        "dataset_path": str(dataset_path.relative_to(project_root)),
        "dataset_hash": ds.content_hash,
        "prompt_hashes": _prompt_hashes(),
        "git_commit": _git_commit(project_root),
        "model": env["model"] if judge_enabled else None,
        "base_url_set": bool(env["base_url"]),
        "judge_enabled": judge_enabled,
        "config": cfg,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # never store secrets
    }
    write_json(run_dir / "manifest.json", manifest)

    diff_brief = None
    baseline_run = args.baseline_run
    if baseline_run is None:
        # auto: previous run in index if any
        index_path = runs_dir / "index.csv"
        if index_path.exists():
            import csv

            with index_path.open(encoding="utf-8") as f:
                prev = list(csv.DictReader(f))
            if prev:
                baseline_run = prev[-1]["run_id"]
    if baseline_run:
        bdir = runs_dir / baseline_run
        if bdir.exists() and (bdir / "cases.csv").exists():
            diff = compare_runs(
                current_dir=run_dir,
                baseline_dir=bdir,
                delta_threshold=float(cfg["regression"]["delta_threshold"]),
                aggregate_metrics=list(cfg["regression"]["metrics"]),
            )
            write_json(run_dir / "diff.json", diff)
            diff_brief = render_diff_md(
                run_dir / "diff.md",
                diff,
                float(cfg["regression"]["delta_threshold"]),
            )
        else:
            (run_dir / "diff.md").write_text(
                f"# Diff\n\nBaseline `{baseline_run}` not found.\n",
                encoding="utf-8",
            )
    else:
        (run_dir / "diff.md").write_text("# Diff\n\nNo baseline run.\n", encoding="utf-8")

    issues = top_issues(rows, int(cfg["summary_top_issues"]))
    render_summary(
        run_dir / "summary.md",
        run_id=run_id,
        metrics=metrics,
        top_issues=issues,
        diff_brief=diff_brief,
        judge_enabled=judge_enabled,
    )
    render_failures(run_dir / "failures.md", rows, int(cfg["failures_top_n"]))

    g = metrics["global"]
    append_index(
        runs_dir / "index.csv",
        {
            "run_id": run_id,
            "label": label,
            "split": args.split,
            "n": g.get("n"),
            "gates_pass": int(metrics["gates"]["pass"]),
            "composite": g.get("composite"),
            "judge_win_rate": g.get("judge_win_rate"),
            "nugget_recall_macro": g.get("nugget_recall_macro"),
            "behavior_accuracy": g.get("behavior_accuracy"),
            "recall_at_k": g.get("recall_at_k"),
            "product_f1": g.get("product_f1"),
            "speed_score": g.get("speed_score"),
            "violation_rate": g.get("violation_rate"),
            "error_rate": g.get("error_rate"),
            "latency_p95_ms": g.get("latency_p95_ms"),
            "judge_enabled": int(judge_enabled),
            "model": env["model"] if judge_enabled else "",
            "git_commit": manifest["git_commit"],
            "dataset_hash": ds.content_hash[:12],
        },
    )

    print(f"Wrote {run_dir}")
    print(
        f"gates={'PASS' if metrics['gates']['pass'] else 'FAIL'} "
        f"composite={g.get('composite')} "
        f"error_rate={g.get('error_rate')} "
        f"violation_rate={g.get('violation_rate')}"
    )
    return 0 if metrics["gates"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
