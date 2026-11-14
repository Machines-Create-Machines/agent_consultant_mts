"""Write run artifacts: csv/json/md reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # stable column order
    keys: list[str] = []
    seen = set()
    preferred = [
        "id",
        "split",
        "tags",
        "source_file",
        "gold_behavior",
        "predicted_behavior",
        "behavior_ok",
        "recall_at_k",
        "hit_at_1",
        "product_precision",
        "product_recall",
        "product_f1",
        "groundedness",
        "violation",
        "violation_items",
        "length_ok",
        "word_count",
        "error",
        "error_type",
        "latency_ms",
        "cold_start",
        "llm_calls",
        "nugget_recall",
        "nugget_hits",
        "judge_vs_gold",
        "judge_vs_baseline",
        "judge_win_score",
        "soft_violation",
    ]
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {k: _csv_val(r.get(k)) for k in keys}
            w.writerow(flat)


def _csv_val(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return round(v, 6)
    return v


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_summary(
    path: Path,
    run_id: str,
    metrics: dict[str, Any],
    top_issues: list[dict[str, Any]],
    diff_brief: str | None,
    judge_enabled: bool,
) -> None:
    g = metrics["global"]
    gates = metrics["gates"]
    status = "PASS" if gates["pass"] else "FAIL"
    lines = [
        f"# Eval summary — `{run_id}`",
        "",
        f"**Gates: {status}**",
        "",
        "| Gate | Value | Threshold | OK |",
        "|---|---:|---:|:---:|",
        f"| error_rate | {_fmt(g.get('error_rate'))} | ≤ {gates['thresholds']['error_rate_max']} | "
        f"{'✓' if gates['checks']['error_rate'] else '✗'} |",
        f"| violation_rate | {_fmt(g.get('violation_rate'))} | ≤ {gates['thresholds']['violation_rate_max']} | "
        f"{'✓' if gates['checks']['violation_rate'] else '✗'} |",
        f"| behavior_accuracy | {_fmt(g.get('behavior_accuracy'))} | ≥ {gates['thresholds']['behavior_accuracy_min']} | "
        f"{'✓' if gates['checks']['behavior_accuracy'] else '✗'} |",
        "",
        f"**Composite:** {_fmt(g.get('composite'))}"
        + (" (judge off — renormalized)" if not judge_enabled else ""),
        "",
        "## Global metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n | {g.get('n')} |",
        f"| recall@k | {_fmt(g.get('recall_at_k'))} |",
        f"| hit@1 | {_fmt(g.get('hit_at_1'))} |",
        f"| product_F1 | {_fmt(g.get('product_f1'))} |",
        f"| groundedness | {_fmt(g.get('groundedness'))} |",
        f"| length_ok_rate | {_fmt(g.get('length_ok_rate'))} |",
        f"| nugget_recall_macro | {_fmt(g.get('nugget_recall_macro'))} |",
        f"| nugget_recall_micro | {_fmt(g.get('nugget_recall_micro'))} |",
        f"| judge_win_rate | {_fmt(g.get('judge_win_rate'))} |",
        f"| latency_p50_ms | {_fmt(g.get('latency_p50_ms'), 1)} |",
        f"| latency_p95_ms | {_fmt(g.get('latency_p95_ms'), 1)} |",
        f"| latency_p99_ms | {_fmt(g.get('latency_p99_ms'), 1)} |",
        f"| cold_start_p50_ms | {_fmt(g.get('cold_start_p50_ms'), 1)} |",
        f"| speed_score | {_fmt(g.get('speed_score'))} |",
        f"| llm_calls_mean | {_fmt(g.get('llm_calls_mean'), 2)} |",
        "",
        "## Behavior confusion (gold × predicted)",
        "",
    ]
    conf = metrics.get("behavior_confusion") or {}
    preds = sorted({p for row in conf.values() for p in row})
    golds = sorted(conf.keys())
    if golds and preds:
        header = "| gold \\ pred | " + " | ".join(preds) + " |"
        sep = "|---|" + "|".join(["---:"] * len(preds)) + "|"
        lines.extend([header, sep])
        for gname in golds:
            cells = [str(conf.get(gname, {}).get(p, 0)) for p in preds]
            lines.append(f"| {gname} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(["## Slices by tag", "", "| Tag | n | behavior_acc | recall@k | product_F1 | viol |", "|---|---:|---:|---:|---:|---:|"])
    for tag, sm in (metrics.get("slices", {}).get("by_tag") or {}).items():
        lines.append(
            f"| {tag} | {sm.get('n')} | {_fmt(sm.get('behavior_accuracy'))} | "
            f"{_fmt(sm.get('recall_at_k'))} | {_fmt(sm.get('product_f1'))} | {_fmt(sm.get('violation_rate'))} |"
        )
    lines.append("")

    lines.extend(["## Top issues", ""])
    if not top_issues:
        lines.append("_No flagged issues._")
    else:
        for i, issue in enumerate(top_issues, 1):
            lines.append(
                f"{i}. `{issue['id']}` — {issue['reason']} "
                f"(recall@k={_fmt(issue.get('recall_at_k'))}, behavior_ok={issue.get('behavior_ok')})"
            )
    lines.append("")

    if diff_brief:
        lines.extend(["## Delta vs baseline", "", diff_brief, ""])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_failures(path: Path, rows: list[dict[str, Any]], top_n: int) -> None:
    scored: list[tuple[float, dict[str, Any], str]] = []
    for r in rows:
        reasons = []
        score = 0.0
        if r.get("error"):
            reasons.append(f"error:{r.get('error_type')}")
            score += 5
        if r.get("violation"):
            reasons.append(f"violation:{r.get('violation_items')}")
            score += 4
        if not r.get("behavior_ok"):
            reasons.append("bad_behavior")
            score += 3
        if float(r.get("recall_at_k") or 0) < 0.5:
            reasons.append("low_recall")
            score += 2
        if r.get("nugget_recall") is not None and float(r["nugget_recall"]) < 0.5:
            reasons.append("low_nuggets")
            score += 2
        if reasons:
            scored.append((score, r, "; ".join(reasons)))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    lines = ["# Failures", ""]
    if not scored:
        lines.append("_No failures flagged._")
    else:
        for score, r, reason in scored[:top_n]:
            lines.append(f"## {r['id']}")
            lines.append("")
            lines.append(f"- reason: {reason}")
            lines.append(f"- tags: {r.get('tags')}")
            lines.append(f"- predicted_behavior: {r.get('predicted_behavior')}")
            lines.append(f"- recall_at_k: {r.get('recall_at_k')}")
            lines.append(f"- product_f1: {r.get('product_f1')}")
            lines.append(f"- nugget_recall: {r.get('nugget_recall')}")
            lines.append(f"- judge_win_score: {r.get('judge_win_score')}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_issues(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    items = []
    for r in rows:
        reason = None
        severity = 0
        if r.get("error"):
            reason, severity = f"error ({r.get('error_type')})", 5
        elif r.get("violation"):
            reason, severity = f"violation ({r.get('violation_items')})", 4
        elif not r.get("behavior_ok"):
            reason, severity = "behavior not allowed", 3
        elif float(r.get("recall_at_k") or 1) < 0.5:
            reason, severity = "weak retrieval", 2
        if reason:
            items.append({**r, "reason": reason, "_sev": severity})
    items.sort(key=lambda x: (-x["_sev"], x["id"]))
    return [{k: v for k, v in it.items() if k != "_sev"} for it in items[:n]]


def append_index(index_path: Path, row: dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "label",
        "split",
        "n",
        "gates_pass",
        "composite",
        "judge_win_rate",
        "nugget_recall_macro",
        "behavior_accuracy",
        "recall_at_k",
        "product_f1",
        "speed_score",
        "violation_rate",
        "error_rate",
        "latency_p95_ms",
        "judge_enabled",
        "model",
        "git_commit",
        "dataset_hash",
    ]
    exists = index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: _csv_val(row.get(k)) for k in fields})
