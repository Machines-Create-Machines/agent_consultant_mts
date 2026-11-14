"""Compare current run to a baseline run by case id."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CASE_SCORE_KEYS = [
    "judge_win_score",
    "nugget_recall",
    "recall_at_k",
    "product_f1",
    "behavior_ok",
]


def _load_cases_csv(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = row
    return out


def _f(v: Any) -> float | None:
    if v is None or v == "" or v == "None":
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        if str(v).lower() in {"true", "1"}:
            return 1.0
        if str(v).lower() in {"false", "0"}:
            return 0.0
        return None


def case_score(row: dict[str, Any]) -> float:
    parts = []
    for k in CASE_SCORE_KEYS:
        v = _f(row.get(k))
        if v is not None:
            parts.append(v)
    return sum(parts) / len(parts) if parts else 0.0


def compare_runs(
    current_dir: Path,
    baseline_dir: Path,
    delta_threshold: float,
    aggregate_metrics: list[str],
) -> dict[str, Any]:
    cur_cases = _load_cases_csv(current_dir / "cases.csv")
    base_cases = _load_cases_csv(baseline_dir / "cases.csv")
    cur_metrics = json.loads((current_dir / "metrics.json").read_text(encoding="utf-8"))
    base_metrics = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))

    improved, worsened, broken = [], [], []
    common = sorted(set(cur_cases) & set(base_cases))
    for cid in common:
        c, b = cur_cases[cid], base_cases[cid]
        c_err = str(c.get("error")) in {"1", "True", "true"}
        b_err = str(b.get("error")) in {"1", "True", "true"}
        if c_err and not b_err:
            broken.append({"id": cid, "reason": c.get("error_type") or "error"})
            continue
        cs, bs = case_score(c), case_score(b)
        delta = cs - bs
        if abs(delta) < delta_threshold:
            continue
        item = {"id": cid, "delta": round(delta, 4), "from": round(bs, 4), "to": round(cs, 4)}
        if delta > 0:
            improved.append(item)
        else:
            worsened.append(item)

    improved.sort(key=lambda x: -x["delta"])
    worsened.sort(key=lambda x: x["delta"])

    agg_deltas = {}
    cg = cur_metrics.get("global") or {}
    bg = base_metrics.get("global") or {}
    for m in aggregate_metrics:
        cv, bv = _f(cg.get(m)), _f(bg.get(m))
        if cv is None or bv is None:
            agg_deltas[m] = None
        else:
            d = cv - bv
            agg_deltas[m] = {
                "delta": round(d, 4),
                "from": round(bv, 4),
                "to": round(cv, 4),
                "significant": abs(d) >= delta_threshold,
            }

    return {
        "baseline_run": baseline_dir.name,
        "improved": improved,
        "worsened": worsened,
        "broken": broken,
        "aggregate_deltas": agg_deltas,
        "n_common": len(common),
    }


def render_diff_md(path: Path, diff: dict[str, Any], delta_threshold: float) -> str:
    lines = [
        f"# Diff vs `{diff['baseline_run']}`",
        "",
        f"Common cases: {diff['n_common']}. Significance threshold: ±{delta_threshold}.",
        "",
        "## Aggregate deltas",
        "",
        "| Metric | from | to | delta | sig |",
        "|---|---:|---:|---:|:---:|",
    ]
    for m, info in (diff.get("aggregate_deltas") or {}).items():
        if info is None:
            lines.append(f"| {m} | — | — | — | — |")
        else:
            lines.append(
                f"| {m} | {info['from']} | {info['to']} | {info['delta']:+} | "
                f"{'yes' if info['significant'] else 'no'} |"
            )
    lines.append("")

    def _section(title: str, items: list[dict[str, Any]], key: str = "delta") -> None:
        lines.append(f"## {title} ({len(items)})")
        lines.append("")
        if not items:
            lines.append("_None._")
            lines.append("")
            return
        for it in items[:50]:
            if "reason" in it:
                lines.append(f"- `{it['id']}` — {it['reason']}")
            else:
                lines.append(
                    f"- `{it['id']}` — {it.get(key, 0):+0.3f} ({it.get('from')} → {it.get('to')})"
                )
        lines.append("")

    _section("Improved", diff.get("improved") or [])
    _section("Worsened", diff.get("worsened") or [])
    _section("Broken", diff.get("broken") or [])

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    # brief for summary.md
    sig = [
        f"{m}: {info['delta']:+.3f}"
        for m, info in (diff.get("aggregate_deltas") or {}).items()
        if info and info.get("significant")
    ]
    brief = (
        f"vs `{diff['baseline_run']}`: "
        f"improved={len(diff.get('improved') or [])}, "
        f"worsened={len(diff.get('worsened') or [])}, "
        f"broken={len(diff.get('broken') or [])}."
    )
    if sig:
        brief += " Significant: " + ", ".join(sig) + "."
    return brief
