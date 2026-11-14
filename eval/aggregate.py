"""Case → slice → global aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def speed_score(p95_ms: float | None, target_ms: float, fail_ms: float) -> float:
    if p95_ms is None:
        return 0.0
    if p95_ms <= target_ms:
        return 1.0
    if p95_ms >= fail_ms:
        return 0.0
    return 1.0 - (p95_ms - target_ms) / (fail_ms - target_ms)


def behavior_confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        gold = r.get("gold_behavior") or "?"
        pred = r.get("predicted_behavior") or ("error" if r.get("error") else "?")
        matrix[str(gold)][str(pred)] += 1
    return {g: dict(preds) for g, preds in matrix.items()}


def _slice_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def col(name: str) -> list[float]:
        out = []
        for r in rows:
            v = r.get(name)
            if v is None:
                continue
            out.append(float(v))
        return out

    errors = sum(1 for r in rows if r.get("error"))
    viols = sum(1 for r in rows if r.get("violation"))
    success = [r for r in rows if not r.get("error")]
    lat = [float(r["latency_ms"]) for r in success if r.get("latency_ms") is not None]
    cold = [
        float(r["latency_ms"])
        for r in success
        if r.get("cold_start") and r.get("latency_ms") is not None
    ]
    warm = [
        float(r["latency_ms"])
        for r in success
        if (not r.get("cold_start")) and r.get("latency_ms") is not None
    ]

    nugget_vals = col("nugget_recall")
    # micro: weight by nugget_weight_sum * recall ≈ sum confirmed weights / sum weights
    micro_num = 0.0
    micro_den = 0.0
    for r in rows:
        if r.get("nugget_recall") is None:
            continue
        wsum = float(r.get("nugget_weight_sum") or 0)
        micro_num += float(r["nugget_recall"]) * wsum
        micro_den += wsum

    return {
        "n": n,
        "error_rate": errors / n,
        "violation_rate": viols / n,
        "behavior_accuracy": _mean([1.0 if r.get("behavior_ok") else 0.0 for r in rows]),
        "recall_at_k": _mean(col("recall_at_k")),
        "hit_at_1": _mean(col("hit_at_1")),
        "product_precision": _mean(col("product_precision")),
        "product_recall": _mean(col("product_recall")),
        "product_f1": _mean(col("product_f1")),
        "groundedness": _mean(col("groundedness")),
        "length_ok_rate": _mean([1.0 if r.get("length_ok") else 0.0 for r in rows]),
        "nugget_recall_macro": _mean(nugget_vals) if nugget_vals else None,
        "nugget_recall_micro": (micro_num / micro_den) if micro_den else None,
        "judge_win_rate": _mean(col("judge_win_score")),
        "latency_p50_ms": _percentile(lat, 50),
        "latency_p95_ms": _percentile(lat, 95),
        "latency_p99_ms": _percentile(lat, 99),
        "cold_start_p50_ms": _percentile(cold, 50),
        "warm_latency_p50_ms": _percentile(warm, 50),
        "llm_calls_mean": _mean([float(r.get("llm_calls") or 0) for r in rows]),
    }


def compute_composite(global_m: dict[str, Any], weights: dict[str, float], speed: float) -> float:
    parts = {
        "judge_win_rate": global_m.get("judge_win_rate"),
        "nugget_recall_macro": global_m.get("nugget_recall_macro"),
        "behavior_accuracy": global_m.get("behavior_accuracy"),
        "recall_at_k": global_m.get("recall_at_k"),
        "product_f1": global_m.get("product_f1"),
        "speed_score": speed,
    }
    # If judge metrics missing (--no-judge), renormalize over available weighted parts.
    available = {k: v for k, v in parts.items() if v is not None}
    if not available:
        return 0.0
    wsum = sum(weights.get(k, 0.0) for k in available)
    if wsum <= 0:
        return 0.0
    score = 0.0
    for k, v in available.items():
        score += (weights.get(k, 0.0) / wsum) * float(v)
    return score


def evaluate_gates(global_m: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    checks = {
        "error_rate": float(global_m.get("error_rate") or 0) <= gates["error_rate_max"],
        "violation_rate": float(global_m.get("violation_rate") or 0) <= gates["violation_rate_max"],
        "behavior_accuracy": float(global_m.get("behavior_accuracy") or 0)
        >= gates["behavior_accuracy_min"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "thresholds": gates,
    }


def aggregate(
    rows: list[dict[str, Any]],
    weights: dict[str, float],
    gates: dict[str, float],
    speed_cfg: dict[str, float],
) -> dict[str, Any]:
    global_m = _slice_metrics(rows)
    spd = speed_score(
        global_m.get("latency_p95_ms"),
        float(speed_cfg["p95_target_ms"]),
        float(speed_cfg["p95_fail_ms"]),
    )
    global_m["speed_score"] = spd
    global_m["composite"] = compute_composite(global_m, weights, spd)
    gate_result = evaluate_gates(global_m, gates)

    by_behavior: dict[str, Any] = {}
    by_tag: dict[str, Any] = {}
    by_source: dict[str, Any] = {}

    behavior_groups: dict[str, list] = defaultdict(list)
    tag_groups: dict[str, list] = defaultdict(list)
    source_groups: dict[str, list] = defaultdict(list)

    for r in rows:
        behavior_groups[str(r.get("gold_behavior") or "?")].append(r)
        for t in str(r.get("tags") or "").split("|"):
            if t:
                tag_groups[t].append(r)
        src = str(r.get("source_file") or "?")
        source_groups[src].append(r)

    for k, grp in sorted(behavior_groups.items()):
        by_behavior[k] = _slice_metrics(grp)
    for k, grp in sorted(tag_groups.items()):
        by_tag[k] = _slice_metrics(grp)
    for k, grp in sorted(source_groups.items()):
        by_source[k] = _slice_metrics(grp)

    return {
        "global": global_m,
        "gates": gate_result,
        "slices": {
            "by_behavior": by_behavior,
            "by_tag": by_tag,
            "by_source_file": by_source,
        },
        "behavior_confusion": behavior_confusion(rows),
    }
