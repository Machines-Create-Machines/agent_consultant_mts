"""Fit simple evidence-gate thresholds from gold chunks (offline helper).

Usage:
  PYTHONPATH=. python3 scripts/calibrate_retrieval.py

Writes suggested GATE_* values to stdout. Does not modify config automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.fuse import fuse
from agent.retrieve import retrieve
from agent.store import cold_start
from agent.understand import fallback_understand

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "questions_dataset.jsonl"


def main() -> None:
    store = cold_start(force=True)
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    pos: list[float] = []
    neg: list[float] = []
    for row in rows[:80]:
        q = row.get("question") or ""
        gold = set(row.get("gold_chunks") or [])
        u = fallback_understand(q)
        ranked = fuse(store, retrieve(store, u))
        if not ranked:
            continue
        top = ranked[0]
        hit = any(g == top.chunk_id or g in top.chunk_id for g in gold) or any(
            top.chunk_id.startswith(g.split("#")[0]) and g in {c.chunk_id for c in ranked[:5]}
            for g in gold
        )
        in_top5 = any(
            any(g == c.chunk_id or c.chunk_id.endswith(g.split("#")[-1]) for g in gold)
            for c in ranked[:5]
        ) or any(c.chunk_id in gold for c in ranked[:5])
        if in_top5:
            pos.append(top.score)
        else:
            neg.append(top.score)
    pos.sort()
    neg.sort()
    print("pos_n", len(pos), "neg_n", len(neg))
    if pos:
        print("pos p50", pos[len(pos) // 2], "p20", pos[max(0, len(pos) // 5)])
    if neg:
        print("neg p80", neg[min(len(neg) - 1, int(len(neg) * 0.8))])
    # Suggested RRF thresholds
    if pos:
        print("suggested GATE_ANSWER_RRF", round(pos[max(0, len(pos) // 5)], 5))
        print("suggested GATE_PARTIAL_RRF", round(pos[0] * 0.7, 5))


if __name__ == "__main__":
    main()
