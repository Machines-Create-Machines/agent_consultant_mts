#!/usr/bin/env python3
"""Build data/baseline_answers.csv: BM25 top-3 + one short LLM call per case."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import ChatClient, load_dotenv, model_available  # noqa: E402
from agent.store import cold_start  # noqa: E402
from eval.loader import load_dataset  # noqa: E402

OUT = ROOT / "data" / "baseline_answers.csv"
SYSTEM = (
    "Ты консультант МТС Energytool. Ответь по-русски на «вы» в 2–3 предложениях "
    "только по приведённому контексту. Сначала прямой ответ, затем один уточняющий "
    "вопрос если нужно. Без markdown и без маркеров источников."
)


def _context(store, question: str, k: int = 3) -> str:
    if store.bm25 is None:
        return ""
    q = store.expand_query_aliases(question)
    parts: list[str] = []
    for idx, _sc in store.bm25.topk(q, k=k):
        cid = store.chunk_ids[idx]
        ch = store.chunks.get(cid)
        if not ch:
            continue
        text = (ch.text_raw or "").strip()
        if text:
            parts.append(f"{ch.heading}: {text[:800]}")
    return "\n\n".join(parts)


def main() -> None:
    load_dotenv()
    store = cold_start()
    ds = load_dataset(ROOT / "data" / "questions_dataset.jsonl", holdout_pct=30, split_seed=42)
    cases = ds.select(split="all")
    client = ChatClient(temperature=0.0, max_tokens=350) if model_available() else None
    rows: list[dict[str, str]] = []
    for i, case in enumerate(cases):
        q = case.raw["question"]
        ctx = _context(store, q)
        if client and ctx:
            try:
                answer = client.chat(
                    [
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": f"Вопрос: {q}\n\nКонтекст:\n{ctx}",
                        },
                    ]
                ).strip()
            except Exception as exc:  # noqa: BLE001
                print(f"llm fail {case.id}: {exc}", flush=True)
                answer = ""
        else:
            answer = ""
        if not answer and ctx:
            answer = ctx.split("\n\n")[0][:400]
        if not answer:
            answer = "В базе знаний нет ответа по этому вопросу."
        rows.append({"id": case.id, "answer": answer})
        if (i + 1) % 10 == 0:
            print(f"baseline {i + 1}/{len(cases)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} rows={len(rows)}")


if __name__ == "__main__":
    main()
