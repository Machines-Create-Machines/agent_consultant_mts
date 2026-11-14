"""
MTS Energytool consultant — thin entry point.

predict(question, history=None) -> dict  (eval contract)
CLI:
  python solution.py --query "вопрос"
  python solution.py --input data/demo_questions.csv --output answers.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from agent.llm import load_dotenv
from agent.pipeline import run
from agent.store import cold_start


def predict(question: str, history: list[dict] | None = None) -> dict:
    load_dotenv()
    return run(question, history)


def _run_csv(input_path: Path, output_path: Path) -> None:
    load_dotenv()
    cold_start()
    # One TCP probe up front; circuit breaker keeps later rows offline-fast
    from agent.llm import probe_connectivity

    probe_connectivity()
    with input_path.open(encoding="utf-8-sig", newline="") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "answer"])
        writer.writeheader()
        for row in reader:
            q = row.get("question") or row.get("Question") or ""
            rid = row.get("id") or row.get("Id") or ""
            out = predict(q, history=[])
            answer = (out.get("answer") or "").strip() or "В базе знаний нет ответа по этому вопросу."
            writer.writerow({"id": rid, "answer": answer})


def _print_one(out: dict) -> None:
    print(out.get("answer") or "")
    print(f"behavior={out.get('behavior')} llm_calls={out.get('llm_calls')}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="MTS Energytool consultant")
    p.add_argument("--query", "-q", default=None, help="Single user question")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.input or args.output:
        if not (args.input and args.output):
            p.error("both --input and --output are required")
        _run_csv(args.input, args.output)
        return
    if not args.query:
        p.error("pass --query or --input/--output")
    _print_one(predict(args.query))


if __name__ == "__main__":
    main()
