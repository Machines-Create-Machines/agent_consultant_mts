"""
MTS Energytool consultant — platform entry point.

Platform (no args):
  reads questions.csv (cwd or DATA_DIR), writes submission.csv (id,answer)

Local:
  python solution.py --query "вопрос"
  python solution.py --input data/demo_questions.csv --output submission.csv

predict(question, history=None) -> dict  (eval contract)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

from agent.llm import load_dotenv
from agent.pipeline import run
from agent.store import cold_start

_DEFAULT_OUTPUT = "submission.csv"
_MAX_ANSWER_CHARS = 10_000
_FALLBACK_ANSWER = "В базе знаний нет ответа по этому вопросу."

# Role markers only at line start (avoids mid-text false splits).
_TURN_RE = re.compile(
    r"(?m)^(?P<role>Пользователь|Ассистент|User|Assistant)\s*:\s*",
    re.IGNORECASE,
)


def _parse_dialog(raw: str) -> tuple[str, list[dict]]:
    """If question embeds a dialog, return (last_user_utterance, prior history)."""
    text = (raw or "").strip()
    if not text:
        return "", []
    parts = list(_TURN_RE.finditer(text))
    if len(parts) < 2:
        return text, []

    turns: list[dict] = []
    for i, m in enumerate(parts):
        start = m.end()
        end = parts[i + 1].start() if i + 1 < len(parts) else len(text)
        content = text[start:end].strip()
        role_raw = m.group("role").lower()
        role = "user" if role_raw.startswith(("пользователь", "user")) else "assistant"
        if content:
            turns.append({"role": role, "content": content})
    if not turns:
        return text, []

    last_user_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return text, []
    question = turns[last_user_idx]["content"]
    history = turns[:last_user_idx]
    return question, history


def predict(question: str, history: list[dict] | None = None) -> dict:
    load_dotenv()
    return run(question, history)


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR") or "data")


def _resolve_default_paths() -> tuple[Path | None, Path]:
    """Platform I/O: questions.csv -> submission.csv."""
    out = Path(_DEFAULT_OUTPUT)
    cwd_q = Path("questions.csv")
    if cwd_q.is_file():
        return cwd_q, out
    data_q = _data_dir() / "questions.csv"
    if data_q.is_file():
        return data_q, out
    return None, out


def _clip_answer(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= _MAX_ANSWER_CHARS:
        return text
    return text[: _MAX_ANSWER_CHARS - 1].rstrip() + "…"


def _run_csv(input_path: Path, output_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    load_dotenv()
    cold_start()
    from agent.llm import probe_connectivity

    # Informational only — does not latch skip_llm for the process.
    probe_connectivity()
    with input_path.open(encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        seen: set[str] = set()
        jobs: list[tuple[int, str, str, list[dict]]] = []
        for row in reader:
            rid = str(row.get("id") or row.get("Id") or "").strip()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            raw_q = row.get("question") or row.get("Question") or ""
            question, history = _parse_dialog(raw_q)
            jobs.append((len(jobs), rid, question, history))

    workers = max(1, int(os.environ.get("MAX_WORKERS", "2")))

    def _one(job: tuple[int, str, str, list[dict]]) -> tuple[int, str, str]:
        idx, rid, question, history = job
        try:
            out = predict(question, history=history)
            answer = _clip_answer(out.get("answer") or "") or _FALLBACK_ANSWER
        except Exception:  # noqa: BLE001 — every row must produce a line
            answer = _FALLBACK_ANSWER
        return idx, rid, answer

    results: list[tuple[int, str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception:  # noqa: BLE001
                # Should not happen (_one catches), but keep row count intact.
                pass

    # Guarantee a row for every job even if a future somehow vanished.
    by_idx = {idx: (rid, ans) for idx, rid, ans in results}
    ordered: list[tuple[str, str]] = []
    for idx, rid, _q, _h in ((j[0], j[1], j[2], j[3]) for j in jobs):
        if idx in by_idx:
            ordered.append(by_idx[idx])
        else:
            ordered.append((rid, _FALLBACK_ANSWER))

    with output_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["id", "answer"])
        writer.writeheader()
        for rid, answer in ordered:
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
    if args.query:
        question, history = _parse_dialog(args.query)
        _print_one(predict(question, history=history or None))
        return

    def_in, def_out = _resolve_default_paths()
    if def_in is None:
        p.error(
            "platform mode expects questions.csv in cwd "
            f"or {_data_dir() / 'questions.csv'} (DATA_DIR)"
        )
    _run_csv(def_in, def_out)


if __name__ == "__main__":
    main()
