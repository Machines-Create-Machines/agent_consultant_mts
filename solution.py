"""
Minimal offline submit baseline (platform-compatible black box for local eval).

I/O contract (inferred from demo_questions.csv / demo_output.csv + eval needs):

  predict(question: str, history: list[dict] | None = None) -> dict | str

dict fields:
  answer, behavior, products, retrieved_ids, retrieved_scores, citations, llm_calls

Batch CSV:
  python solution.py --input demo_questions.csv --output answers.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_PRODUCT_HINTS = [
    ("Солнечные электростанции", ("солнеч", "панел", "сэс", "solar", "23-")),
    ("Облачная платформа учёта", ("платформ", "аскуэ", "учёт", "учет", "askue", "22-")),
    ("Выход на оптовый рынок", ("оптов", "орэм", "orem", "21-")),
    ("Распознавание показаний по фотографии", ("фото", "показан", "25-")),
    ("Поиск коммерческих потерь", ("потер", "хищен", "24-")),
]

_CHUNKS: list[tuple[str, str]] | None = None


def _norm_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _load_chunks() -> list[tuple[str, str]]:
    global _CHUNKS
    if _CHUNKS is not None:
        return _CHUNKS
    kb = ROOT / "knowledge-base"
    out: list[tuple[str, str]] = []
    for path in sorted(kb.rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        heading = path.stem
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf, heading
            body = "\n".join(buf).strip()
            cid = f"{path.name}#{_norm_heading(heading)}"
            out.append((cid, f"{heading}\n{body}"))
            buf = []

        for line in lines:
            m = re.match(r"^(#{1,3})\s+(.+)$", line)
            if m:
                flush()
                heading = m.group(2).strip()
            else:
                buf.append(line)
        flush()
    _CHUNKS = out
    return out


def _tokens(text: str) -> set[str]:
    text = (text or "").lower().replace("ё", "е")
    return {t for t in re.findall(r"[a-zа-я0-9]+", text, flags=re.I) if len(t) > 2}


def _score(q: set[str], text: str) -> float:
    ct = _tokens(text)
    if not q or not ct:
        return 0.0
    return len(q & ct) / (len(q) ** 0.5)


def _products(text: str) -> list[str]:
    low = text.lower().replace("ё", "е")
    found = []
    for name, keys in _PRODUCT_HINTS:
        if any(k in low for k in keys):
            found.append(name)
    return found


def predict(question: str, history: list[dict] | None = None) -> dict:
    history = history or []
    ctx = " ".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in history)
    qtoks = _tokens(f"{ctx} {question}")
    ranked: list[tuple[str, float, str]] = []
    for cid, text in _load_chunks():
        s = _score(qtoks, text)
        if s > 0:
            ranked.append((cid, s, text))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    top = ranked[:5]
    top_score = top[0][1] if top else 0.0
    behavior = "clarify" if top_score < 1.0 else "answer"

    if not top:
        answer = (
            "По этому вопросу у меня недостаточно данных в базе знаний. "
            "Могу связать вас со специалистом."
        )
    else:
        body = re.sub(r"\s+", " ", top[0][2]).strip()
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", body) if p.strip()]
        snippet = " ".join(parts[:2]) if parts else body[:400]
        if len(snippet) > 500:
            snippet = snippet[:497].rsplit(" ", 1)[0] + "…"
        if behavior == "clarify":
            answer = (
                f"{snippet} Уточните, пожалуйста, мощность объекта и основную задачу."
            )
        else:
            answer = (
                f"{snippet} Подскажите регион и параметры объекта — так оценка будет точнее."
            )

    ids = [c[0] for c in top]
    products = sorted(set(_products(answer + " " + question + " " + " ".join(ids))))
    return {
        "answer": answer,
        "behavior": behavior,
        "products": products,
        "retrieved_ids": ids,
        "retrieved_scores": [round(c[1], 4) for c in top],
        "citations": ids[:2],
        "llm_calls": 0,
        "stages": {"retrieve_ms": 0, "generate_ms": 0},
    }


def _run_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open(encoding="utf-8", newline="") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["id", "answer"])
        writer.writeheader()
        for row in reader:
            out = predict(row.get("question") or "", history=[])
            writer.writerow({"id": row.get("id"), "answer": out["answer"]})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.input and args.output:
        _run_csv(args.input, args.output)
        return
    print(predict("можно поставить панели на оба ската восток и запад")["answer"])


if __name__ == "__main__":
    main()
