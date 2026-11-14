"""Deterministic (no-LLM) metric checkers."""

from __future__ import annotations

import re
from typing import Any

from eval.kb import Chunk, normalize_heading, texts_for_ids
from eval.loader import Case
from eval.products import extract_products, product_prf


def _norm_chunk_id(cid: str) -> str:
    if "#" not in cid:
        return cid
    file_name, heading = cid.split("#", 1)
    return f"{file_name}#{normalize_heading(heading)}"


def _id_in(needle: str, haystack: list[str]) -> bool:
    n = _norm_chunk_id(needle)
    return any(_norm_chunk_id(h) == n for h in haystack)

_NUMBER_RE = re.compile(
    r"(?<![\w./])(\d+(?:[.,]\d+)?(?:\s*[-–—]\s*\d+(?:[.,]\d+)?)?)\s*"
    r"(квт·?ч|квт\.?ч|квт|мвт|руб\.?|₽|%|лет|года|год|мес(?:яц(?:ев|а)?)?|"
    r"дн(?:ей|я)?|час(?:ов|а)?|мин(?:ут)?|сек)?",
    re.IGNORECASE,
)
_UNIT_WORDS = {
    "квт",
    "квтч",
    "квт·ч",
    "мвт",
    "руб",
    "₽",
    "%",
    "лет",
    "года",
    "год",
}


def _norm(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def recall_at_k(gold_chunks: list[str], retrieved_ids: list[str], k: int) -> float:
    if not gold_chunks:
        return 1.0
    top = retrieved_ids[:k]
    hits = sum(1 for g in gold_chunks if _id_in(g, top))
    return hits / len(gold_chunks)


def hit_at_1(gold_chunks: list[str], retrieved_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    if not gold_chunks:
        return 1.0
    return 1.0 if _id_in(retrieved_ids[0], gold_chunks) else 0.0


def behavior_ok(predicted: str | None, allowed: list[str]) -> bool:
    if not predicted:
        return False
    return predicted in allowed


def forbidden_hits(answer: str, forbidden: list[str]) -> list[str]:
    ans = _norm(answer)
    hits: list[str] = []
    for item in forbidden or []:
        needle = _norm(item)
        if needle and needle in ans:
            hits.append(item)
    return hits


def groundedness_score(answer: str, citation_ids: list[str], chunks: dict[str, Chunk]) -> float:
    """Share of answer numbers/units/products that appear in cited chunk text."""
    if not answer.strip():
        return 0.0
    cited = _norm(texts_for_ids(citation_ids, chunks))
    if not cited:
        # no citations → nothing grounded
        claims = _extract_claims(answer)
        return 1.0 if not claims else 0.0

    claims = _extract_claims(answer)
    if not claims:
        return 1.0
    ok = 0
    for claim in claims:
        if claim in cited:
            ok += 1
    return ok / len(claims)


def _extract_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for m in _NUMBER_RE.finditer(answer):
        num = m.group(1).replace(",", ".")
        unit = (m.group(2) or "").lower().replace(" ", "")
        token = _norm(f"{num} {unit}".strip())
        if token:
            claims.append(token)
        if unit:
            claims.append(_norm(unit))
    for p in extract_products(answer):
        claims.append(_norm(p))
    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for c in claims:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def score_case(
    case: Case,
    answer_rec: dict[str, Any],
    chunks: dict[str, Chunk],
    k: int,
    max_words: int,
) -> dict[str, Any]:
    raw = case.raw
    answer = answer_rec.get("answer") or ""
    error = answer_rec.get("error")
    retrieved = list(answer_rec.get("retrieved_ids") or [])
    citations = list(answer_rec.get("citations") or retrieved[:k])
    predicted_behavior = answer_rec.get("behavior")
    predicted_products = answer_rec.get("products")
    if predicted_products is None:
        predicted_products = sorted(extract_products(answer))
    else:
        predicted_products = list(predicted_products)

    is_error = bool(error) or not str(answer).strip()
    viol = forbidden_hits(answer, raw.get("forbidden") or []) if not is_error else []
    p, r, f1 = product_prf(raw.get("expected_products") or [], predicted_products)

    metrics = {
        "id": case.id,
        "split": case.split,
        "tags": "|".join(raw.get("tags") or []),
        "source_file": _source_file(raw.get("gold_chunks") or []),
        "gold_behavior": raw.get("behavior"),
        "predicted_behavior": predicted_behavior or "",
        "behavior_ok": False if is_error else behavior_ok(predicted_behavior, raw.get("allowed_behaviors") or []),
        "recall_at_k": 0.0 if is_error else recall_at_k(raw.get("gold_chunks") or [], retrieved, k),
        "hit_at_1": 0.0 if is_error else hit_at_1(raw.get("gold_chunks") or [], retrieved),
        "product_precision": 0.0 if is_error else p,
        "product_recall": 0.0 if is_error else r,
        "product_f1": 0.0 if is_error else f1,
        "groundedness": 0.0 if is_error else groundedness_score(answer, citations, chunks),
        "violation": bool(viol),
        "violation_items": "|".join(viol),
        "length_ok": False if is_error else word_count(answer) <= max_words,
        "word_count": word_count(answer),
        "error": is_error,
        "error_type": (error or ("empty_answer" if not str(answer).strip() else "")),
        "latency_ms": answer_rec.get("latency_ms"),
        "cold_start": bool(answer_rec.get("cold_start")),
        "llm_calls": int(answer_rec.get("llm_calls") or 0),
        "n_retrieved": len(retrieved),
        "n_citations": len(citations),
        "n_nuggets": len(raw.get("nuggets") or []),
        "nugget_weight_sum": sum(float(n.get("w", 1)) for n in (raw.get("nuggets") or [])),
        # LLM-filled later
        "nugget_recall": None,
        "nugget_hits": None,
        "judge_vs_gold": None,
        "judge_vs_baseline": None,
        "judge_win_score": None,
        "soft_violation": None,
    }
    return metrics


def _source_file(gold_chunks: list[str]) -> str:
    files = []
    for g in gold_chunks:
        files.append(g.split("#", 1)[0] if "#" in g else g)
    return files[0] if files else ""
