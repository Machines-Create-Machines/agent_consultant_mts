"""Consultant metrics: retrieval, citation, catalog coverage, relevance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"\S+")


def mrr_at_k(ranked_ids: list[str], gold_any: list[str], k: int = 5) -> float:
    gold = set(gold_any or [])
    if not gold:
        return 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        if any(g in cid or cid in g for g in gold):
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked_ids: list[str], gold_any: list[str], k: int = 5) -> float:
    gold = set(gold_any or [])
    if not gold or k <= 0:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if any(g in cid or cid in g for g in gold))
    return hits / float(k)


def citation_precision(cites: list[str], gold_any: list[str]) -> float:
    if not cites:
        return 0.0 if gold_any else 1.0
    gold = set(gold_any or [])
    if not gold:
        return 1.0
    hits = sum(1 for c in cites if any(g in c or c in g for g in gold))
    return hits / float(len(cites))


def catalog_coverage(answer: str, products: list[str], required: list[str] | None = None) -> float:
    """Fraction of required products mentioned in answer or products list."""
    req = required or []
    if not req:
        return 1.0
    blob = ((answer or "") + " " + " ".join(products or [])).lower().replace("ё", "е")
    hits = 0
    for p in req:
        plow = p.lower().replace("ё", "е")
        short = plow.split()[0] if plow else ""
        if plow in blob or (short and short in blob):
            hits += 1
            continue
        # loose aliases
        aliases = {
            "выход на оптовый рынок": ("оптов", "орэм"),
            "облачная платформа учёта": ("платформ", "аскуэ", "учёт", "учет"),
            "солнечные электростанции": ("солнеч",),
            "поиск коммерческих потерь": ("потер", "хищен"),
            "распознавание показаний по фотографии": ("фото", "распозна"),
        }
        if any(a in blob for a in aliases.get(plow, ())):
            hits += 1
    return hits / float(len(req))


def context_relevance(answer: str, forbid: list[str] | None = None) -> float:
    """1.0 if answer avoids known irrelevant fragments, else 0.0."""
    low = (answer or "").lower().replace("ё", "е")
    for bad in forbid or []:
        if bad.lower().replace("ё", "е") in low:
            return 0.0
    # Heuristic: lone power-threshold answers are low relevance for overview.
    if "присоединённая мощность" in low or "присоединенная мощность" in low:
        if not any(x in low for x in ("солнеч", "фото", "потер", "аскуэ", "платформ")):
            return 0.0
    return 1.0


def long_copy_overlap(answer: str, source: str, n: int = 12) -> float:
    """Fraction of answer n-grams that appear verbatim in source."""
    aw = [w.lower().replace("ё", "е") for w in _WORD_RE.findall(answer or "")]
    sw = [w.lower().replace("ё", "е") for w in _WORD_RE.findall(source or "")]
    if len(aw) < n or len(sw) < n:
        return 0.0
    src = {" ".join(sw[i : i + n]) for i in range(len(sw) - n + 1)}
    hits = 0
    total = 0
    for i in range(len(aw) - n + 1):
        total += 1
        if " ".join(aw[i : i + n]) in src:
            hits += 1
    return hits / float(total) if total else 0.0


def directness_naturalness(answer: str) -> float:
    """Simple proxy: starts with substance, not table/heading dump."""
    text = (answer or "").strip()
    if not text:
        return 0.0
    low = text.lower().replace("ё", "е")
    score = 1.0
    if low.startswith("таблица в разделе") or low.startswith("основной вопрос"):
        score -= 0.6
    if low.startswith("документ:") or low.startswith("раздел:") or low.startswith("назначение:"):
        score -= 0.6
    if "|" in text[:80] or text.startswith("###"):
        score -= 0.4
    words = _WORD_RE.findall(text)
    if len(words) < 12:
        score -= 0.2
    if len(words) > 180:
        score -= 0.2
    return max(0.0, min(1.0, score))


def summarize_scenario_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics from scenario runner outputs."""
    mrrs: list[float] = []
    precs: list[float] = []
    cite_ps: list[float] = []
    covs: list[float] = []
    relev: list[float] = []
    direct: list[float] = []
    unsupported_wrong = 0
    unsupported_total = 0
    for row in rows:
        gold = row.get("gold_chunks_any") or []
        retrieved = row.get("retrieved_ids") or []
        cites = row.get("citations") or []
        answer = row.get("answer") or ""
        products = row.get("products") or []
        if gold:
            mrrs.append(mrr_at_k(retrieved, gold, 5))
            precs.append(precision_at_k(retrieved, gold, 5))
            cite_ps.append(citation_precision(cites, gold))
        req = row.get("must_cover_products") or []
        if req or row.get("kind") == "catalog_overview":
            covs.append(catalog_coverage(answer, products, req or None))
            relev.append(context_relevance(answer, row.get("forbid") or []))
        direct.append(directness_naturalness(answer))
        expect = set(row.get("expect_behavior") or [])
        beh = row.get("behavior") or ""
        if "refuse" in expect or "off_topic" in expect:
            unsupported_total += 1
            if beh not in ("refuse", "off_topic"):
                unsupported_wrong += 1
    return {
        "mrr@5": sum(mrrs) / len(mrrs) if mrrs else None,
        "precision@5": sum(precs) / len(precs) if precs else None,
        "citation_precision": sum(cite_ps) / len(cite_ps) if cite_ps else None,
        "catalog_coverage": sum(covs) / len(covs) if covs else None,
        "context_relevance": sum(relev) / len(relev) if relev else None,
        "directness_naturalness": sum(direct) / len(direct) if direct else None,
        "unsupported_answer_rate": (
            unsupported_wrong / unsupported_total if unsupported_total else 0.0
        ),
        "n": len(rows),
    }


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
