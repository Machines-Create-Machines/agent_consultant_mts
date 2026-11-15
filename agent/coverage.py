"""Coverage gates from coverage_clarity_explaning.md."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent import config
from agent.schemas import Aspect, ScoredChunk, UnderstandResult

ATTR_TERMS = {
    "цена": ["цена", "стоимость", "тариф", "руб", "₽", "прайс", "оплата"],
    "срок": ["срок", "дней", "недель", "месяц", "внедрение", "подключ"],
    "применимость": ["подходит", "для", "применяется", "рассчитан", "сегмент"],
    "механизм": ["как", "за счёт", "позволяет", "выявля", "снижа", "переда"],
    "наличие": ["есть", "поддерживает", "возможность", "функция"],
    "гарантия": ["гарантия", "срок службы", "обслуживание"],
    "сравнение": ["сравн", "отлича", "versus", "или"],
    "условия": ["если", "требуется", "услови", "при"],
}

_NUM = re.compile(
    r"\d[\d\s.,]*\s*(?:руб|₽|дн|мес|кВт|А|шт|%|млн|тыс)",
    re.I,
)


def _lemma_overlap(keywords: list[str], text: str) -> bool:
    low = (text or "").lower().replace("ё", "е")
    for kw in keywords:
        k = (kw or "").lower().replace("ё", "е").strip()
        if len(k) >= 2 and k in low:
            return True
    return False


def number_near_attr(candidates: list[ScoredChunk], attr_terms: list[str], window: int = 120) -> bool:
    for c in candidates:
        text = c.text or ""
        low = text.lower().replace("ё", "е")
        for term in attr_terms:
            t = term.lower().replace("ё", "е")
            pos = low.find(t)
            while pos >= 0:
                lo = max(0, pos - window)
                hi = min(len(text), pos + len(t) + window)
                if _NUM.search(text[lo:hi]):
                    return True
                pos = low.find(t, pos + 1)
    return False


def aspect_strength(a: Aspect, evidence: list[ScoredChunk], T_chunk: float) -> float:
    cand = [c for c in evidence if c.score >= T_chunk]
    if not cand:
        # soften: use top evidence if scores not calibrated
        cand = evidence[:3]
    topic = any(_lemma_overlap(a.keywords, c.text) for c in cand)
    attr_terms = ATTR_TERMS.get(a.attr, [a.attr])
    attr = any(_lemma_overlap(attr_terms, c.text) for c in cand)
    s = 1.0 if (topic and attr) else 0.5 if topic else 0.0
    if a.attr in ("цена", "срок", "гарантия") and s > 0:
        if not number_near_attr(cand, attr_terms, window=120):
            s = min(s, 0.5)
    return s


@dataclass
class CoverageResult:
    level: str  # high | partial | none
    raw: float
    top1: float
    missing_aspects: list[str]
    strengths: dict[str, float]


def compute_coverage(
    understand: UnderstandResult,
    evidence: list[ScoredChunk],
    *,
    used_rerank: bool = False,
    catalog: list[str] | None = None,
    linked_entities: bool = True,
) -> CoverageResult:
    thr = config.THRESHOLDS_RERANK if used_rerank else config.THRESHOLDS_NO_RERANK
    # normalize scores to [0,1] via min-max if not reranked
    scores = [c.score for c in evidence]
    if scores:
        lo, hi = min(scores), max(scores)
        if hi - lo > 1e-12 and not used_rerank:
            normed = [
                ScoredChunk(
                    chunk_id=c.chunk_id,
                    score=(c.score - lo) / (hi - lo),
                    text=c.text,
                    source_file=c.source_file,
                    heading=c.heading,
                    n_tokens=c.n_tokens,
                    products=c.products,
                    branches=c.branches,
                )
                for c in evidence
            ]
        else:
            # already ~[0,1] fusion or rerank
            normed = evidence
            if not used_rerank:
                # fusion scores already roughly [0,1]
                pass
    else:
        normed = []

    top1 = normed[0].score if normed else 0.0
    top3 = normed[2].score if len(normed) > 2 else (normed[-1].score if normed else 0.0)

    aspects = list(understand.aspects) or [
        Aspect(name="q", attr="механизм", keywords=[understand.standalone_query], weight=1)
    ]

    strengths: dict[str, float] = {}
    for a in aspects:
        strengths[a.name] = aspect_strength(a, normed, thr.T_chunk)

    wsum = sum(max(a.weight, 0.1) for a in aspects) or 1.0
    coverage_raw = sum(max(a.weight, 0.1) * strengths[a.name] for a in aspects) / wsum

    missing = [a.name for a in aspects if strengths[a.name] < 0.5]

    # named product not in catalog
    catalog = catalog or []
    for ent in understand.entities:
        # if looks like product mention but unknown — only when strongly product-shaped
        pass

    if top1 < thr.T_floor:
        return CoverageResult("none", coverage_raw, top1, missing or ["вопрос"], strengths)
    if not linked_entities and top1 < thr.T_mid:
        return CoverageResult("none", coverage_raw, top1, missing or ["вопрос"], strengths)

    if coverage_raw >= 0.80 and top1 >= thr.T_high:
        level = "high"
    elif coverage_raw >= 0.35:
        level = "partial"
    else:
        level = "none"

    if level == "high" and (top1 - top3) < thr.delta:
        level = "partial"

    # Soft floor for BM25-only: if we have decent top1 fusion score, avoid NONE
    if level == "none" and top1 >= thr.T_chunk * 0.9 and coverage_raw >= 0.25:
        level = "partial"

    return CoverageResult(level, coverage_raw, top1, missing, strengths)
