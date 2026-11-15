"""Canonical product aliases for planner / retrieval (platform-safe, no eval import)."""

from __future__ import annotations

import re

PRODUCT_ALIASES: dict[str, list[str]] = {
    "Солнечные электростанции": [
        "солнечные электростанции",
        "солнечная электростанция",
        "солнечные станции",
        "солнечная станция",
        "солнечные панели",
        "сэс",
        "solar",
    ],
    "Облачная платформа учёта": [
        "облачная платформа учёта",
        "облачная платформа учета",
        "облачную платформу учёта",
        "облачную платформу учета",
        "платформа учёта",
        "платформа учета",
        "платформу учёта",
        "платформу учета",
        "аскуэ",
        "askue",
        "система учёта",
        "система учета",
    ],
    "Выход на оптовый рынок": [
        "выход на оптовый рынок",
        "оптовый рынок",
        "орэм",
        "orem",
    ],
    "Распознавание показаний по фотографии": [
        "распознавание показаний",
        "показания по фото",
        "по фотографии",
        "фото сервис",
        "многотарифный счётчик",
        "двухтарифный счётчик",
        "тарифные счётчики",
        "meter reading",
    ],
    "Поиск коммерческих потерь": [
        "поиск коммерческих потерь",
        "коммерческих потерь",
        "коммерческие потери",
        "поиск потерь",
        "хищен",
    ],
}


def _norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s/+.-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _stem_tokens(text: str) -> list[str]:
    """Rough RU stems: keep first 5+ chars to tolerate case endings."""
    out: list[str] = []
    for t in _norm(text).split():
        if len(t) <= 4:
            out.append(t)
        else:
            out.append(t[:5])
    return out


def extract_products(text: str) -> set[str]:
    if not text:
        return set()
    norm = _norm(text)
    stems = set(_stem_tokens(norm))
    found: set[str] = set()
    for canonical, aliases in PRODUCT_ALIASES.items():
        variants = [_norm(canonical)] + [_norm(a) for a in aliases]
        if any(v and v in norm for v in variants):
            found.add(canonical)
            continue
        # Morph-tolerant: require ≥2 significant stem hits from a multi-word alias.
        for v in variants:
            vtoks = [t for t in _stem_tokens(v) if len(t) >= 4]
            if len(vtoks) >= 2 and sum(1 for t in vtoks if t in stems) >= 2:
                found.add(canonical)
                break
    return found
