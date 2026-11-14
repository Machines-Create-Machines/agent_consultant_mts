"""Product alias glossary for product precision/recall."""

from __future__ import annotations

import re
from typing import Iterable

# Canonical names match expected_products in the dataset.
PRODUCT_ALIASES: dict[str, list[str]] = {
    "Солнечные электростанции": [
        "солнечные электростанции",
        "солнечная электростанция",
        "солнечные станции",
        "солнечная станция",
        "сетевая солнечная",
        "гибридная солнечная",
        "солнечные панели",
        "сэс",
        "solar",
    ],
    "Облачная платформа учёта": [
        "облачная платформа учёта",
        "облачная платформа учета",
        "платформа учёта",
        "платформа учета",
        "аскуэ",
        "askue",
        "удалённый сбор",
        "удаленный сбор",
        "система учёта",
        "система учета",
    ],
    "Выход на оптовый рынок": [
        "выход на оптовый рынок",
        "оптовый рынок",
        "орэм",
        "orem",
        "сопровождение выхода",
    ],
    "Распознавание показаний по фотографии": [
        "распознавание показаний",
        "показания по фото",
        "по фотографии",
        "фото сервис",
        "сервис по фото",
        "meter reading",
    ],
    "Поиск коммерческих потерь": [
        "поиск коммерческих потерь",
        "коммерческих потерь",
        "коммерческие потери",
        "поиск потерь",
        "хищен",
        "losses",
    ],
}


def _norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s/+.-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def extract_products(text: str) -> set[str]:
    """Map free text to canonical product names via aliases."""
    if not text:
        return set()
    norm = _norm(text)
    found: set[str] = set()
    for canonical, aliases in PRODUCT_ALIASES.items():
        variants = [_norm(canonical)] + [_norm(a) for a in aliases]
        if any(v and v in norm for v in variants):
            found.add(canonical)
    return found


def product_prf(
    expected: Iterable[str], predicted: Iterable[str]
) -> tuple[float, float, float]:
    """Precision / recall / F1. Empty gold → precision=1 if answer empty of products."""
    gold = set(expected or [])
    pred = set(predicted or [])
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold and pred:
        return 0.0, 1.0, 0.0
    if gold and not pred:
        return 1.0, 0.0, 0.0
    tp = len(gold & pred)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
