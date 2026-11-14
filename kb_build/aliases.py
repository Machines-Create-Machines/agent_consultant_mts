"""Gold chunk_id aliases: dataset id -> canonical chunker id."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

# Dataset gold ids that differ from markdown anchors (punctuation / sub-anchors).
GOLD_TO_CANONICAL: dict[str, str] = {
    "11-price-categories.md#Зачем категорий несколько а не одна цена": (
        "11-price-categories.md#Зачем категорий несколько, а не одна цена"
    ),
    "11-price-categories.md#Какая категория действует если её не выбирали": (
        "11-price-categories.md#Какая категория действует, если её не выбирали"
    ),
    "11-price-categories.md#Что нужно чтобы сменить категорию": (
        "11-price-categories.md#Что нужно, чтобы сменить категорию"
    ),
    "17-faq.md#Нужно ли менять учёт если мы ничего не меняем": (
        "17-faq.md#Нужно ли менять учёт, если мы ничего не меняем"
    ),
    "21-orem.md#Не подходит если": "21-orem.md#Не подходит, если",
    "21-orem.md#Подходит если": "21-orem.md#Подходит, если",
    "21-orem.md#Что если расчёт покажет что нам это невыгодно": (
        "21-orem.md#Что если расчёт покажет, что нам это невыгодно"
    ),
    "22-askue.md#Пилотное подключение": "22-askue.md#Пилотное подключение",
    "22-askue.md#Требования законодательства которые закрывает платформа": (
        "22-askue.md#Требования законодательства, которые закрывает платформа"
    ),
    "23-solar.md#Не подходит": "23-solar.md#Не подходит, если",
    "23-solar.md#Окупаемость: две разные рамки": (
        "23-solar.md#Окупаемость: две разные рамки, и называть их надо вместе"
    ),
    "24-losses.md#Как работает решение": (
        "24-losses.md#Как работает решение — общий принцип"
    ),
    "24-losses.md#Требования к данным": "24-losses.md#Требования к данным",
    "24-losses.md#Что нужно от клиента": "24-losses.md#Что нужно от клиента",
}


def nfc_chunk_id(chunk_id: str) -> str:
    return unicodedata.normalize("NFC", chunk_id)


def load_required_gold_ids(dataset_path: Path) -> set[str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for case in data:
        for cid in case.get("gold_chunks", []):
            out.add(nfc_chunk_id(cid))
    return out


def build_id_aliases(required_gold: set[str], produced_canonical: set[str]) -> dict[str, str]:
    """Map gold chunk_id -> canonical chunk_id when they differ."""
    aliases: dict[str, str] = {}
    for gold in sorted(required_gold):
        canonical = GOLD_TO_CANONICAL.get(gold, gold)
        canonical = nfc_chunk_id(canonical)
        if canonical != gold:
            if canonical in produced_canonical:
                aliases[gold] = canonical
        elif gold not in produced_canonical:
            # gold might still need alias if only canonical variant exists under another key
            pass
    # Ensure explicit entries from GOLD_TO_CANONICAL when canonical is produced
    for gold, canonical in GOLD_TO_CANONICAL.items():
        gold = nfc_chunk_id(gold)
        canonical = nfc_chunk_id(canonical)
        if gold in required_gold and canonical in produced_canonical and gold != canonical:
            aliases[gold] = canonical
    return aliases


def resolve_id(chunk_id: str, aliases: dict[str, str]) -> str:
    """Normalize id for set comparisons (gold id -> canonical if aliased)."""
    chunk_id = nfc_chunk_id(chunk_id)
    return aliases.get(chunk_id, chunk_id)