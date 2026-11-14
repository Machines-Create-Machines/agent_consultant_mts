"""Enrich raw chunks into full schema records."""

from __future__ import annotations

import re
from typing import Any

from kb_build.chunker import RawChunk, count_tokens, nfc

CANONICAL_PRODUCTS = frozenset(
    {
        "Солнечные электростанции",
        "Облачная платформа учёта",
        "Распознавание показаний по фотографии",
        "Выход на оптовый рынок",
        "Поиск коммерческих потерь",
    }
)

FILE_PRODUCT: dict[str, str | None] = {
    "21-orem.md": "Выход на оптовый рынок",
    "22-askue.md": "Облачная платформа учёта",
    "23-solar.md": "Солнечные электростанции",
    "24-losses.md": "Поиск коммерческих потерь",
    "25-meter-reading.md": "Распознавание показаний по фотографии",
}

REF_SECTION_PRODUCT: dict[tuple[str, str], str] = {
    ("01-numbers-and-indicators.md", "Выход на оптовый рынок"): "Выход на оптовый рынок",
    ("01-numbers-and-indicators.md", "Платформа учёта"): "Облачная платформа учёта",
    ("01-numbers-and-indicators.md", "Солнечные станции"): "Солнечные электростанции",
    ("01-numbers-and-indicators.md", "Неуниверсальные показатели"): "Солнечные электростанции",
    ("03-calculation-and-requests.md", "Бесплатная предварительная оценка"): "Солнечные электростанции",
    ("03-calculation-and-requests.md", "Для платформы учёта"): "Облачная платформа учёта",
}

CONSTRAINT_HEADINGS = frozenset(
    {
        "Чего утверждать нельзя",
        "Что мы не делаем",
        "Не подходит, если",
        "Не подходит",
        "Не подходит энергосбытовым компаниям",
        "Не подходит промышленным потребителям",
    }
)

SEGMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "СМБ": ("малый", "мелк", "смб", "среднему бизнесу"),
    "предприятие": ("предприяти", "промышлен", "производств"),
    "СНТ": ("снт", "товариществ", "садовод", "дач"),
    "сетевая": ("сетев", "сетевые компании", "электросет"),
    "МТС": ("мтс",),
}


def _infer_products(chunk: RawChunk) -> list[str]:
    found: set[str] = set()
    key = REF_SECTION_PRODUCT.get((chunk.source_file, chunk.heading))
    if key:
        found.add(key)
    fp = FILE_PRODUCT.get(chunk.source_file)
    if fp:
        found.add(fp)
    text_l = chunk.text_raw.lower()
    if "оптов" in text_l and chunk.source_file in ("10-market.md", "12-power-thresholds.md"):
        if "21-orem" in text_l or "оптовый рынок" in chunk.heading.lower():
            found.add("Выход на оптовый рынок")
    for name in CANONICAL_PRODUCTS:
        if name.lower()[:12] in text_l:
            found.add(name)
    return sorted(found & CANONICAL_PRODUCTS)


def _infer_segments(text: str) -> list[str]:
    tl = text.lower()
    out: list[str] = []
    for seg, keys in SEGMENT_KEYWORDS.items():
        if any(k in tl for k in keys):
            out.append(seg)
    return out or ["предприятие"]


def _infer_topics(heading: str, text: str) -> list[str]:
    topics: list[str] = []
    hl = heading.lower()
    mapping = [
        ("учёт", ("учёт", "учет", "аскуэ", "прибор")),
        ("оптовый рынок", ("оптов", "орэм")),
        ("солнеч", ("солнеч", "панел", "инсоля")),
        ("тариф", ("тариф", "категор")),
        ("потери", ("потер", "хищен")),
        ("мощность", ("квт", "мвт", "мощност")),
    ]
    blob = hl + " " + text.lower()
    for label, keys in mapping:
        if any(k in blob for k in keys):
            topics.append(label)
    return topics[:6] or ["общее"]


def _related_ids(chunk: RawChunk, by_file_heading: dict[tuple[str, str], str]) -> list[str]:
    h = chunk.heading.split("--")[0]
    if h not in CONSTRAINT_HEADINGS and "утверждать нельзя" not in h:
        return []
    sf = chunk.source_file
    candidates = [
        ("Что это", f"{sf}#Что это"),
        ("Кому подходит", f"{sf}#Кому подходит"),
        ("Подходит, если", f"{sf}#Подходит, если"),
        ("Какой результат даёт продукт", f"{sf}#Какой результат даёт продукт"),
    ]
    out: list[str] = []
    for _, cid in candidates:
        fn, hd = cid.split("#", 1)
        if (fn, hd) in by_file_heading:
            out.append(nfc(cid))
    if h in ("Не подходит, если", "Не подходит") and (sf, "Кому подходит") in by_file_heading:
        out.insert(0, nfc(f"{sf}#Кому подходит"))
    return out[:4]


def _hyp_questions(heading: str) -> list[str]:
    h = heading.split("--")[0]
    qs = [
        f"расскажите про {h.lower()}",
        f"что важно знать про {h.lower()}",
    ]
    if "?" not in h and len(h) < 60:
        qs.append(f"как у вас с «{h.lower()}»")
    return qs[:3]


def _table_caption(text: str, heading: str) -> str | None:
    if "  " not in text and "|" not in text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
    if len(lines) < 2:
        return None
    return (
        f"Таблица в разделе «{heading}»: сводные показатели и пояснения. "
        f"Используйте числа из таблицы только вместе с оговорками из того же раздела."
    )


def _text_indexed(chunk: RawChunk) -> str:
    path = " > ".join(chunk.heading_path)
    intro = f"Документ: {chunk.source_file}. Раздел: {path}."
    return f"{intro}\n\n{chunk.text_raw}".strip()


def finalize_chunks(raw: list[RawChunk]) -> list[dict[str, Any]]:
    by_fh: dict[tuple[str, str], str] = {}
    for c in raw:
        if c.part == 1 and c.is_parent_full:
            by_fh[(c.source_file, c.heading.split("--")[0])] = c.chunk_id

    records: list[dict[str, Any]] = []
    for c in raw:
        caption = _table_caption(c.text_raw, c.heading) if c.chunk_type == "table" else None
        if c.chunk_type == "table" and not caption:
            caption = _table_caption(c.text_raw, c.heading) or (
                f"Табличные данные раздела «{c.heading}»."
            )
        rec: dict[str, Any] = {
            "chunk_id": nfc(c.chunk_id),
            "parent_chunk_id": nfc(c.parent_chunk_id),
            "source_file": c.source_file,
            "source_sha": c.source_sha,
            "heading": c.heading,
            "heading_path": c.heading_path,
            "part": c.part,
            "n_parts": c.n_parts,
            "char_span": c.char_span,
            "chunk_type": c.chunk_type,
            "text_raw": c.text_raw,
            "text_indexed": _text_indexed(c),
            "caption": caption,
            "hyp_questions": _hyp_questions(c.heading),
            "products": _infer_products(c),
            "segments": _infer_segments(c.text_raw),
            "topics": _infer_topics(c.heading, c.text_raw),
            "related_ids": _related_ids(c, by_fh),
            "n_tokens": count_tokens(c.text_raw),
        }
        records.append(rec)
    return records