"""Full knowledge-base prompt for the consultant writer."""

from __future__ import annotations

import re

from agent.schemas import ScoredChunk, SourceBlock
from agent.store import Store

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")

# Product files first so catalog/recommendation questions see them immediately.
_FILE_ORDER = (
    "20-portfolio.md",
    "21-orem.md",
    "22-askue.md",
    "23-solar.md",
    "24-losses.md",
    "25-meter-reading.md",
    "26-general.md",
    "00-product-map.md",
    "02-product-scope-and-limits.md",
    "01-numbers-and-indicators.md",
    "10-market.md",
    "11-price-categories.md",
    "12-power-thresholds.md",
    "13-metering.md",
    "14-tariffs.md",
    "15-microgeneration.md",
    "16-regulation.md",
    "17-faq.md",
    "03-calculation-and-requests.md",
)


def _file_sort_key(name: str) -> tuple[int, str]:
    try:
        return (_FILE_ORDER.index(name), name)
    except ValueError:
        return (len(_FILE_ORDER), name)


def _annotate_headings(clean_text: str, headings: list[dict[str, str]]) -> str:
    by_head = {str(h.get("heading") or "").strip(): str(h.get("chunk_id") or "") for h in headings}
    lines: list[str] = []
    for line in (clean_text or "").splitlines():
        m = _HEADING_RE.match(line)
        if m:
            title = m.group(2).strip()
            cid = by_head.get(title)
            if cid:
                line = f"{line} {{id={cid}}}"
        lines.append(line)
    return "\n".join(lines).strip()


def format_full_kb(store: Store) -> tuple[str, list[str]]:
    """Return (prompt text, section ids). Built once and cached on the store."""
    cards = []
    for card in store.cards:
        cards.append(f"- {card.product}: {card.what_it_is or card.summary}")
    parts: list[str] = []
    if cards:
        parts.append("PRODUCT CARDS:\n" + "\n".join(cards))

    section_ids: list[str] = []
    docs = list(store.documents.values())
    if docs:
        docs.sort(key=lambda d: _file_sort_key(d.source_file))
        for doc in docs:
            for h in doc.headings:
                cid = store.resolve_id(str(h.get("chunk_id") or ""))
                if cid:
                    section_ids.append(cid)
            prod = f" product={doc.product}" if doc.product else ""
            body = _annotate_headings(doc.clean_text, doc.headings)
            parts.append(
                f"===== {doc.source_file} | {doc.title}{prod} =====\n{body}"
            )
    else:
        # Fallback: concatenate chunks grouped by file.
        by_file: dict[str, list] = {}
        for cid in store.chunk_ids:
            ch = store.chunks[cid]
            by_file.setdefault(ch.source_file, []).append(ch)
        for sf in sorted(by_file, key=_file_sort_key):
            blocks = []
            for ch in by_file[sf]:
                section_ids.append(ch.chunk_id)
                text = (ch.text_raw or "").strip()
                if ch.chunk_type == "table" and ch.caption:
                    text = f"{ch.caption}\n{text}"
                blocks.append(f"{{id={ch.chunk_id}}} {ch.heading}\n{text}")
            parts.append(f"===== {sf} =====\n" + "\n\n".join(blocks))

    text = "\n\n".join(p for p in parts if p).strip()
    return text, list(dict.fromkeys(section_ids))


def all_chunks_as_scored(store: Store) -> list[ScoredChunk]:
    out: list[ScoredChunk] = []
    for cid in store.chunk_ids:
        ch = store.chunks.get(cid)
        if not ch:
            continue
        text = ch.text_raw or ""
        if ch.chunk_type == "table" and ch.caption:
            text = f"{ch.caption}\n{text}"
        out.append(
            ScoredChunk(
                chunk_id=ch.chunk_id,
                score=1.0,
                text=text,
                source_file=ch.source_file,
                heading=ch.heading,
                n_tokens=ch.n_tokens,
                products=list(ch.products),
            )
        )
    return out


def section_blocks(store: Store) -> list[SourceBlock]:
    blocks: list[SourceBlock] = []
    for cid in getattr(store, "kb_section_ids", None) or []:
        ch = store.get_chunk(cid)
        if not ch:
            continue
        blocks.append(
            SourceBlock(
                chunk_id=ch.chunk_id,
                heading=ch.heading,
                text=ch.text_raw or "",
                source_file=ch.source_file,
                products=list(ch.products),
            )
        )
    return blocks
