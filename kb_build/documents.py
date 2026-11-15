"""Clean document manifests and product dossiers for Consultant v4."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kb_build.cards import PRODUCT_CARDS
from kb_build.chunker import HR_RE, file_sha, parse_file
from kb_build.enrich import FILE_PRODUCT

_WS = re.compile(r"[ \t]+")


def clean_markdown(text: str) -> str:
    lines: list[str] = []
    for line in (text or "").splitlines():
        if HR_RE.match(line.strip()):
            continue
        lines.append(_WS.sub(" ", line).rstrip())
    out: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank and out:
                out.append("")
            blank = True
            continue
        blank = False
        out.append(line)
    return "\n".join(out).strip()


def build_documents(kb_dir: Path, chunk_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_file_head: dict[tuple[str, str], str] = {}
    for rec in chunk_records:
        sf = rec.get("source_file") or ""
        hd = rec.get("heading") or ""
        cid = rec.get("chunk_id") or ""
        if sf and hd and cid and (sf, hd) not in by_file_head:
            by_file_head[(sf, hd)] = cid

    docs: list[dict[str, Any]] = []
    for path in sorted(kb_dir.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title, sections = parse_file(path)
        headings: list[dict[str, str]] = []
        for sec in sections:
            cid = by_file_head.get((path.name, sec.heading), f"{path.name}#{sec.heading}")
            headings.append({"heading": sec.heading, "chunk_id": cid})
        product = FILE_PRODUCT.get(path.name)
        clean = clean_markdown(raw)
        docs.append(
            {
                "source_file": path.name,
                "source_sha": file_sha(path),
                "title": title,
                "product": product,
                "rel_path": str(path.relative_to(kb_dir)),
                "clean_text": clean,
                "n_chars": len(clean),
                "n_sections": len(sections),
                "headings": headings,
            }
        )
    return docs


def build_dossiers(chunk_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {r["chunk_id"]: r for r in chunk_records}
    out: list[dict[str, Any]] = []
    for card in PRODUCT_CARDS:
        snippets: list[dict[str, str]] = []
        for cid in (card.get("overview_chunk_ids") or card.get("chunk_ids") or [])[:3]:
            rec = by_id.get(cid)
            if not rec:
                continue
            body = (rec.get("text_raw") or "").strip()
            if rec.get("chunk_type") == "table" and rec.get("caption"):
                body = f"{rec['caption']}\n{body}"
            snippets.append(
                {
                    "chunk_id": cid,
                    "heading": rec.get("heading") or "",
                    "text": body[:900],
                }
            )
        out.append(
            {
                "product": card["product"],
                "source_file": card["source_file"],
                "pitch": card.get("what_it_is") or card.get("summary") or "",
                "fits": card.get("fits") or "",
                "not_for": list(card.get("not_for") or []),
                "tasks": list(card.get("tasks") or []),
                "overview_chunk_ids": list(card.get("overview_chunk_ids") or []),
                "snippets": snippets,
            }
        )
    return out
