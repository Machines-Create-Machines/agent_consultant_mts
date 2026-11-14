"""Knowledge-base chunk loader for groundedness and stub retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def normalize_heading(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_name: str
    heading: str
    text: str


def _heading_level(line: str) -> int | None:
    m = re.match(r"^(#{1,6})\s+(.+)$", line)
    if not m:
        return None
    return len(m.group(1))


def _load_chunks_jsonl(chunks_path: Path) -> dict[str, Chunk]:
    import json

    chunks: dict[str, Chunk] = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row["chunk_id"]
        chunks[cid] = Chunk(
            chunk_id=cid,
            file_name=row.get("source_file", cid.split("#", 1)[0]),
            heading=row.get("heading", cid.split("#", 1)[-1]),
            text=row.get("text_raw") or "",
        )
    return chunks


def load_chunks(kb_dir: Path) -> dict[str, Chunk]:
    """Load chunks keyed by `file.md#Heading` (prefers `build/chunks.jsonl` if present)."""
    # data/knowledge-base -> ROOT/build; legacy knowledge-base -> ROOT/build
    for candidate in (
        kb_dir.parent / "build" / "chunks.jsonl",
        kb_dir.parent.parent / "build" / "chunks.jsonl",
    ):
        if candidate.exists():
            return _load_chunks_jsonl(candidate)

    chunks: dict[str, Chunk] = {}
    if not kb_dir.exists():
        raise FileNotFoundError(f"knowledge base not found: {kb_dir}")

    for path in sorted(kb_dir.rglob("*.md")):
        file_name = path.name
        lines = path.read_text(encoding="utf-8").splitlines()
        current_heading = path.stem
        current_level = 1
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf, current_heading
            body = "\n".join(buf).strip()
            if not body and not current_heading:
                return
            cid = f"{file_name}#{current_heading}"
            chunks[cid] = Chunk(
                chunk_id=cid,
                file_name=file_name,
                heading=current_heading,
                text=body,
            )
            # normalized alias without punctuation differences
            alias = f"{file_name}#{normalize_heading(current_heading)}"
            chunks.setdefault(alias, chunks[cid])
            buf = []

        for line in lines:
            level = _heading_level(line)
            if level is not None and level <= 3:
                flush()
                current_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
                current_level = level
            else:
                buf.append(line)
        flush()
        _ = current_level  # silence unused in some paths

    return chunks


def resolve_chunk_id(chunk_id: str, chunks: dict[str, Chunk]) -> Chunk | None:
    if chunk_id in chunks:
        return chunks[chunk_id]
    if "#" not in chunk_id:
        return None
    file_name, heading = chunk_id.split("#", 1)
    alias = f"{file_name}#{normalize_heading(heading)}"
    return chunks.get(alias)


def texts_for_ids(ids: list[str], chunks: dict[str, Chunk]) -> str:
    parts: list[str] = []
    for cid in ids:
        ch = resolve_chunk_id(cid, chunks)
        if ch:
            parts.append(ch.text)
    return "\n".join(parts)
