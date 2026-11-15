"""Blocking validation for build artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kb_build.aliases import build_id_aliases, load_required_gold_ids, nfc_chunk_id, resolve_id
from kb_build.enrich import CANONICAL_PRODUCTS


def validate_build(
    build_dir: Path,
    kb_dir: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    chunks_path = build_dir / "chunks.jsonl"
    aliases_path = build_dir / "id_aliases.json"
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(ln) for ln in lines if ln.strip()]
    by_id = {c["chunk_id"]: c for c in chunks}
    aliases: dict[str, str] = json.loads(aliases_path.read_text(encoding="utf-8"))

    required = load_required_gold_ids(dataset_path)
    produced_norm = {resolve_id(c["chunk_id"], aliases) for c in chunks}
    required_norm = {resolve_id(r, aliases) for r in required}
    # Gold ids must appear literally in chunks.jsonl
    produced_literal = {nfc_chunk_id(c["chunk_id"]) for c in chunks}
    missing_gold_literal = sorted(required - produced_literal)

    missing_after_alias = sorted(
        r for r in required if resolve_id(r, aliases) not in {c["chunk_id"] for c in chunks} and r not in produced_literal
    )

    dup_ids = [cid for cid in by_id if sum(1 for c in chunks if c["chunk_id"] == cid) > 1]

    errors: list[str] = []
    warnings: list[str] = []

    if missing_gold_literal:
        errors.append(f"missing gold chunk_ids in chunks.jsonl: {missing_gold_literal[:10]}")

    if dup_ids:
        errors.append(f"duplicate chunk_id values: {dup_ids[:5]}")

    for c in chunks:
        if not (c.get("text_raw") or "").strip():
            errors.append(f"empty text_raw: {c['chunk_id']}")
        for p in c.get("products") or []:
            if p not in CANONICAL_PRODUCTS:
                errors.append(f"unknown product {p!r} in {c['chunk_id']}")
        for rid in c.get("related_ids") or []:
            if rid not in by_id and rid not in aliases:
                if resolve_id(rid, aliases) not in by_id:
                    errors.append(f"related_ids missing target {rid} from {c['chunk_id']}")
        span = c.get("char_span") or []
        if len(span) != 2 or span[0] > span[1]:
            warnings.append(f"bad char_span for {c['chunk_id']}")

    # card chunk ids exist (literal or alias)
    cards_path = build_dir / "cards.jsonl"
    if cards_path.exists():
        for line in cards_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            card = json.loads(line)
            for cid in card.get("chunk_ids") or []:
                if cid not in produced_literal and resolve_id(cid, aliases) not in by_id:
                    warnings.append(f"card missing chunk {cid}")

    dos_path = build_dir / "dossiers.jsonl"
    docs_path = build_dir / "documents.jsonl"
    if not docs_path.exists():
        warnings.append("missing documents.jsonl")
    if dos_path.exists():
        n_dos = sum(1 for ln in dos_path.read_text(encoding="utf-8").splitlines() if ln.strip())
        if n_dos != 5:
            warnings.append(f"expected 5 product dossiers, got {n_dos}")
    else:
        warnings.append("missing dossiers.jsonl")

    ok = not errors
    return {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "chunks": len(chunks),
            "required_gold": len(required),
            "missing_gold_literal": len(missing_gold_literal),
        },
        "missing_gold_literal": missing_gold_literal,
        "required_norm_covered": len(required_norm - produced_norm) == 0,
    }