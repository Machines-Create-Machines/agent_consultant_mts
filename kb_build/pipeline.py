"""Orchestrate KB build artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kb_build import CHUNKER_VERSION
from kb_build.aliases import build_id_aliases, load_required_gold_ids, nfc_chunk_id
from kb_build.cards import PRODUCT_CARDS
from kb_build.chunker import build_raw_chunks, file_sha, parse_file
from kb_build.documents import build_documents, build_dossiers
from kb_build.enrich import finalize_chunks
from kb_build.glossary import GLOSSARY
from kb_build.validate import validate_build


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB = ROOT / "data" / "knowledge-base"
DEFAULT_OUT = ROOT / "build"
DEFAULT_DATASET = ROOT / "data" / "questions_dataset.jsonl"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _clone_for_gold_id(rec: dict[str, Any], gold_id: str) -> dict[str, Any]:
    out = dict(rec)
    out["chunk_id"] = nfc_chunk_id(gold_id)
    out["parent_chunk_id"] = nfc_chunk_id(gold_id)
    hp = list(out.get("heading_path") or [])
    if gold_id.split("#", 1)[-1] not in hp:
        hp.append(gold_id.split("#", 1)[-1])
    out["heading_path"] = hp
    out["heading"] = gold_id.split("#", 1)[-1]
    return out


def apply_gold_aliases(records: list[dict[str, Any]], aliases: dict[str, str]) -> list[dict[str, Any]]:
    by_id = {r["chunk_id"]: r for r in records}
    out = list(records)
    existing = {r["chunk_id"] for r in records}
    for gold, canonical in sorted(aliases.items()):
        gold = nfc_chunk_id(gold)
        canonical = nfc_chunk_id(canonical)
        if gold in existing:
            continue
        base = by_id.get(canonical)
        if not base:
            continue
        out.append(_clone_for_gold_id(base, gold))
        existing.add(gold)
    out.sort(key=lambda r: (r["source_file"], r["chunk_id"], r.get("part", 1)))
    return out


def build_docs(kb_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(kb_dir.rglob("*.md")):
        title, sections = parse_file(path)
        docs.append(
            {
                "source_file": path.name,
                "source_sha": file_sha(path),
                "title": title,
                "n_sections": len(sections),
                "rel_path": str(path.relative_to(kb_dir)),
            }
        )
    return docs


def run_build(
    kb_dir: Path = DEFAULT_KB,
    out_dir: Path = DEFAULT_OUT,
    dataset_path: Path = DEFAULT_DATASET,
) -> dict[str, Any]:
    raw = build_raw_chunks(kb_dir)
    records = finalize_chunks(raw)
    required = load_required_gold_ids(dataset_path)
    canonical_ids = {r["chunk_id"] for r in records}
    aliases = build_id_aliases(required, canonical_ids)
    # Merge auto-detected with explicit unresolved gold
    for gold in sorted(required):
        gold = nfc_chunk_id(gold)
        if gold in canonical_ids:
            continue
        from kb_build.aliases import GOLD_TO_CANONICAL

        canon = nfc_chunk_id(GOLD_TO_CANONICAL.get(gold, gold))
        if canon in canonical_ids and gold != canon:
            aliases[gold] = canon

    records = apply_gold_aliases(records, aliases)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "chunks.jsonl", records)
    _write_jsonl(out_dir / "docs.jsonl", build_docs(kb_dir))
    _write_jsonl(out_dir / "documents.jsonl", build_documents(kb_dir, records))
    _write_jsonl(out_dir / "dossiers.jsonl", build_dossiers(records))
    _write_jsonl(out_dir / "cards.jsonl", PRODUCT_CARDS)

    (out_dir / "glossary.json").write_text(
        json.dumps(GLOSSARY, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "id_aliases.json").write_text(
        json.dumps(aliases, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_shas = {p.name: file_sha(p) for p in sorted(kb_dir.rglob("*.md"))}
    manifest = {
        "chunker_version": CHUNKER_VERSION,
        "source_shas": source_shas,
        "counts": {
            "chunks": len(records),
            "docs": len(source_shas),
            "gold_required": len(required),
            "aliases": len(aliases),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_build(out_dir, kb_dir, dataset_path)
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report