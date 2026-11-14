"""Dataset loader, schema validation, hashes, and deterministic splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "id": str,
    "history": list,
    "question": str,
    "behavior": str,
    "allowed_behaviors": list,
    "gold_chunks": list,
    "nuggets": list,
    "forbidden": list,
    "expected_products": list,
    "gold_answer": str,
    "tags": list,
}

GOLD_FIELDS = ("gold_answer", "nuggets", "gold_chunks", "forbidden", "expected_products", "behavior", "allowed_behaviors")


@dataclass
class Case:
    raw: dict[str, Any]
    split: str = "dev"

    @property
    def id(self) -> str:
        return self.raw["id"]

    def public_input(self) -> dict[str, Any]:
        """Fields safe to send to the bot (no gold leakage)."""
        out = {
            "id": self.raw["id"],
            "question": self.raw["question"],
            "history": self.raw.get("history") or [],
        }
        for g in GOLD_FIELDS:
            assert g not in out
        return out


@dataclass
class Dataset:
    cases: list[Case]
    path: Path
    content_hash: str
    by_id: dict[str, Case] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_id = {c.id: c for c in self.cases}

    def select(
        self,
        split: str = "dev",
        tag: str | None = None,
        limit: int | None = None,
    ) -> list[Case]:
        out = list(self.cases)
        if split != "all":
            out = [c for c in out if c.split == split]
        if tag:
            out = [c for c in out if tag in (c.raw.get("tags") or [])]
        # Deterministic order: by id
        out.sort(key=lambda c: c.id)
        if limit is not None:
            out = out[:limit]
        return out


def _fail(msg: str) -> None:
    raise ValueError(f"dataset schema error: {msg}")


def validate_case(obj: Any, index: int) -> dict[str, Any]:
    if not isinstance(obj, dict):
        _fail(f"case[{index}] is not an object")
    for key, typ in REQUIRED_FIELDS.items():
        if key not in obj:
            _fail(f"case[{index}] missing field '{key}'")
        if not isinstance(obj[key], typ):
            _fail(f"case[{index}].{key} expected {typ.__name__}, got {type(obj[key]).__name__}")
    if not obj["id"]:
        _fail(f"case[{index}] empty id")
    if not obj["question"]:
        _fail(f"case[{index}] ({obj['id']}) empty question")
    for i, turn in enumerate(obj["history"]):
        if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
            _fail(f"case[{index}] ({obj['id']}) history[{i}] must have role/content")
    for i, n in enumerate(obj["nuggets"]):
        if not isinstance(n, dict) or "text" not in n or "w" not in n:
            _fail(f"case[{index}] ({obj['id']}) nuggets[{i}] must have text/w")
        if not isinstance(n["text"], str) or not isinstance(n["w"], (int, float)):
            _fail(f"case[{index}] ({obj['id']}) nuggets[{i}] bad types")
    for field in ("allowed_behaviors", "gold_chunks", "forbidden", "expected_products", "tags"):
        for i, item in enumerate(obj[field]):
            if not isinstance(item, str):
                _fail(f"case[{index}] ({obj['id']}) {field}[{i}] must be str")
    return obj


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def assign_split(case_id: str, holdout_pct: int, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "holdout" if bucket < holdout_pct else "dev"


def load_dataset(
    path: Path,
    holdout_pct: int = 30,
    split_seed: int = 42,
) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # true JSONL fallback
        data = []
        for i, line in enumerate(raw_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                _fail(f"line {i}: {exc}")
    if not isinstance(data, list) or not data:
        _fail("root must be a non-empty JSON array (or JSONL)")

    cases: list[Case] = []
    seen: set[str] = set()
    for i, obj in enumerate(data):
        validated = validate_case(obj, i)
        cid = validated["id"]
        if cid in seen:
            _fail(f"duplicate id '{cid}'")
        seen.add(cid)
        split = assign_split(cid, holdout_pct, split_seed)
        cases.append(Case(raw=validated, split=split))

    cases.sort(key=lambda c: c.id)
    return Dataset(cases=cases, path=path, content_hash=_file_hash(path))
