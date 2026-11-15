#!/usr/bin/env python3
"""Stage a platform-compatible flat ZIP (no nested agent/ package)."""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "dist" / "stage"
DIST_ZIP = ROOT / "dist" / "solution.zip"

_FROM_AGENT_MOD = re.compile(r"^(\s*)from agent\.(\w+)(.*)$", re.M)
_FROM_AGENT_PKG = re.compile(r"^(\s*)from agent import (.+)$", re.M)
_IMPORT_AGENT = re.compile(r"^(\s*)import agent\.(\w+)(.*)$", re.M)


def _flatten_imports(src: str) -> str:
    src = _FROM_AGENT_MOD.sub(r"\1from \2\3", src)
    src = _IMPORT_AGENT.sub(r"\1import \2\3", src)

    def _pkg_import(m: re.Match[str]) -> str:
        indent, names = m.group(1), m.group(2)
        # from agent import config, pipeline -> import config / from pipeline ...
        parts = [n.strip() for n in names.split(",") if n.strip()]
        lines = []
        for part in parts:
            if " as " in part:
                real, _, alias = part.partition(" as ")
                real, alias = real.strip(), alias.strip()
                lines.append(f"{indent}import {real} as {alias}" if real == alias else f"{indent}import {real} as {alias}")
            else:
                lines.append(f"{indent}import {part}")
        return "\n".join(lines)

    return _FROM_AGENT_PKG.sub(_pkg_import, src)


def _rewrite_config(src: str) -> str:
    # Flat layout: modules sit next to build/
    return src.replace(
        "ROOT = Path(__file__).resolve().parent.parent",
        "ROOT = Path(__file__).resolve().parent",
        1,
    )


def main() -> None:
    if not (ROOT / "build" / "chunks.jsonl").is_file():
        raise SystemExit("build/chunks.jsonl missing; run: python -m kb_build")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    (STAGE / "build").mkdir()

    # Flatten agent modules (skip package __init__)
    for path in sorted((ROOT / "agent").glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        text = _flatten_imports(text)
        if path.name == "config.py":
            text = _rewrite_config(text)
        (STAGE / path.name).write_text(text, encoding="utf-8")

    sol = (ROOT / "solution.py").read_text(encoding="utf-8")
    sol = _flatten_imports(sol)
    (STAGE / "solution.py").write_text(sol, encoding="utf-8")

    shutil.copy2(ROOT / "requirements.txt", STAGE / "requirements.txt")
    for name in ("chunks.jsonl", "cards.jsonl", "glossary.json", "id_aliases.json"):
        shutil.copy2(ROOT / "build" / name, STAGE / "build" / name)
    for name in ("embeddings.npy", "embedding_ids.json"):
        src = ROOT / "build" / name
        if src.is_file():
            shutil.copy2(src, STAGE / "build" / name)

    if DIST_ZIP.exists():
        DIST_ZIP.unlink()
    with zipfile.ZipFile(DIST_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGE).as_posix())
    print(f"wrote {DIST_ZIP}")


if __name__ == "__main__":
    main()
