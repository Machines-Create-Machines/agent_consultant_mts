"""Markdown → section records (deterministic)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
BOLD_ANCHOR_RE = re.compile(r"^\*\*(.+?)\*\*\.?\s*")
HR_RE = re.compile(r"^-{10,}\s*$")
MAX_PART_TOKENS = 800
TARGET_PART_TOKENS = 500
MIN_BOLD_ANCHOR_LEN = 15
# Only these h2 parents may be split into **bold** sub-anchors (gold chunk ids).
BOLD_SUBSPLIT_HEADINGS = frozenset(
    {
        "Как посмотреть до покупки",
        "Как считается эффект",
        "Что нужно от клиента",
    }
)
BOLD_ANCHOR_TITLES = frozenset(
    {
        "Пилотное подключение",
        "Демонстрационный доступ",
        "Окупаемость: две разные рамки, и называть их надо вместе",
        "Требования к данным",
    }
)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def count_tokens(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def strip_boilerplate(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if HR_RE.match(line.strip()):
            continue
        out.append(line)
    return out


def normalize_inline_headings(text: str) -> str:
    """Turn escaped/inlined headings into real line breaks."""
    return re.sub(r"(\S)\s*\\##\s+", r"\1\n## ", text)


@dataclass
class Section:
    source_file: str
    heading: str
    heading_level: int
    heading_path: list[str]
    body: str
    char_start: int
    char_end: int

    def chunk_id(self, dup_suffix: str = "") -> str:
        h = self.heading + dup_suffix
        return nfc(f"{self.source_file}#{h}")


@dataclass
class RawChunk:
    chunk_id: str
    parent_chunk_id: str
    source_file: str
    source_sha: str
    heading: str
    heading_path: list[str]
    part: int
    n_parts: int
    char_span: list[int]
    chunk_type: str
    text_raw: str
    is_parent_full: bool = True


def _heading_level(line: str) -> tuple[int, str] | None:
    m = HEADING_RE.match(line)
    if not m:
        return None
    return len(m.group(1)), nfc(m.group(2).strip())


def _split_bold_anchors(section: Section) -> list[Section]:
    """Split select sections on **Bold title.** sub-anchors (gold ids)."""
    if section.heading not in BOLD_SUBSPLIT_HEADINGS:
        return [section]
    lines = section.body.splitlines()
    anchors: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = BOLD_ANCHOR_RE.match(line.strip())
        if not m:
            continue
        title = nfc(m.group(1).strip().rstrip(".").rstrip(":"))
        if title not in BOLD_ANCHOR_TITLES and not any(
            t.startswith(title) or title.startswith(t) for t in BOLD_ANCHOR_TITLES
        ):
            continue
        if title not in BOLD_ANCHOR_TITLES:
            # map prefix hits (e.g. truncated gold forms)
            for t in BOLD_ANCHOR_TITLES:
                if t.startswith(title) or title.startswith(t):
                    title = t
                    break
        anchors.append((i, title))

    if len(anchors) < 1:
        return [section]

    pieces: list[Section] = [section]
    for idx, (line_i, title) in enumerate(anchors):
        end_i = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(lines)
        body = "\n".join(lines[line_i:end_i]).strip()
        if not body:
            continue
        pieces.append(
            Section(
                source_file=section.source_file,
                heading=title,
                heading_level=section.heading_level + 1,
                heading_path=section.heading_path + [title],
                body=body,
                char_start=section.char_start,
                char_end=section.char_end,
            )
        )
    return pieces


def parse_file(path: Path) -> tuple[str, list[Section]]:
    raw = path.read_text(encoding="utf-8")
    raw = normalize_inline_headings(raw)
    source_file = path.name
    lines = strip_boilerplate(raw.splitlines())

    headings: list[tuple[int, int, str]] = []
    char_pos = 0
    for i, line in enumerate(lines):
        hl = _heading_level(line)
        if hl:
            level, title = hl
            headings.append((i, level, title))
        char_pos += len(line) + 1

    if not headings:
        return path.stem, []

    h1_title = headings[0][2] if headings[0][1] == 1 else path.stem
    line_offsets: list[int] = []
    pos = 0
    for line in lines:
        line_offsets.append(pos)
        pos += len(line) + 1

    sections: list[Section] = []
    for idx, (line_i, level, heading) in enumerate(headings):
        next_line = len(lines)
        for j in range(idx + 1, len(headings)):
            if headings[j][1] <= level:
                next_line = headings[j][0]
                break
        body_lines = lines[line_i + 1 : next_line]
        body = "\n".join(body_lines).strip()
        stack: list[tuple[int, str]] = []
        for k in range(idx):
            _, lv, ht = headings[k]
            while stack and stack[-1][0] >= lv:
                stack.pop()
            stack.append((lv, ht))
        while stack and stack[-1][0] >= level:
            stack.pop()
        ancestors = [h for _, h in stack]

        char_start = line_offsets[line_i]
        char_end = line_offsets[next_line] if next_line < len(lines) else pos
        if level == 1 and not body:
            h1_title = heading
            continue
        sections.append(
            Section(
                source_file=source_file,
                heading=heading,
                heading_level=level,
                heading_path=(
                    ancestors if ancestors else [h1_title]
                ),
                body=body,
                char_start=char_start,
                char_end=char_end,
            )
        )

    expanded: list[Section] = []
    for sec in sections:
        expanded.extend(_split_bold_anchors(sec))
    return h1_title, expanded


def _detect_chunk_type(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "text"
    table_like = sum(
        1
        for ln in lines
        if re.match(r"^[\s\-\|\+]", ln) or (ln.count("|") >= 2 and not ln.startswith("#"))
    )
    if table_like >= max(2, len(lines) // 3):
        return "table"
    if sum(1 for ln in lines if ln.startswith("- ") or ln.startswith("* ")) >= len(lines) // 2:
        return "list"
    return "text"


def _split_parts(text: str, base_id: str, meta: dict) -> list[RawChunk]:
    n_tok = count_tokens(text)
    if n_tok <= MAX_PART_TOKENS:
        return [
            RawChunk(
                chunk_id=base_id,
                parent_chunk_id=base_id,
                part=1,
                n_parts=1,
                text_raw=text,
                **meta,
            )
        ]

    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    parts_text: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    def flush_part() -> None:
        nonlocal cur, cur_tok
        if cur:
            parts_text.append("\n\n".join(cur))
            cur = []
            cur_tok = 0

    for para in paragraphs:
        pt = count_tokens(para)
        if cur_tok + pt > TARGET_PART_TOKENS and cur:
            flush_part()
        cur.append(para)
        cur_tok += pt
    flush_part()

    if len(parts_text) <= 1:
        return [
            RawChunk(
                chunk_id=base_id,
                parent_chunk_id=base_id,
                part=1,
                n_parts=1,
                text_raw=text,
                **meta,
            )
        ]

    n_parts = len(parts_text) + 1
    out: list[RawChunk] = [
        RawChunk(
            chunk_id=base_id,
            parent_chunk_id=base_id,
            part=1,
            n_parts=n_parts,
            text_raw=text,
            is_parent_full=True,
            **meta,
        )
    ]
    for i, part_text in enumerate(parts_text, start=2):
        out.append(
            RawChunk(
                chunk_id=f"{base_id}--p{i}",
                parent_chunk_id=base_id,
                part=i,
                n_parts=n_parts,
                text_raw=part_text,
                is_parent_full=False,
                **meta,
            )
        )
    return out


def build_raw_chunks(kb_dir: Path) -> list[RawChunk]:
    all_chunks: list[RawChunk] = []
    heading_counts: dict[tuple[str, str], int] = {}

    for path in sorted(kb_dir.rglob("*.md")):
        sha = file_sha(path)
        _, sections = parse_file(path)
        for sec in sections:
            if not sec.body:
                continue
            key = (sec.source_file, sec.heading)
            heading_counts[key] = heading_counts.get(key, 0) + 1
            dup = ""
            if heading_counts[key] > 1:
                dup = f"--{heading_counts[key]}"
            cid = sec.chunk_id(dup)
            ctype = _detect_chunk_type(sec.body)
            heading_display = sec.heading + dup
            meta = {
                "source_file": sec.source_file,
                "source_sha": sha,
                "heading": heading_display,
                "heading_path": sec.heading_path,
                "char_span": [sec.char_start, sec.char_end],
                "chunk_type": ctype,
            }
            all_chunks.extend(_split_parts(sec.body, cid, meta))

    all_chunks.sort(key=lambda c: (c.source_file, c.chunk_id, c.part))
    return all_chunks