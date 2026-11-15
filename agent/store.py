"""Cold start: load build/ into SQLite + in-memory BM25 (+ optional dense)."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import config
from agent.schemas import Card, Chunk, Document

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
_LOCK = threading.RLock()
_STORE: "Store | None" = None


def tokenize(text: str) -> list[str]:
    text = (text or "").lower().replace("ё", "е")
    return [t for t in _TOKEN_RE.findall(text) if len(t) > 1]


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.df: dict[str, int] = defaultdict(int)
        self.tf: list[dict[str, int]] = []
        for doc in corpus_tokens:
            tf: dict[str, int] = defaultdict(int)
            for t in doc:
                tf[t] += 1
            self.tf.append(tf)
            for t in tf:
                self.df[t] += 1
        self.idf = {
            t: math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }

    def score(self, query_tokens: list[str], idx: int) -> float:
        score = 0.0
        dl = self.doc_len[idx]
        tf = self.tf[idx]
        for q in query_tokens:
            if q not in tf:
                continue
            idf = self.idf.get(q, 0.0)
            freq = tf[q]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
        return score

    def topk(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        qtoks = tokenize(query)
        if not qtoks or not self.N:
            return []
        scored = [(i, self.score(qtoks, i)) for i in range(self.N)]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


@dataclass
class Store:
    chunks: dict[str, Chunk] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)
    glossary: dict[str, list[str]] = field(default_factory=dict)
    id_aliases: dict[str, str] = field(default_factory=dict)
    entity_index: dict[str, list[str]] = field(default_factory=dict)
    product_names: list[str] = field(default_factory=list)
    bm25: BM25 | None = None
    bm25_q: BM25 | None = None  # hyp questions index (flat)
    q_chunk_map: list[str] = field(default_factory=list)  # bm25_q idx -> chunk_id
    embeddings: Any = None  # np.ndarray | None
    embedding_ids: list[str] = field(default_factory=list)
    emb_index: dict[str, int] = field(default_factory=dict)
    documents: dict[str, Document] = field(default_factory=dict)
    dossiers: list[dict[str, Any]] = field(default_factory=list)
    kb_prompt: str = ""
    kb_section_ids: list[str] = field(default_factory=list)
    db_path: Path | None = None

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        cid = self.resolve_id(chunk_id)
        return self.chunks.get(cid)

    def resolve_id(self, chunk_id: str) -> str:
        return self.id_aliases.get(chunk_id, chunk_id)

    def expand_query_aliases(self, query: str) -> str:
        """Append glossary synonyms found in query (lexical branch only)."""
        low = query.lower().replace("ё", "е")
        # Photo-OCR queries: avoid flooding metering FAQ via ПУ→счётчик expansion.
        photoish = any(k in low for k in ("фото", "фотограф", "распозна", "показан"))
        extras: list[str] = []
        for canonical, aliases in self.glossary.items():
            terms = [canonical] + list(aliases)
            can_low = (canonical or "").lower().replace("ё", "е")
            if photoish and can_low in ("пу", "счётчик", "счетчик", "прибор учёта", "прибор учета"):
                continue
            if any(t.lower().replace("ё", "е") in low for t in terms if t):
                extras.extend(terms)
        if not extras:
            return query
        return query + " " + " ".join(dict.fromkeys(extras))


def _chunk_from_row(obj: dict[str, Any]) -> Chunk:
    source = obj.get("source_file") or ""
    text_raw = obj.get("text_raw") or ""
    text_ctx = obj.get("text_indexed") or obj.get("text_ctx") or text_raw
    return Chunk(
        chunk_id=obj["chunk_id"],
        doc_id=source.replace(".md", "") if source else "",
        source_file=source,
        heading=obj.get("heading") or "",
        heading_path=list(obj.get("heading_path") or []),
        text_raw=text_raw,
        text_ctx=text_ctx,
        chunk_type=obj.get("chunk_type") or "text",
        parent_chunk_id=obj.get("parent_chunk_id"),
        n_tokens=int(obj.get("n_tokens") or 0),
        products=list(obj.get("products") or []),
        segments=list(obj.get("segments") or []),
        topics=list(obj.get("topics") or []),
        hyp_questions=list(obj.get("hyp_questions") or []),
        related_ids=list(obj.get("related_ids") or []),
        caption=obj.get("caption"),
    )


def _card_from_row(obj: dict[str, Any]) -> Card:
    product = obj.get("product") or obj.get("name") or ""
    return Card(
        product=product,
        summary=obj.get("summary") or "",
        source_file=obj.get("source_file") or "",
        chunk_ids=list(obj.get("chunk_ids") or []),
        aliases=list(obj.get("aliases") or []),
        tasks=list(obj.get("tasks") or []),
        segments=list(obj.get("segments") or []),
        requires=list(obj.get("requires") or []),
        not_for=list(obj.get("not_for") or []),
        pricing_model=obj.get("pricing_model"),
        caveats=list(obj.get("caveats") or []),
        what_it_is=str(obj.get("what_it_is") or ""),
        fits=str(obj.get("fits") or ""),
        overview_chunk_ids=list(obj.get("overview_chunk_ids") or []),
    )


def _build_entity_index(
    chunks: dict[str, Chunk], glossary: dict[str, list[str]]
) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    # product names as entities
    product_terms: dict[str, list[str]] = {}
    for ch in chunks.values():
        for p in ch.products:
            product_terms.setdefault(p, []).append(ch.chunk_id)

    for name, cids in product_terms.items():
        index[name.lower()] = list(dict.fromkeys(cids))

    for canonical, aliases in glossary.items():
        terms = [canonical] + list(aliases)
        for term in terms:
            tlow = term.lower().replace("ё", "е")
            if len(tlow) < 2:
                continue
            hits: list[str] = []
            for cid, ch in chunks.items():
                blob = (ch.text_ctx + " " + " ".join(ch.products)).lower().replace("ё", "е")
                if tlow in blob:
                    hits.append(cid)
            if hits:
                index[tlow] = list(dict.fromkeys(hits))
    return dict(index)


def _write_sqlite(db_path: Path, store: Store) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE chunks (
              chunk_id TEXT PRIMARY KEY,
              doc_id TEXT,
              source_file TEXT,
              heading TEXT,
              heading_path TEXT,
              text_raw TEXT,
              text_ctx TEXT,
              chunk_type TEXT,
              parent_chunk_id TEXT,
              n_tokens INTEGER,
              products TEXT,
              segments TEXT,
              topics TEXT,
              hyp_questions TEXT
            );
            CREATE TABLE cards (
              product TEXT PRIMARY KEY,
              summary TEXT,
              source_file TEXT,
              chunk_ids TEXT,
              payload TEXT
            );
            CREATE TABLE glossary (
              canonical TEXT PRIMARY KEY,
              aliases TEXT
            );
            CREATE TABLE entity_index (
              entity TEXT PRIMARY KEY,
              chunk_ids TEXT
            );
            CREATE TABLE relations (
              src TEXT, rel_type TEXT, dst TEXT, chunk_id TEXT
            );
            """
        )
        for ch in store.chunks.values():
            conn.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ch.chunk_id,
                    ch.doc_id,
                    ch.source_file,
                    ch.heading,
                    json.dumps(ch.heading_path, ensure_ascii=False),
                    ch.text_raw,
                    ch.text_ctx,
                    ch.chunk_type,
                    ch.parent_chunk_id,
                    ch.n_tokens,
                    json.dumps(ch.products, ensure_ascii=False),
                    json.dumps(ch.segments, ensure_ascii=False),
                    json.dumps(ch.topics, ensure_ascii=False),
                    json.dumps(ch.hyp_questions, ensure_ascii=False),
                ),
            )
        for card in store.cards:
            conn.execute(
                "INSERT INTO cards VALUES (?,?,?,?,?)",
                (
                    card.product,
                    card.summary,
                    card.source_file,
                    json.dumps(card.chunk_ids, ensure_ascii=False),
                    json.dumps(
                        {
                            "aliases": card.aliases,
                            "tasks": card.tasks,
                            "segments": card.segments,
                            "caveats": card.caveats,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        for can, aliases in store.glossary.items():
            conn.execute(
                "INSERT INTO glossary VALUES (?,?)",
                (can, json.dumps(aliases, ensure_ascii=False)),
            )
        for ent, cids in store.entity_index.items():
            conn.execute(
                "INSERT INTO entity_index VALUES (?,?)",
                (ent, json.dumps(cids, ensure_ascii=False)),
            )
        conn.commit()
    finally:
        conn.close()


def _manifest_mtime(build_dir: Path) -> float:
    man = build_dir / "manifest.json"
    if man.exists():
        return man.stat().st_mtime
    chunks = build_dir / "chunks.jsonl"
    return chunks.stat().st_mtime if chunks.exists() else 0.0


def _need_rebuild(build_dir: Path, db_path: Path) -> bool:
    if not db_path.exists():
        return True
    return db_path.stat().st_mtime < _manifest_mtime(build_dir)


def _load_dense(build_dir: Path, store: Store) -> None:
    npy = build_dir / "embeddings.npy"
    ids_path = build_dir / "embedding_ids.json"
    if not npy.exists() or not ids_path.exists():
        return
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return
    try:
        emb = np.asarray(np.load(str(npy)), dtype=np.float32)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if len(ids) != len(emb):
            return
        # Ensure L2 rows (float16 roundtrip / older indexes)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        store.embeddings = emb / norms
        store.embedding_ids = list(ids)
        store.emb_index = {cid: i for i, cid in enumerate(store.embedding_ids)}
    except Exception:  # noqa: BLE001
        store.embeddings = None


def _build_from_jsonl(build_dir: Path) -> Store:
    chunks_path = build_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing {chunks_path}. Run: python -m kb_build"
        )

    store = Store()
    chunks: dict[str, Chunk] = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ch = _chunk_from_row(json.loads(line))
        chunks[ch.chunk_id] = ch
    store.chunks = chunks
    store.chunk_ids = list(chunks.keys())

    cards_path = build_dir / "cards.jsonl"
    if cards_path.exists():
        for line in cards_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                store.cards.append(_card_from_row(json.loads(line)))

    gloss_path = build_dir / "glossary.json"
    if gloss_path.exists():
        store.glossary = json.loads(gloss_path.read_text(encoding="utf-8"))

    alias_path = build_dir / "id_aliases.json"
    if alias_path.exists():
        store.id_aliases = json.loads(alias_path.read_text(encoding="utf-8"))

    store.product_names = [c.product for c in store.cards] or list(config.PRODUCT_NAMES)
    store.entity_index = _build_entity_index(store.chunks, store.glossary)

    docs_path = build_dir / "documents.jsonl"
    if docs_path.exists():
        for line in docs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            sf = obj.get("source_file") or ""
            store.documents[sf] = Document(
                source_file=sf,
                title=obj.get("title") or sf,
                product=obj.get("product"),
                clean_text=obj.get("clean_text") or "",
                n_chars=int(obj.get("n_chars") or 0),
                headings=list(obj.get("headings") or []),
            )
    dos_path = build_dir / "dossiers.jsonl"
    if dos_path.exists():
        store.dossiers = [
            json.loads(line)
            for line in dos_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # BM25 on text_ctx + hyp questions concatenated per chunk
    corpus: list[list[str]] = []
    for cid in store.chunk_ids:
        ch = store.chunks[cid]
        blob = ch.text_ctx + "\n" + "\n".join(ch.hyp_questions)
        corpus.append(tokenize(blob))
    store.bm25 = BM25(corpus)

    # Separate question index
    q_corpus: list[list[str]] = []
    q_map: list[str] = []
    for cid in store.chunk_ids:
        ch = store.chunks[cid]
        for q in ch.hyp_questions or []:
            q_corpus.append(tokenize(q))
            q_map.append(cid)
    store.bm25_q = BM25(q_corpus) if q_corpus else None
    store.q_chunk_map = q_map

    _load_dense(build_dir, store)
    from agent.corpus import format_full_kb

    store.kb_prompt, store.kb_section_ids = format_full_kb(store)
    if len(store.kb_prompt) > config.MAX_KB_CHARS:
        store.kb_prompt = store.kb_prompt[: config.MAX_KB_CHARS]
    return store


def cold_start(build_dir: Path | None = None, force: bool = False) -> Store:
    """Load or rebuild runtime store (module-level singleton)."""
    global _STORE
    with _LOCK:
        if _STORE is not None and not force:
            return _STORE
        build_dir = build_dir or config.BUILD_DIR
        db_path = config.RUNTIME_DB
        store = _build_from_jsonl(build_dir)
        store.db_path = db_path
        if force or _need_rebuild(build_dir, db_path):
            try:
                _write_sqlite(db_path, store)
            except Exception:  # noqa: BLE001 — sqlite is optional cache
                pass
        _STORE = store
        return store


def get_store() -> Store:
    if _STORE is None:
        return cold_start()
    return _STORE
