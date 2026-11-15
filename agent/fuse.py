"""RRF fusion with parent/part dedupe and product diversity."""

from __future__ import annotations

from agent import config
from agent.store import Store
from agent.schemas import ScoredChunk


def _rank_map(scores: dict[str, float]) -> dict[str, int]:
    return {
        cid: i + 1
        for i, (cid, _) in enumerate(sorted(scores.items(), key=lambda x: -x[1]))
    }


def fuse(
    store: Store,
    branches: dict[str, dict[str, float]],
    *,
    top_n: int | None = None,
    diversify_products: bool = False,
) -> list[ScoredChunk]:
    top_n = top_n or config.FUSE_TOP_N
    k = int(config.RRF_K)

    dense = dict(branches.get("dense") or {})
    bm25 = dict(branches.get("bm25") or {})
    cards = dict(branches.get("cards") or {})
    questions = dict(branches.get("questions") or {})
    entity = dict(branches.get("entity") or {})

    dense_rank = _rank_map(dense)
    bm25_rank = _rank_map(bm25)
    q_rank = _rank_map(questions)
    e_rank = _rank_map(entity)

    all_ids = set(dense) | set(bm25) | set(cards) | set(questions) | set(entity)
    fused: list[tuple[str, float, dict[str, float]]] = []
    for cid in all_ids:
        score = 0.0
        parts: dict[str, float] = {}
        if cid in dense_rank:
            contrib = 1.0 / (k + dense_rank[cid])
            score += contrib
            parts["dense"] = contrib
        if cid in bm25_rank:
            contrib = 1.0 / (k + bm25_rank[cid])
            score += contrib
            parts["bm25"] = contrib
        if cid in q_rank:
            contrib = 0.85 / (k + q_rank[cid])
            score += contrib
            parts["questions"] = contrib
        if cid in e_rank:
            contrib = 0.7 / (k + e_rank[cid])
            score += contrib
            parts["entity"] = contrib
        if cid in cards:
            boost = 0.01 * float(cards[cid])
            score += boost
            parts["cards"] = boost
        fused.append((cid, score, parts))

    fused.sort(key=lambda x: -x[1])

    per_doc: dict[str, int] = {}
    per_parent: dict[str, int] = {}
    per_product: dict[str, int] = {}
    capped: list[tuple[str, float, dict[str, float]]] = []
    for cid, score, parts in fused:
        ch = store.get_chunk(cid)
        if not ch:
            continue
        parent = ch.parent_chunk_id or cid
        if "--p" in cid and parent != cid:
            if per_parent.get(parent, 0) >= 1:
                continue
        doc = ch.source_file or cid
        if per_doc.get(doc, 0) >= (2 if diversify_products else 3):
            continue
        if diversify_products and ch.products:
            if any(per_product.get(p, 0) >= 2 for p in ch.products):
                if len(capped) < top_n // 2:
                    continue
        per_doc[doc] = per_doc.get(doc, 0) + 1
        per_parent[parent] = per_parent.get(parent, 0) + 1
        for p in ch.products:
            per_product[p] = per_product.get(p, 0) + 1
        capped.append((cid, score, parts))
        if len(capped) >= top_n:
            break

    out: list[ScoredChunk] = []
    for cid, score, parts in capped:
        ch = store.get_chunk(cid)
        assert ch is not None
        text = ch.text_raw or ""
        if ch.chunk_type == "table" and ch.caption:
            text = f"{ch.caption}\n{ch.text_raw}"
        out.append(
            ScoredChunk(
                chunk_id=cid,
                score=score,
                text=text,
                source_file=ch.source_file,
                heading=ch.heading,
                n_tokens=ch.n_tokens,
                products=list(ch.products),
                branches=parts,
            )
        )
    return out


def normalize_scores_for_coverage(items: list[ScoredChunk]) -> list[ScoredChunk]:
    """Identity — keep absolute/RRF/rerank scores (no min-max inflation)."""
    return list(items)
