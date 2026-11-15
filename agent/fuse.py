"""Min-max fusion + doc cap + MMR."""

from __future__ import annotations

from agent import config
from agent.retrieve import _minmax
from agent.store import Store, tokenize
from agent.schemas import ScoredChunk


def fuse(
    store: Store,
    branches: dict[str, dict[str, float]],
    *,
    top_n: int | None = None,
) -> list[ScoredChunk]:
    top_n = top_n or config.FUSE_TOP_N
    weights = config.FUSION_WEIGHTS

    # cards/relations fold into dense-like boost via entity or dense channel
    dense = dict(branches.get("dense") or {})
    for cid, sc in (branches.get("cards") or {}).items():
        dense[cid] = max(dense.get(cid, 0.0), sc)
    for cid, sc in (branches.get("relations") or {}).items():
        dense[cid] = max(dense.get(cid, 0.0), sc)

    bm25_n = _minmax(branches.get("bm25") or {})
    dense_n = _minmax(dense)
    quest_n = _minmax(branches.get("questions") or {})
    ent_n = _minmax(branches.get("entity") or {})

    all_ids = set(bm25_n) | set(dense_n) | set(quest_n) | set(ent_n)
    fused: list[tuple[str, float, dict[str, float]]] = []
    for cid in all_ids:
        parts = {
            "dense": dense_n.get(cid, 0.0),
            "bm25": bm25_n.get(cid, 0.0),
            "questions": quest_n.get(cid, 0.0),
            "entity": ent_n.get(cid, 0.0),
        }
        # If dense missing entirely, redistribute weight to bm25
        w = dict(weights)
        if not dense_n:
            w["bm25"] = w["bm25"] + w["dense"]
            w["dense"] = 0.0
        score = sum(w[k] * parts[k] for k in w)
        fused.append((cid, score, parts))

    fused.sort(key=lambda x: -x[1])

    # dedup + cap 2 per source_file
    per_doc: dict[str, int] = {}
    capped: list[tuple[str, float, dict[str, float]]] = []
    for cid, score, parts in fused:
        ch = store.get_chunk(cid)
        if not ch:
            continue
        doc = ch.source_file or cid
        if per_doc.get(doc, 0) >= 2:
            continue
        per_doc[doc] = per_doc.get(doc, 0) + 1
        capped.append((cid, score, parts))

    # MMR
    selected = _mmr(store, capped, top_n=top_n, lam=config.MMR_LAMBDA)

    out: list[ScoredChunk] = []
    for cid, score, parts in selected:
        ch = store.get_chunk(cid)
        assert ch is not None
        out.append(
            ScoredChunk(
                chunk_id=cid,
                score=score,
                text=ch.text_raw,
                source_file=ch.source_file,
                heading=ch.heading,
                n_tokens=ch.n_tokens,
                products=list(ch.products),
                branches=parts,
            )
        )
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _mmr(
    store: Store,
    candidates: list[tuple[str, float, dict[str, float]]],
    *,
    top_n: int,
    lam: float,
) -> list[tuple[str, float, dict[str, float]]]:
    if not candidates:
        return []
    token_sets = {
        cid: set(tokenize(store.chunks[cid].text_raw if cid in store.chunks else ""))
        for cid, _, _ in candidates
    }
    selected: list[tuple[str, float, dict[str, float]]] = []
    remaining = list(candidates)
    while remaining and len(selected) < top_n:
        best_i = 0
        best_val = -1e18
        for i, (cid, score, parts) in enumerate(remaining):
            if not selected:
                val = score
            else:
                max_sim = max(
                    _jaccard(token_sets[cid], token_sets[s[0]]) for s in selected
                )
                val = lam * score - (1 - lam) * max_sim
            if val > best_val:
                best_val = val
                best_i = i
        selected.append(remaining.pop(best_i))
    return selected


def normalize_scores_for_coverage(items: list[ScoredChunk]) -> list[ScoredChunk]:
    """Min-max scores to [0,1] for thresholding without reranker."""
    if not items:
        return items
    vals = [x.score for x in items]
    lo, hi = min(vals), max(vals)
    out: list[ScoredChunk] = []
    for x in items:
        ns = 1.0 if hi - lo < 1e-9 else (x.score - lo) / (hi - lo)
        out.append(
            ScoredChunk(
                chunk_id=x.chunk_id,
                score=ns,
                text=x.text,
                source_file=x.source_file,
                heading=x.heading,
                n_tokens=x.n_tokens,
                products=list(x.products),
                branches=dict(x.branches),
            )
        )
    return out
