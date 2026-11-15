"""Hybrid retrieve branches A–E (+ F stub for comparison)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent import config
from agent.store import Store, tokenize
from agent.schemas import Card, UnderstandResult


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# Lexical expansions for phrasings the glossary does not cover.
_QUERY_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("симк", "сим-карт", "sim"), "облачная платформа учёта АСКУЭ передача данных"),
    (("кабинет", "цех", "цеха"), "технический учёт АСКУЭ платформа магазины"),
    (("фасад", "скат", "кровл", "крыш"), "солнечные панели электростанция"),
)


def expand_query(query: str) -> str:
    low = (query or "").lower().replace("ё", "е")
    extras = [hint for keys, hint in _QUERY_HINTS if any(k in low for k in keys)]
    if not extras:
        return query
    return query + " " + " ".join(extras)


def _dense_scores(store: Store, query: str, top: int) -> dict[str, float]:
    if store.embeddings is None or not store.embedding_ids:
        return {}
    try:
        import numpy as np  # type: ignore

        from agent.llm import embed_query
    except ImportError:
        return {}
    vec = embed_query(query)
    if not vec:
        return {}
    # Query is L2-normalized in embed_query; index rows L2-normalized at build.
    # Cosine == dot product.
    q = np.asarray(vec, dtype=np.float32)
    mat = np.asarray(store.embeddings, dtype=np.float32)
    if q.ndim != 1 or mat.ndim != 2 or q.shape[0] != mat.shape[1]:
        return {}
    qn = float(np.linalg.norm(q)) or 1.0
    q = q / qn
    mn = np.linalg.norm(mat, axis=1, keepdims=True)
    mn = np.where(mn == 0, 1.0, mn)
    mat_n = mat / mn
    sims = mat_n @ q
    k = min(top, sims.shape[0])
    if k <= 0:
        return {}
    idx = np.argpartition(-sims, k - 1)[:k]
    return {store.embedding_ids[int(i)]: float(sims[int(i)]) for i in idx if float(sims[int(i)]) > 0.0}


def _card_alias_hits(store: Store, query: str) -> list[tuple[Card, float]]:
    low = (query or "").lower().replace("ё", "е")
    hits: list[tuple[Card, float]] = []
    for card in store.cards:
        names = [card.product] + list(card.aliases) + list(card.tasks)
        score = 0.0
        for n in names:
            nlow = n.lower().replace("ё", "е")
            if nlow and nlow in low:
                score = max(score, 1.0 if nlow == low else 0.85)
            else:
                # token overlap
                nt = set(tokenize(n))
                qt = set(tokenize(query))
                if nt and qt:
                    ov = len(nt & qt) / max(len(nt), 1)
                    if ov >= 0.5:
                        score = max(score, 0.5 + 0.4 * ov)
        if score > 0:
            hits.append((card, score))
    hits.sort(key=lambda x: -x[1])
    return hits[:3]


def branch_a_cards(store: Store, query: str) -> dict[str, float]:
    """Card alias/task match → boost linked chunk_ids."""
    scores: dict[str, float] = {}
    for card, sc in _card_alias_hits(store, query):
        for cid in card.chunk_ids:
            cid = store.resolve_id(cid)
            scores[cid] = max(scores.get(cid, 0.0), sc)
    return scores


def branch_b_dense(store: Store, query: str) -> dict[str, float]:
    return _dense_scores(store, query, 30)


def branch_c_bm25(store: Store, query: str) -> dict[str, float]:
    if store.bm25 is None:
        return {}
    q = store.expand_query_aliases(query)
    scores: dict[str, float] = {}
    for idx, sc in store.bm25.topk(q, k=30):
        cid = store.chunk_ids[idx]
        scores[cid] = sc
    return scores


def branch_d_questions(store: Store, query: str) -> dict[str, float]:
    if store.bm25_q is None:
        return {}
    agg: dict[str, float] = defaultdict(float)
    for idx, sc in store.bm25_q.topk(query, k=15):
        cid = store.q_chunk_map[idx]
        agg[cid] = max(agg[cid], sc)
    return dict(agg)


def branch_e_entity(store: Store, entities: list[str], query: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    terms = list(entities)
    # also scan glossary terms present in query
    low = (query or "").lower().replace("ё", "е")
    for can, aliases in store.glossary.items():
        for t in [can] + list(aliases):
            if t and t.lower().replace("ё", "е") in low:
                terms.append(t)
    for term in terms:
        tlow = (term or "").lower().replace("ё", "е")
        cids = store.entity_index.get(tlow) or []
        if not cids:
            # fuzzy: substring key match
            for key, vals in store.entity_index.items():
                if tlow in key or key in tlow:
                    cids = vals
                    break
        for cid in cids[:10]:
            scores[cid] = max(scores.get(cid, 0.0), 1.0)
    return scores


def branch_f_relations(store: Store, understand: UnderstandResult) -> dict[str, float]:
    """1-hop related_ids for comparison only."""
    if understand.intent != "comparison":
        return {}
    scores: dict[str, float] = {}
    # seed from card hits
    for card, sc in _card_alias_hits(store, understand.standalone_query):
        for cid in card.chunk_ids[:3]:
            ch = store.get_chunk(cid)
            if not ch:
                continue
            for rid in ch.related_ids[:5]:
                rid = store.resolve_id(rid)
                scores[rid] = max(scores.get(rid, 0.0), 0.6 * sc)
    return scores


def retrieve(
    store: Store,
    understand: UnderstandResult,
    *,
    slots: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Return per-branch chunk_id -> raw score."""
    q0 = expand_query(understand.standalone_query or "")
    slot_bits: list[str] = []
    for val in (slots or {}).values():
        if isinstance(val, dict):
            val = val.get("value")
        if val:
            slot_bits.append(str(val))
    if slot_bits:
        q0 = q0 + " " + " ".join(slot_bits)
    variants = [q0] + [expand_query(v) for v in (understand.query_variants or [])]
    # merge BM25 over variants (max)
    bm25: dict[str, float] = {}
    dense: dict[str, float] = {}
    quest: dict[str, float] = {}
    for q in variants:
        for cid, sc in branch_c_bm25(store, q).items():
            bm25[cid] = max(bm25.get(cid, 0.0), sc)
        for cid, sc in branch_b_dense(store, q).items():
            dense[cid] = max(dense.get(cid, 0.0), sc)
        for cid, sc in branch_d_questions(store, q).items():
            quest[cid] = max(quest.get(cid, 0.0), sc)

    return {
        "cards": branch_a_cards(store, q0),
        "dense": dense,
        "bm25": bm25,
        "questions": quest,
        "entity": branch_e_entity(store, understand.entities, q0),
        "relations": branch_f_relations(store, understand),
    }


def intent_k(intent: str) -> int:
    return int(config.RETRIEVE_K.get(intent, config.RETRIEVE_K["default"]))


def matched_cards(store: Store, query: str) -> list[Card]:
    return [c for c, _ in _card_alias_hits(store, query)]
