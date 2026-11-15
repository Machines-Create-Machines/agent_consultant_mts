"""Hybrid retrieve: BM25 + dense + question/entity indexes (Consultant v4)."""

from __future__ import annotations

from typing import Any

from agent import config
from agent.schemas import Card, PlannerResult, UnderstandResult
from agent.store import Store


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


def _card_boost(store: Store, query: str, product: str | None) -> dict[str, float]:
    low = (query or "").lower().replace("ё", "е")
    scores: dict[str, float] = {}
    for card in store.cards:
        names = [card.product] + list(card.aliases) + list(card.tasks)
        hit = False
        exact = bool(product and card.product == product)
        if exact:
            hit = True
        else:
            for n in names:
                nlow = (n or "").lower().replace("ё", "е")
                if nlow and nlow in low:
                    hit = True
                    break
        if not hit:
            continue
        overview = {store.resolve_id(c) for c in (card.overview_chunk_ids or [])}
        for cid in list(card.overview_chunk_ids or []) + list(card.chunk_ids):
            cid = store.resolve_id(cid)
            if cid in overview:
                boost = 3.0 if exact else 1.5
            else:
                boost = 0.8 if exact else 0.4
            scores[cid] = max(scores.get(cid, 0.0), boost)
    return scores


def _catalog_boost(store: Store) -> dict[str, float]:
    scores: dict[str, float] = {}
    for cid in (
        "20-portfolio.md#Быстрая навигация по задаче",
        "20-portfolio.md#Обзор продуктов: что подходит под вашу задачу",
    ):
        resolved = store.resolve_id(cid)
        if store.get_chunk(resolved):
            scores[resolved] = 1.2
    for card in store.cards:
        for cid in card.overview_chunk_ids or card.chunk_ids[:2]:
            cid = store.resolve_id(cid)
            scores[cid] = max(scores.get(cid, 0.0), 1.0)
    return scores


def branch_b_dense(store: Store, query: str) -> dict[str, float]:
    return _dense_scores(store, query, config.DENSE_TOP)


def branch_c_bm25(store: Store, query: str) -> dict[str, float]:
    if store.bm25 is None:
        return {}
    q = store.expand_query_aliases(query)
    scores: dict[str, float] = {}
    for idx, sc in store.bm25.topk(q, k=config.BM25_TOP):
        cid = store.chunk_ids[idx]
        scores[cid] = sc
    return scores


def branch_questions(store: Store, query: str) -> dict[str, float]:
    if store.bm25_q is None or not store.q_chunk_map:
        return {}
    scores: dict[str, float] = {}
    for idx, sc in store.bm25_q.topk(query, k=20):
        if 0 <= idx < len(store.q_chunk_map):
            cid = store.q_chunk_map[idx]
            scores[cid] = max(scores.get(cid, 0.0), float(sc))
    return scores


def branch_entity(store: Store, query: str) -> dict[str, float]:
    low = (query or "").lower().replace("ё", "е")
    if not low:
        return {}
    scores: dict[str, float] = {}
    for ent, cids in store.entity_index.items():
        if len(ent) < 4 or ent not in low:
            continue
        for cid in cids[:8]:
            scores[cid] = max(scores.get(cid, 0.0), 1.0)
    return scores


def retrieve(
    store: Store,
    understand: UnderstandResult | PlannerResult,
    *,
    slots: dict[str, Any] | None = None,
    skip_dense: bool = False,
) -> dict[str, dict[str, float]]:
    del slots
    q0 = getattr(understand, "standalone_query", "") or ""
    product = getattr(understand, "explicit_product", None)
    intent = getattr(understand, "consult_intent", "") or ""
    scope = getattr(understand, "query_scope", "specific") or "specific"
    cards = _card_boost(store, q0, product)
    if intent == "catalog" or scope == "catalog_overview":
        for cid, sc in _catalog_boost(store).items():
            cards[cid] = max(cards.get(cid, 0.0), sc)
    qlow = q0.lower().replace("ё", "е")
    if "energytool" in qlow or "энерджи" in qlow:
        for cid in store.chunk_ids:
            if cid.startswith("26-general.md"):
                cards[cid] = max(cards.get(cid, 0.0), 1.4)
    products = list(getattr(understand, "explicit_products", None) or [])
    if product and product not in products:
        products = [product] + products
    for p in products[1:]:
        extra = _card_boost(store, q0, p)
        for cid, sc in extra.items():
            cards[cid] = max(cards.get(cid, 0.0), sc)
    return {
        "dense": {} if skip_dense else branch_b_dense(store, q0),
        "bm25": branch_c_bm25(store, q0),
        "cards": cards,
        "questions": branch_questions(store, q0),
        "entity": branch_entity(store, q0),
        "relations": {},
    }


def intent_k(intent: str) -> int:
    return int(config.RETRIEVE_K.get(intent, config.RETRIEVE_K["default"]))


def matched_cards(store: Store, query: str) -> list[Card]:
    low = (query or "").lower().replace("ё", "е")
    out: list[Card] = []
    for card in store.cards:
        names = [card.product] + list(card.aliases) + list(card.tasks)
        if any((n or "").lower().replace("ё", "е") in low for n in names if n):
            out.append(card)
    return out
