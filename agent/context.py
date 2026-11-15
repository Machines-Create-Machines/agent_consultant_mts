"""Doc-first context router (Consultant v4)."""

from __future__ import annotations

from agent import config
from agent.schemas import ContextPack, PlannerResult, ScoredChunk, SourceBlock
from agent.store import Store

_CONSTRAINT_HEAD = (
    "не подходит",
    "чего утверждать нельзя",
    "что мы не делаем",
    "важное ограничение",
)


def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    cut = text[: n - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "…"


def _clean_section(text: str) -> str:
    """Drop table-caption boilerplate that leaks into answers."""
    lines = []
    for line in (text or "").splitlines():
        low = line.lower().replace("ё", "е")
        if low.startswith("таблица в разделе"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _dossier(store: Store, product: str) -> dict | None:
    for d in store.dossiers:
        if d.get("product") == product:
            return d
    card = next((c for c in store.cards if c.product == product), None)
    if not card:
        return None
    return {
        "product": card.product,
        "source_file": card.source_file,
        "pitch": card.what_it_is or card.summary,
        "fits": card.fits,
        "not_for": list(card.not_for),
        "tasks": list(card.tasks),
        "overview_chunk_ids": list(card.overview_chunk_ids or card.chunk_ids[:2]),
        "snippets": [],
    }


def dossier_block(store: Store, product: str) -> SourceBlock | None:
    d = _dossier(store, product)
    if not d:
        return None
    parts = [d.get("pitch") or ""]
    if d.get("fits"):
        parts.append("Кому подходит: " + str(d["fits"]))
    if d.get("not_for"):
        parts.append("Не для: " + "; ".join(d["not_for"][:2]))
    cite = ""
    for sn in d.get("snippets") or []:
        cite = sn.get("chunk_id") or cite
        extra = _clean_section(sn.get("text") or "")
        if extra:
            parts.append(_clip(extra, 420))
            break
    if not cite:
        ids = d.get("overview_chunk_ids") or []
        cite = ids[0] if ids else f"{d.get('source_file')}#Что это"
    return SourceBlock(
        chunk_id=cite,
        heading=product,
        text=" ".join(p for p in parts if p).strip(),
        source_file=str(d.get("source_file") or ""),
        products=[product],
    )


def cards_text(store: Store, products: list[str] | None = None) -> str:
    wanted = set(products or [])
    lines: list[str] = []
    for card in store.cards:
        if wanted and card.product not in wanted:
            continue
        lines.append(f"- {card.product}: {card.what_it_is or card.summary}")
    if not lines:
        for card in store.cards:
            lines.append(f"- {card.product}: {card.what_it_is or card.summary}")
    return "\n".join(lines)


def _parent_lift(store: Store, sc: ScoredChunk) -> ScoredChunk:
    ch = store.get_chunk(sc.chunk_id)
    if not ch:
        return sc
    if ch.n_tokens and ch.n_tokens < 180 and ch.parent_chunk_id and ch.parent_chunk_id != ch.chunk_id:
        parent = store.get_chunk(ch.parent_chunk_id)
        if parent:
            return ScoredChunk(
                chunk_id=parent.chunk_id,
                score=sc.score,
                text=_clean_section(parent.text_raw),
                source_file=parent.source_file,
                heading=parent.heading,
                n_tokens=parent.n_tokens,
                products=list(parent.products),
                branches=dict(sc.branches),
            )
    return ScoredChunk(
        chunk_id=ch.chunk_id,
        score=sc.score,
        text=_clean_section(ch.text_raw if ch.chunk_type != "table" else (ch.caption or ch.text_raw)),
        source_file=ch.source_file,
        heading=ch.heading,
        n_tokens=ch.n_tokens,
        products=list(ch.products),
        branches=dict(sc.branches),
    )


def _is_constraint(heading: str) -> bool:
    low = (heading or "").lower().replace("ё", "е")
    return any(h in low for h in _CONSTRAINT_HEAD)


def _section_blocks(
    store: Store,
    ranked: list[ScoredChunk],
    *,
    limit: int,
    allow_constraint: bool,
    budget: int,
) -> list[SourceBlock]:
    blocks: list[SourceBlock] = []
    seen: set[str] = set()
    used = 0
    for sc in ranked:
        item = _parent_lift(store, sc)
        if item.chunk_id in seen:
            continue
        if _is_constraint(item.heading) and not allow_constraint:
            continue
        text = _clip(item.text, 1200)
        if not text:
            continue
        if used + len(text) > budget and blocks:
            break
        blocks.append(
            SourceBlock(
                chunk_id=item.chunk_id,
                heading=item.heading,
                text=text,
                source_file=item.source_file,
                products=list(item.products),
            )
        )
        seen.add(item.chunk_id)
        used += len(text)
        if len(blocks) >= limit:
            break
    return blocks


def _product_doc_blocks(
    store: Store,
    product: str,
    ranked: list[ScoredChunk],
    *,
    query: str = "",
) -> list[SourceBlock]:
    card = next((c for c in store.cards if c.product == product), None)
    d = _dossier(store, product)
    source_file = (card.source_file if card else "") or (d.get("source_file") if d else "")
    doc = store.documents.get(source_file) if source_file else None
    blocks: list[SourceBlock] = []
    intro = dossier_block(store, product)
    if intro:
        blocks.append(intro)
    # Prefer overview sections, then retrieved from this file, then full clean doc if small.
    overview_ids = list((card.overview_chunk_ids if card else []) or (d.get("overview_chunk_ids") if d else []) or [])
    seen = {b.chunk_id for b in blocks}
    budget = config.MAX_DOC_CHARS
    used = sum(len(b.text) for b in blocks)

    def add_chunk(cid: str) -> None:
        nonlocal used
        cid = store.resolve_id(cid)
        if cid in seen:
            return
        ch = store.get_chunk(cid)
        if not ch:
            return
        text = _clean_section(ch.text_raw)
        if ch.chunk_type == "table":
            text = _clean_section(ch.caption or text)
        text = _clip(text, 1400)
        if not text:
            return
        if used + len(text) > budget and blocks:
            return
        blocks.append(
            SourceBlock(
                chunk_id=ch.chunk_id,
                heading=ch.heading,
                text=text,
                source_file=ch.source_file,
                products=list(ch.products) or [product],
            )
        )
        seen.add(ch.chunk_id)
        used += len(text)

    for cid in overview_ids:
        add_chunk(cid)
    qlow = (query or "").lower().replace("ё", "е")
    want_constraint = any(
        x in qlow for x in ("нельзя", "огранич", "начисл", "не подходит", "верифик")
    )
    if want_constraint and card:
        for cid in card.chunk_ids:
            ch = store.get_chunk(cid)
            if ch and _is_constraint(ch.heading):
                add_chunk(cid)
    for sc in ranked:
        if source_file and sc.source_file == source_file:
            add_chunk(sc.chunk_id)
        elif product in (sc.products or []):
            add_chunk(sc.chunk_id)
    if doc and doc.clean_text and used < budget // 2 and doc.n_chars <= budget:
        # Remaining room: attach truncated clean document as one block.
        rest = _clip(doc.clean_text, budget - used)
        blocks.append(
            SourceBlock(
                chunk_id=overview_ids[0] if overview_ids else f"{source_file}#",
                heading=doc.title,
                text=rest,
                source_file=source_file,
                products=[product],
            )
        )
    return blocks


def build_context(
    store: Store,
    planner: PlannerResult,
    ranked: list[ScoredChunk],
    *,
    used_rerank: bool = False,
) -> ContextPack:
    intent = planner.consult_intent or "energy_fact"
    products = list(planner.explicit_products or [])
    if planner.explicit_product and planner.explicit_product not in products:
        products.insert(0, planner.explicit_product)

    if intent == "catalog":
        blocks = [b for p in store.product_names if (b := dossier_block(store, p))]
        ids = [b.chunk_id for b in blocks]
        return ContextPack(
            intent=intent,
            blocks=blocks,
            cards_text=cards_text(store),
            retrieved_ids=ids + [c.chunk_id for c in ranked[:8]],
            retrieved_scores=[round(c.score, 4) for c in ranked[:8]],
            products=list(store.product_names),
            used_rerank=used_rerank,
        )

    if intent == "product_fact" and products:
        blocks = _product_doc_blocks(
            store, products[0], ranked, query=planner.standalone_query
        )
        return ContextPack(
            intent=intent,
            blocks=blocks,
            cards_text=cards_text(store, products[:1]),
            retrieved_ids=[b.chunk_id for b in blocks] + [c.chunk_id for c in ranked[:6]],
            retrieved_scores=[round(c.score, 4) for c in ranked[:6]],
            products=products[:1],
            used_rerank=used_rerank,
        )

    if intent in ("recommendation", "comparison"):
        cand = products[:]
        if not cand:
            cand = list(store.product_names)
        cand = cand[:3]
        blocks = [b for p in cand if (b := dossier_block(store, p))]
        extra = _section_blocks(
            store,
            ranked,
            limit=4,
            allow_constraint=True,
            budget=config.MAX_EVIDENCE_CHARS // 2,
        )
        seen = {b.chunk_id for b in blocks}
        for b in extra:
            if b.chunk_id not in seen:
                blocks.append(b)
                seen.add(b.chunk_id)
        return ContextPack(
            intent=intent,
            blocks=blocks,
            cards_text=cards_text(store, cand),
            retrieved_ids=[b.chunk_id for b in blocks],
            retrieved_scores=[round(c.score, 4) for c in ranked[:8]],
            products=cand,
            used_rerank=used_rerank,
        )

    # energy_fact
    allow_c = any(
        x in (planner.standalone_query or "").lower()
        for x in ("нельзя", "огранич", "начисл", "не подходит")
    )
    blocks = _section_blocks(
        store,
        ranked,
        limit=5,
        allow_constraint=allow_c,
        budget=config.MAX_EVIDENCE_CHARS,
    )
    prods: list[str] = []
    for b in blocks:
        for p in b.products:
            if p not in prods:
                prods.append(p)
    return ContextPack(
        intent=intent,
        blocks=blocks,
        cards_text=cards_text(store, prods),
        retrieved_ids=[b.chunk_id for b in blocks] + [c.chunk_id for c in ranked[:8]],
        retrieved_scores=[round(c.score, 4) for c in ranked[:8]],
        products=prods[:5],
        used_rerank=used_rerank,
    )
