"""Consultant v4 writer: clean sources, no extractive top-1 fallback."""

from __future__ import annotations

import re
from typing import Any

from agent import config, prompts, templates
from agent.context import cards_text as cards_overview
from agent.corpus import all_chunks_as_scored
from agent.llm import get_client, model_available
from agent.schemas import (
    ContextPack,
    GenerateResult,
    PlannerResult,
    ScoredChunk,
    SourceBlock,
    UnderstandResult,
)
from agent.store import Store

_BAD_START = (
    "таблица в разделе",
    "основной вопрос",
    "документ:",
    "раздел:",
    "назначение:",
    "согласно разделу",
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


def format_sources(pack: ContextPack, store: Store | None = None) -> str:
    if getattr(config, "FULL_KB", False) and store is not None and getattr(store, "kb_prompt", ""):
        return store.kb_prompt
    blocks: list[str] = []
    total = 0
    budget = config.MAX_KB_CHARS if getattr(config, "FULL_KB", False) else config.MAX_EVIDENCE_CHARS
    for b in pack.blocks:
        text = (b.text or "").strip()
        if not text:
            continue
        chunk = f"[{b.chunk_id}] {b.heading}\n{text}"
        if total + len(chunk) > budget and blocks:
            break
        blocks.append(chunk)
        total += len(chunk)
    return "\n\n".join(blocks)


def pack_as_scored(pack: ContextPack) -> list[ScoredChunk]:
    out: list[ScoredChunk] = []
    for i, b in enumerate(pack.blocks):
        out.append(
            ScoredChunk(
                chunk_id=b.chunk_id,
                score=1.0 - i * 0.01,
                text=b.text,
                source_file=b.source_file,
                heading=b.heading,
                n_tokens=len((b.text or "").split()),
                products=list(b.products),
            )
        )
    return out


def _dialog_summary(history: list[dict] | None) -> str:
    history = history or []
    parts: list[str] = []
    for turn in history[-6:]:
        if "user" in turn and isinstance(turn.get("user"), str):
            parts.append(f"user: {turn['user']}")
            if isinstance(turn.get("assistant"), str):
                parts.append(f"assistant: {turn['assistant']}")
            continue
        role = turn.get("role") or "user"
        content = turn.get("content") or turn.get("text") or ""
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts) if parts else "(пусто)"


def _clip_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return (text or "").strip()
    cut = " ".join(words[:max_words]).rstrip(".,;:")
    return cut + "."


def _natural_sentences(text: str, n: int = 2, max_words: int = 90) -> str:
    text = _WS.sub(" ", text or "").strip()
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    kept: list[str] = []
    for p in parts:
        low = p.lower().replace("ё", "е")
        if any(low.startswith(b) for b in _BAD_START):
            continue
        if "таблица в разделе" in low:
            continue
        kept.append(p)
        if len(kept) >= n:
            break
    return _clip_words(" ".join(kept), max_words)


def catalog_template(store: Store) -> GenerateResult:
    bits = [
        "У МТС Energytool пять решений по учёту и снижению затрат на электроэнергию."
    ]
    cites: list[str] = []
    products: list[str] = []
    for card in store.cards:
        products.append(card.product)
        pitch = (card.what_it_is or card.summary or "").rstrip(".")
        bits.append(f"{card.product} — {pitch}.")
        ids = card.overview_chunk_ids or card.chunk_ids
        if ids:
            cites.append(ids[0])
    answer = _clip_words(" ".join(bits), 140)
    return GenerateResult(
        answer=answer,
        cited_ids=cites[:8],
        products=products,
        answer_with_cites=answer,
        mode="answer",
    )


def _card_for(store: Store, product: str):
    return next((c for c in store.cards if c.product == product), None)


def product_template(store: Store, product: str, query: str = "") -> GenerateResult:
    card = _card_for(store, product)
    if not card:
        return GenerateResult(answer=templates.NOT_IN_KB, cited_ids=[], mode="not_found")
    pitch = (card.what_it_is or card.summary or "").rstrip(".")
    parts = [f"{pitch}."]
    qlow = (query or "").lower().replace("ё", "е")
    if card.fits:
        parts.append(f"Подходит: {card.fits.rstrip('.')}.")
    if card.not_for and any(
        x in qlow for x in ("нельзя", "огранич", "начисл", "не подходит", "верифик", "сразу")
    ):
        parts.append("Важно: " + "; ".join(card.not_for[:2]).rstrip(".") + ".")
    cites = list(card.overview_chunk_ids or card.chunk_ids[:2])
    if any(x in qlow for x in ("начисл", "верифик", "нельзя")):
        for cid in card.chunk_ids:
            if "ограничен" in cid.lower() or "не делаем" in cid.lower() or "нельзя" in cid.lower():
                if cid not in cites:
                    cites.append(cid)
    answer = _clip_words(" ".join(parts), 120)
    return GenerateResult(
        answer=answer,
        cited_ids=cites[:5],
        products=[product],
        answer_with_cites=answer,
        mode="answer",
    )


def recommend_template(store: Store, products: list[str], query: str = "") -> GenerateResult:
    cand = list(products or [])[:3] or list(store.product_names)[:3]
    bits = ["Под вашу задачу обычно смотрят такие решения Energytool."]
    cites: list[str] = []
    for p in cand:
        card = _card_for(store, p)
        if not card:
            continue
        pitch = (card.what_it_is or card.summary or "").rstrip(".")
        bits.append(f"{p} — {pitch}.")
        ids = card.overview_chunk_ids or card.chunk_ids
        if ids:
            cites.append(ids[0])
        if card.not_for:
            bits.append("Ограничение: " + card.not_for[0].rstrip(".") + ".")
            break
    answer = _clip_words(" ".join(bits), 130)
    return GenerateResult(
        answer=answer,
        cited_ids=cites[:6],
        products=cand,
        answer_with_cites=answer,
        mode="answer",
        follow_up="Какой объект и что сейчас важнее — счета, учёт или потери?",
    )


def _from_blocks(pack: ContextPack, store: Store, query: str) -> GenerateResult:
    if not pack.blocks:
        return GenerateResult(answer=templates.NOT_IN_KB, cited_ids=[], mode="not_found")
    texts: list[str] = []
    cites: list[str] = []
    products: list[str] = []
    for b in pack.blocks[:4]:
        snip = _natural_sentences(b.text, n=2, max_words=70)
        if snip:
            texts.append(snip)
        if b.chunk_id and b.chunk_id not in cites:
            cites.append(b.chunk_id)
        for p in b.products:
            if p not in products:
                products.append(p)
        if len(texts) >= 2:
            break
    answer = _clip_words(" ".join(texts), 130)
    if not answer:
        if pack.intent == "catalog":
            return catalog_template(store)
        if pack.products:
            return product_template(store, pack.products[0], query)
        return GenerateResult(answer=templates.NOT_IN_KB, cited_ids=[], mode="not_found")
    return GenerateResult(
        answer=answer,
        cited_ids=cites[:5],
        products=products[:5],
        answer_with_cites=answer,
        mode="answer",
    )


def synthetic_answer(
    store: Store,
    planner: PlannerResult | UnderstandResult,
    pack: ContextPack,
) -> GenerateResult:
    """Dossier/section templates — never raw handbook top-1."""
    intent = getattr(planner, "consult_intent", None) or pack.intent or "energy_fact"
    q = getattr(planner, "standalone_query", "") or ""
    products = list(getattr(planner, "explicit_products", None) or pack.products or [])
    if intent == "catalog":
        return catalog_template(store)
    if intent == "product_fact" and products:
        base = product_template(store, products[0], q)
        extra = _from_blocks(pack, store, q)
        if extra.mode != "not_found" and extra.answer and extra.answer != templates.NOT_IN_KB:
            merged = _clip_words(base.answer + " " + extra.answer, 140)
            cites = list(dict.fromkeys(list(base.cited_ids) + list(extra.cited_ids)))
            return GenerateResult(
                answer=merged,
                cited_ids=cites[:6],
                products=list(dict.fromkeys(list(base.products) + list(extra.products))),
                answer_with_cites=merged,
                mode="answer",
            )
        return base
    if intent in ("recommendation", "comparison"):
        return recommend_template(store, products, q)
    return _from_blocks(pack, store, q)


def _parse_sentences(data: dict[str, Any]) -> GenerateResult:
    sentences = data.get("sentences")
    texts: list[str] = []
    cites: list[str] = []
    if isinstance(sentences, list) and sentences:
        for item in sentences:
            if not isinstance(item, dict):
                continue
            t = str(item.get("text") or "").strip()
            if not t:
                continue
            texts.append(t)
            for c in item.get("citations") or []:
                cs = str(c).strip().strip("{}").removeprefix("id=").strip()
                if cs and cs not in cites:
                    cites.append(cs)
        answer = " ".join(texts)
    else:
        answer = str(data.get("answer") or data.get("direct_answer") or "").strip()
        cites = [str(x) for x in (data.get("cited_ids") or [])]
        if not answer and data.get("support_points"):
            pts = [str(x).strip() for x in data.get("support_points") or [] if str(x).strip()]
            answer = " ".join(pts)
    products = [str(x) for x in (data.get("products") or [])]
    follow = data.get("follow_up") or data.get("next_move")
    follow_s = str(follow).strip() if follow else None
    if follow_s in ("null", "None"):
        follow_s = None
    mode = str(data.get("mode") or "answer")
    if follow_s and follow_s not in (answer or ""):
        next_action = "clarify"
    else:
        next_action = "none"
        if follow_s and follow_s not in (answer or ""):
            answer = f"{answer.rstrip()} {follow_s}".strip()
    # Attach a single follow-up question only when writer asked to.
    if follow_s and next_action == "clarify" and follow_s not in (answer or ""):
        answer = f"{answer.rstrip()} {follow_s}".strip()
    return GenerateResult(
        answer=answer,
        cited_ids=cites,
        products=products,
        next_action=next_action,
        missing_slot=None,
        answer_with_cites=answer,
        mode=mode,
        follow_up=follow_s,
    )


def generate(
    store: Store,
    understand: UnderstandResult | PlannerResult,
    pack: ContextPack,
    history: list[dict] | None = None,
    *,
    skip_llm: bool = False,
    mode_hint: str = "answer",
    slots: dict | None = None,
    ranked: list[ScoredChunk] | None = None,
) -> tuple[GenerateResult, str, list[ScoredChunk], int]:
    del slots, ranked
    evidence_blocks = format_sources(pack, store)
    used = all_chunks_as_scored(store) if getattr(config, "FULL_KB", False) else pack_as_scored(pack)
    q = getattr(understand, "standalone_query", "") or ""
    intent = getattr(understand, "consult_intent", None) or pack.intent or "energy_fact"

    if skip_llm or not model_available() or not (pack.blocks or evidence_blocks):
        fb = synthetic_answer(store, understand, pack)
        return fb, evidence_blocks, used, 0

    client = get_client()
    if client is None:
        return synthetic_answer(store, understand, pack), evidence_blocks, used, 0

    messages = [
        {"role": "system", "content": prompts.P6_SYSTEM},
        {
            "role": "user",
            "content": prompts.p6_user(
                standalone_query=q,
                cards_text=pack.cards_text or cards_overview(store, pack.products),
                evidence_blocks=evidence_blocks,
                dialog_summary=_dialog_summary(history),
                consult_intent=intent,
                mode_hint=mode_hint,
            ),
        },
    ]
    before = client.calls
    try:
        data = client.chat_json(
            messages,
            temperature=config.P6_TEMP,
            max_tokens=config.P6_MAX_TOKENS,
            stage="P6",
            max_json_attempts=int(config.MAX_ATTEMPTS.get("P6", 1)),
        )
        n = max(0, client.calls - before)
        if not data:
            return synthetic_answer(store, understand, pack), evidence_blocks, used, n
        result = _parse_sentences(data)
        if not (result.answer or "").strip():
            return synthetic_answer(store, understand, pack), evidence_blocks, used, n
        return result, evidence_blocks, used, n
    except Exception:  # noqa: BLE001
        n = max(0, client.calls - before)
        return synthetic_answer(store, understand, pack), evidence_blocks, used, n


# Compatibility shims
def query_topic_cues(question: str) -> dict[str, Any]:
    return {"photo_meter": False, "preferred_products": [], "anchors": []}


def select_aligned(ranked: list[ScoredChunk], question: str, store: Store | None = None) -> list[ScoredChunk]:
    del question, store
    return list(ranked)


def evidence_aligned(ranked: list[ScoredChunk], question: str, store: Store | None = None) -> bool:
    del question, store
    return bool(ranked)


def chunk_matches_topic(sc: ScoredChunk, cues: dict[str, Any], store: Store | None = None) -> bool:
    del cues, store
    return True


def build_evidence(
    store: Store,
    ranked: list[ScoredChunk],
    **kwargs: Any,
) -> tuple[str, list[ScoredChunk], str]:
    """Legacy helper — prefer ContextPack. Kept for tests."""
    del kwargs
    pack = ContextPack(
        intent="energy_fact",
        blocks=[
            SourceBlock(
                chunk_id=c.chunk_id,
                heading=c.heading,
                text=c.text,
                source_file=c.source_file,
                products=list(c.products),
            )
            for c in ranked[:8]
        ],
        cards_text=cards_overview(store),
        retrieved_ids=[c.chunk_id for c in ranked[:8]],
        retrieved_scores=[c.score for c in ranked[:8]],
        products=[],
    )
    return format_sources(pack), pack_as_scored(pack), pack.cards_text
