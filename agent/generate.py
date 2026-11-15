"""P6 answer generation + evidence assembly."""

from __future__ import annotations

import re
from typing import Any

from agent import config, prompts, templates
from agent.llm import get_client, model_available
from agent.store import Store
from agent.schemas import GenerateResult, ScoredChunk, SlotValue, UnderstandResult


def _small_to_big(store: Store, sc: ScoredChunk) -> ScoredChunk:
    ch = store.get_chunk(sc.chunk_id)
    if not ch:
        return sc
    if ch.n_tokens and ch.n_tokens < 200 and ch.parent_chunk_id:
        parent = store.get_chunk(ch.parent_chunk_id)
        if parent and parent.chunk_id != ch.chunk_id:
            return ScoredChunk(
                chunk_id=parent.chunk_id,
                score=sc.score,
                text=parent.text_raw,
                source_file=parent.source_file,
                heading=parent.heading,
                n_tokens=parent.n_tokens,
                products=list(parent.products),
                branches=dict(sc.branches),
            )
    return sc


def build_evidence(
    store: Store,
    ranked: list[ScoredChunk],
    *,
    max_chars: int | None = None,
) -> tuple[str, list[ScoredChunk], str]:
    """Return (evidence_blocks, used_chunks, cards_text)."""
    max_chars = max_chars or config.MAX_EVIDENCE_CHARS
    expanded: list[ScoredChunk] = []
    seen: set[str] = set()
    for sc in ranked:
        item = _small_to_big(store, sc)
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        expanded.append(item)

    # Best first, second last (middle less attended)
    if len(expanded) >= 2:
        ordered = [expanded[0]] + expanded[2:] + [expanded[1]]
    else:
        ordered = list(expanded)

    blocks: list[str] = []
    used: list[ScoredChunk] = []
    total = 0
    for sc in ordered:
        text = (sc.text or "").strip()
        if not text:
            continue
        if _is_table_like(sc, store):
            continue
        block = f"[{sc.chunk_id}] | {sc.heading}\n{text}"
        if total + len(block) > max_chars and used:
            break
        blocks.append(block)
        used.append(sc)
        total += len(block)
    # If everything was tables, keep first non-empty as last resort
    if not used:
        for sc in ordered:
            text = (sc.text or "").strip()
            if not text:
                continue
            blocks.append(f"[{sc.chunk_id}] | {sc.heading}\n{text[:600]}")
            used.append(sc)
            break

    # Cards for products present in used chunks
    products: set[str] = set()
    for sc in used:
        products.update(sc.products)
    card_lines: list[str] = []
    for card in store.cards:
        if card.product in products or any(
            cid in seen for cid in (store.resolve_id(x) for x in card.chunk_ids)
        ):
            card_lines.append(
                f"- {card.product}: {card.summary} "
                f"(chunks: {', '.join(card.chunk_ids[:3])})"
            )
    return "\n\n".join(blocks), used, "\n".join(card_lines) or "(нет)"


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


_TABLE_ONLY = (
    "В материалах есть таблица по этому вопросу, специалист разберёт её с вами."
)


def _is_table_like(chunk: ScoredChunk, store: Store | None) -> bool:
    if store is not None:
        ch = store.get_chunk(chunk.chunk_id)
        if ch is not None and (ch.chunk_type or "").lower() == "table":
            return True
    text = chunk.text or ""
    dash_lines = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[\-—–\s|]+", stripped) and stripped.count("-") + stripped.count("—") + stripped.count("–") >= 3:
            dash_lines += 1
    return dash_lines > 2


def safe_answer(
    ranked: list[ScoredChunk],
    store: Store | None = None,
    *,
    max_words: int = 60,
    specialist_tail: bool = True,
) -> GenerateResult:
    """Single extractive fallback: text chunks only, short, no markdown."""
    from agent.verify import clean_answer

    if not ranked:
        return GenerateResult(answer=templates.NOT_IN_KB, cited_ids=[])
    pick = next((c for c in ranked if not _is_table_like(c, store)), None)
    if pick is None:
        return GenerateResult(
            answer=_TABLE_ONLY,
            cited_ids=[ranked[0].chunk_id],
            products=list(ranked[0].products)[:3],
        )
    text = re.sub(r"\s+", " ", pick.text or "").strip()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    snippet = " ".join(parts[:2]) if parts else text[:400]
    words = snippet.split()
    if len(words) > max_words:
        snippet = " ".join(words[:max_words]).rstrip(".,;:") + "…"
    snippet = clean_answer(snippet)
    if specialist_tail and templates.SPECIALIST_TAIL not in snippet:
        snippet = f"{snippet} {templates.SPECIALIST_TAIL}".strip()
    return GenerateResult(
        answer=snippet,
        cited_ids=[pick.chunk_id],
        products=list(pick.products)[:3],
        answer_with_cites=f"{snippet} [{pick.chunk_id}]",
    )


def _extractive_fallback(
    ranked: list[ScoredChunk], store: Store | None = None
) -> GenerateResult:
    return safe_answer(ranked, store, specialist_tail=False)


def generate(
    store: Store,
    understand: UnderstandResult,
    slots: dict[str, SlotValue],
    ranked: list[ScoredChunk],
    history: list[dict] | None = None,
    *,
    skip_llm: bool = False,
) -> tuple[GenerateResult, str, list[ScoredChunk], int]:
    """Return (result, evidence_blocks, used_chunks, llm_calls)."""
    evidence_blocks, used, cards_text = build_evidence(store, ranked)
    if skip_llm or not model_available() or not used:
        fb = _extractive_fallback(used or ranked, store)
        return fb, evidence_blocks, used, 0

    client = get_client()
    if client is None:
        return _extractive_fallback(used, store), evidence_blocks, used, 0

    plain_slots = {k: {"value": v.value, "source": v.source} for k, v in slots.items()}
    messages = [
        {"role": "system", "content": prompts.P6_SYSTEM},
        {
            "role": "user",
            "content": prompts.p6_user(
                slots=plain_slots,
                standalone_query=understand.standalone_query,
                cards_text=cards_text,
                evidence_blocks=evidence_blocks,
                dialog_summary=_dialog_summary(history),
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
            max_json_attempts=int(config.MAX_ATTEMPTS.get("P6", 2)),
        )
        n = max(0, client.calls - before)
        if not data or not data.get("answer"):
            return _extractive_fallback(used, store), evidence_blocks, used, n
        answer = str(data.get("answer") or "")
        cited = [str(x) for x in (data.get("cited_ids") or [])]
        products = [str(x) for x in (data.get("products") or [])]
        return (
            GenerateResult(
                answer=answer,
                cited_ids=cited,
                products=products,
                next_action=str(data.get("next_action") or "none"),
                missing_slot=data.get("missing_slot"),
                answer_with_cites=answer,
            ),
            evidence_blocks,
            used,
            n,
        )
    except Exception:  # noqa: BLE001
        return (
            _extractive_fallback(used, store),
            evidence_blocks,
            used,
            max(0, client.calls - before),
        )
