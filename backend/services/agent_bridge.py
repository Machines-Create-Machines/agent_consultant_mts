"""Bridge: FastAPI ↔ agent.pipeline (EnergyTool KB)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_FALLBACK = (
    "Не удалось получить ответ ассистента. Попробуйте переформулировать вопрос "
    "или повторить запрос позже."
)


def cold_start() -> None:
    """Load KB into memory (call once on app startup)."""
    from agent.store import cold_start as agent_cold_start

    agent_cold_start()
    logger.info("agent_bridge: KB cold_start complete")


def _documents_from_citations(citations: list[str]) -> list[dict[str, str]]:
    """Map pipeline citation chunk_ids → UI documents [{id_cr, title}]."""
    docs: list[dict[str, str]] = []
    seen: set[str] = set()
    store = None
    try:
        from agent.store import get_store

        store = get_store()
    except Exception:  # noqa: BLE001
        store = None

    for cid in citations or []:
        key = (cid or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        id_cr = key
        title = key
        if store is not None:
            ch = store.get_chunk(key)
            if ch is not None:
                id_cr = ch.doc_id or ch.source_file or key
                title = (ch.heading or ch.source_file or key).strip() or key
        docs.append({"id_cr": id_cr, "title": title})
    return docs


def answer(question: str, history: list[dict] | None = None) -> dict[str, Any]:
    """Run agent pipeline; never raises. Returns {answer, documents}."""
    history = history or []
    try:
        from agent.pipeline import run

        result = run(question, history)
        text = (result.get("answer") or "").strip()
        if not text:
            text = _FALLBACK
        documents = _documents_from_citations(list(result.get("citations") or []))
        return {"answer": text, "documents": documents}
    except Exception as exc:  # noqa: BLE001 — LLM/fatal → fallback for UI
        logger.exception("agent_bridge.answer failed: %s", exc)
        return {"answer": _FALLBACK, "documents": []}
