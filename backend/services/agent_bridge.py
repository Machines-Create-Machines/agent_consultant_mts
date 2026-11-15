"""Bridge: FastAPI ↔ agent.pipeline (EnergyTool KB)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_FALLBACK = (
    "Не удалось получить ответ ассистента. Попробуйте переформулировать вопрос "
    "или повторить запрос позже."
)

# Safe debug keys only — never prompts / secrets / PII.
_DEBUG_KEYS = (
    "total_ms",
    "route",
    "intent",
    "question_kind",
    "decision",
    "outcome",
    "top_score",
    "used_rerank",
    "degraded_retrieval",
    "degrade_level",
    "cited_ids",
    "doc_titles",
    "coverage",
    "retrieve_ms",
    "rerank_ms",
    "generate_ms",
    "understand_ms",
    "llm_available",
    "cache_hit",
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


def _debug_meta(result: dict[str, Any]) -> dict[str, Any]:
    stages = dict(result.get("stages") or {})
    meta: dict[str, Any] = {}
    for key in _DEBUG_KEYS:
        if key in stages and stages[key] is not None:
            meta[key] = stages[key]
    meta.setdefault("behavior", result.get("behavior"))
    meta.setdefault("products", list(result.get("products") or [])[:5])
    meta.setdefault("citations", list(result.get("citations") or [])[:8])
    meta.setdefault("llm_calls", result.get("llm_calls"))
    # Never include llm_trace / traceback / raw prompts.
    return meta


def answer(
    question: str,
    history: list[dict] | None = None,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """Run agent pipeline; never raises. Returns {answer, documents[, debug]}."""
    history = history or []
    try:
        from agent.pipeline import run

        result = run(question, history)
        text = (result.get("answer") or "").strip()
        if not text:
            text = _FALLBACK
        documents = _documents_from_citations(list(result.get("citations") or []))
        out: dict[str, Any] = {"answer": text, "documents": documents}
        if debug:
            out["debug"] = _debug_meta(result)
        return out
    except Exception as exc:  # noqa: BLE001 — LLM/fatal → fallback for UI
        logger.exception("agent_bridge.answer failed: %s", exc)
        out = {"answer": _FALLBACK, "documents": []}
        if debug:
            out["debug"] = {"error": type(exc).__name__}
        return out
