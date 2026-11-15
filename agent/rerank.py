"""Conditional platform reranker (ON by default when endpoint reachable)."""

from __future__ import annotations

import json
import ssl
import urllib.request
from typing import Any

from agent import config
from agent.llm import load_dotenv, resolve_rerank_env
from agent.schemas import ScoredChunk, UnderstandResult

_FACTUAL = {"pricing", "other", "product_consult"}


def should_rerank(understand: UnderstandResult, ranked: list[ScoredChunk]) -> bool:
    if len(ranked) < 3:
        return False
    if resolve_rerank_env() is None:
        return False
    intent = understand.intent or ""
    top = ranked[0]
    if intent in _FACTUAL:
        b = top.branches or {}
        dense_ok = b.get("dense", 0) >= 0.7 or b.get("cards", 0) >= 0.7
        bm25_ok = b.get("bm25", 0) >= 0.7
        if dense_ok and bm25_ok:
            return False
        if ranked[0].score - ranked[2].score > 0.15:
            return False
    return True


def rerank(
    query: str,
    ranked: list[ScoredChunk],
    *,
    top_n: int | None = None,
) -> tuple[list[ScoredChunk], bool]:
    """Return (reranked_or_original, used_rerank). One batch request."""
    load_dotenv()
    top_n = int(top_n or config.RERANK_TOP_N)
    env = resolve_rerank_env()
    if not env or not ranked:
        return ranked[:top_n], False

    endpoint = env["endpoint"]
    candidates = ranked[: max(12, top_n)]
    docs = [{"id": c.chunk_id, "text": (c.text or "")[:1500]} for c in candidates]
    payload: dict[str, Any] = {
        "model": env["model"],
        "query": query,
        "documents": [d["text"] for d in docs],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {env['api_key']}",
        },
        method="POST",
    )
    ctx = None
    if not env.get("ssl_verify", True):
        ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, timeout=8.0, context=ctx) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return ranked[:top_n], False

    scores: list[float] = []
    if isinstance(body, dict):
        if "results" in body:
            order = sorted(
                body["results"],
                key=lambda x: -float(x.get("relevance_score") or x.get("score") or 0),
            )
            out: list[ScoredChunk] = []
            for item in order[:top_n]:
                idx = int(item.get("index", 0))
                if 0 <= idx < len(candidates):
                    sc = candidates[idx]
                    score = float(
                        item.get("relevance_score") or item.get("score") or sc.score
                    )
                    out.append(
                        ScoredChunk(
                            chunk_id=sc.chunk_id,
                            score=score,
                            text=sc.text,
                            source_file=sc.source_file,
                            heading=sc.heading,
                            n_tokens=sc.n_tokens,
                            products=list(sc.products),
                            branches=dict(sc.branches),
                        )
                    )
            return out or ranked[:top_n], True
        if "scores" in body:
            scores = [float(x) for x in body["scores"]]
    if scores and len(scores) == len(docs):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order[:top_n]:
            sc = candidates[i]
            out.append(
                ScoredChunk(
                    chunk_id=sc.chunk_id,
                    score=scores[i],
                    text=sc.text,
                    source_file=sc.source_file,
                    heading=sc.heading,
                    n_tokens=sc.n_tokens,
                    products=list(sc.products),
                    branches=dict(sc.branches),
                )
            )
        return out, True
    return ranked[:top_n], False
