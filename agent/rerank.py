"""Always-on platform reranker (Consultant v2)."""

from __future__ import annotations

import json
import ssl
import urllib.request
from typing import Any

from agent import config
from agent.llm import load_dotenv, resolve_rerank_env
from agent.schemas import ScoredChunk, UnderstandResult


def should_rerank(understand: UnderstandResult, ranked: list[ScoredChunk]) -> bool:
    """Rerank only when the intent is ambiguous and the endpoint is configured."""
    if len(ranked) < 2:
        return False
    if resolve_rerank_env() is None:
        return False
    intent = getattr(understand, "consult_intent", "") or ""
    if intent == "catalog":
        return False
    if intent == "product_fact" and getattr(understand, "explicit_product", None):
        return False
    margin = float(ranked[0].score) - float(ranked[1].score)
    top = float(ranked[0].score)
    # RRF scores sit near 0.016; reranker scores are larger after a prior pass.
    ambiguous = margin < 0.006 or top < 0.022
    if intent in ("recommendation", "comparison", "energy_fact"):
        return True if ambiguous or intent == "energy_fact" else ambiguous
    return ambiguous


def _table_to_compact(text: str) -> str:
    """Keep table facts as compact lines instead of dropping markdown tables."""
    import re

    out: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[\-—–\s|]+", stripped):
            continue
        cells = [c.strip() for c in re.split(r"\s{2,}|\|", stripped) if c.strip()]
        if len(cells) >= 2:
            out.append(": ".join(cells[:4]))
        else:
            out.append(stripped)
    return "\n".join(out)


def _doc_text(c: ScoredChunk) -> str:
    products = ", ".join(c.products) if c.products else ""
    head = f"{c.source_file} > {c.heading}".strip(" >")
    prefix = f"{head}\nproduct: {products}\n" if products else f"{head}\n"
    raw = c.text or ""
    # Bundles already carry compact context; flatten residual tables only.
    body = _table_to_compact(raw) if "|" in raw else raw
    body = body[:2200]
    return (prefix + body).strip()


def rerank(
    query: str,
    ranked: list[ScoredChunk],
    *,
    top_n: int | None = None,
) -> tuple[list[ScoredChunk], bool]:
    load_dotenv()
    top_n = int(top_n or config.RERANK_TOP_N)
    env = resolve_rerank_env()
    if not env or not ranked:
        return ranked[:top_n], False

    endpoint = env["endpoint"]
    candidates = ranked[: max(config.RERANK_CANDIDATES, top_n)]
    docs = [_doc_text(c) for c in candidates]
    payload: dict[str, Any] = {
        "model": env["model"],
        "query": query,
        "documents": docs,
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

    if isinstance(body, dict) and "results" in body:
        order = sorted(
            body["results"],
            key=lambda x: -float(x.get("relevance_score") or x.get("score") or 0),
        )
        out: list[ScoredChunk] = []
        for item in order[:top_n]:
            idx = int(item.get("index", 0))
            if 0 <= idx < len(candidates):
                sc = candidates[idx]
                score = float(item.get("relevance_score") or item.get("score") or sc.score)
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
    if isinstance(body, dict) and "scores" in body:
        scores = [float(x) for x in body["scores"]]
        if len(scores) == len(candidates):
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
