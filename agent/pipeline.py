"""Consultant v4 orchestrator: planner → doc-first context → writer."""

from __future__ import annotations

import hashlib
import threading
import time
import traceback
from typing import Any

from agent import config, templates
from agent.context import build_context
from agent.fuse import fuse
from agent.generate import catalog_template, generate, synthetic_answer
from agent.guard import check as guard_check
from agent.lead import lead_active_from_history, step_lead
from agent.llm import (
    case_llm_calls,
    get_llm_trace,
    load_dotenv,
    model_available,
    probe_connectivity,
    reset_case_llm_state,
)
from agent.policy import route
from agent.rerank import rerank, should_rerank
from agent.retrieve import intent_k, retrieve
from agent.slots import merge_slots, slots_from_history
from agent.store import cold_start
from agent.schemas import PipelineResult
from agent.understand import understand
from agent.verify import clean_answer, needs_repair, repair, strip_citations, verify

_ANSWER_CACHE: dict[str, dict[str, Any]] = {}
_ANSWER_CACHE_LOCK = threading.Lock()
_KB_VERSION: str | None = None


def _kb_version() -> str:
    global _KB_VERSION
    if _KB_VERSION is not None:
        return _KB_VERSION
    try:
        p = config.BUILD_DIR / "chunks.jsonl"
        _KB_VERSION = str(int(p.stat().st_mtime)) if p.exists() else "0"
    except OSError:
        _KB_VERSION = "0"
    return _KB_VERSION


def _hist_blob(history: list[dict]) -> str:
    parts: list[str] = []
    for t in history:
        role = str(t.get("role") or "")
        content = str(t.get("content") or t.get("text") or "")
        parts.append(f"{role}:{content}")
    return "\n".join(parts)


def _cache_key(question: str, history: list[dict] | None) -> str:
    hist = history or []
    raw = (
        f"{config.PROMPT_VERSION}\n"
        f"{(question or '').strip().lower()}\n"
        f"{_hist_blob(hist)}\n"
        f"{_kb_version()}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _result(
    answer: str,
    behavior: str,
    *,
    products: list[str] | None = None,
    retrieved_ids: list[str] | None = None,
    retrieved_scores: list[float] | None = None,
    citations: list[str] | None = None,
    llm_calls: int = 0,
    stages: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return PipelineResult(
        answer=clean_answer(answer or templates.NOT_IN_KB) or templates.NOT_IN_KB,
        behavior=behavior,
        products=products or [],
        retrieved_ids=retrieved_ids or [],
        retrieved_scores=retrieved_scores or [],
        citations=citations or [],
        llm_calls=llm_calls,
        stages=stages or {},
    ).to_dict()


def _history_products_and_cites(history: list[dict], store) -> tuple[list[str], list[str]]:
    products: list[str] = []
    cites: list[str] = []
    catalog = list(getattr(store, "product_names", None) or config.PRODUCT_NAMES)
    for turn in history or []:
        text = ""
        if isinstance(turn.get("assistant"), str):
            text = turn["assistant"]
        elif str(turn.get("role") or "").lower() in ("assistant", "bot"):
            text = str(turn.get("content") or "")
        if not text:
            continue
        low = text.lower().replace("ё", "е")
        for p in catalog:
            plow = p.lower().replace("ё", "е")
            if plow and plow in low and p not in products:
                products.append(p)
        for cid in getattr(store, "chunk_ids", []) or []:
            if cid in text and cid not in cites:
                cites.append(cid)
    return products, cites


def _consult_gate(
    intent: str,
    top_score: float,
    used_rerank: bool,
    n_blocks: int,
    n_products: int,
) -> str:
    if intent == "catalog" and n_products >= 4:
        return "answer"
    if intent in ("product_fact", "comparison") and n_blocks >= 1:
        return "answer"
    if intent == "recommendation" and n_blocks >= 1:
        return "answer"
    if n_blocks <= 0:
        return "not_in_kb"
    if used_rerank:
        hi, mid = config.GATE_ANSWER, config.GATE_PARTIAL
    else:
        hi, mid = config.GATE_ANSWER_RRF, config.GATE_PARTIAL_RRF
    if top_score >= hi:
        return "answer"
    if top_score >= mid:
        return "partial"
    return "partial" if n_blocks >= 2 else "not_in_kb"


def run(
    question: str,
    history: list[dict] | None = None,
    *,
    skip_llm: bool = False,
    skip_rerank: bool = False,
) -> dict[str, Any]:
    load_dotenv()
    t0 = time.perf_counter()
    reset_case_llm_state(deadline=None)
    stages: dict[str, Any] = {}
    history = history or []

    ck = _cache_key(question, history)
    with _ANSWER_CACHE_LOCK:
        cached = _ANSWER_CACHE.get(ck)
    if cached is not None:
        out = dict(cached)
        out_stages = dict(out.get("stages") or {})
        out_stages["cache_hit"] = True
        out_stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["stages"] = out_stages
        return out

    try:
        out = _run_inner(
            question,
            history,
            skip_llm=skip_llm,
            skip_rerank=skip_rerank,
            stages=stages,
            t0=t0,
        )
        out["llm_calls"] = case_llm_calls()
        st = dict(out.get("stages") or {})
        st["llm_trace"] = get_llm_trace()
        out["stages"] = st
        st_out = out.get("stages") or {}
        if int(out.get("llm_calls") or 0) >= 1 and not st_out.get("degrade"):
            with _ANSWER_CACHE_LOCK:
                if len(_ANSWER_CACHE) < 512:
                    _ANSWER_CACHE[ck] = {
                        k: out[k]
                        for k in (
                            "answer",
                            "behavior",
                            "products",
                            "retrieved_ids",
                            "retrieved_scores",
                            "citations",
                            "llm_calls",
                        )
                        if k in out
                    }
                    _ANSWER_CACHE[ck]["stages"] = {"cached": True}
        return out
    except Exception as exc:  # noqa: BLE001
        stages["error"] = f"{type(exc).__name__}: {exc}"
        stages["traceback"] = traceback.format_exc()[-1500:]
        stages["llm_trace"] = get_llm_trace()
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return _result(templates.NOT_IN_KB, "refuse", llm_calls=0, stages=stages)


def _run_inner(
    question: str,
    history: list[dict],
    *,
    skip_llm: bool,
    skip_rerank: bool,
    stages: dict[str, Any],
    t0: float,
) -> dict[str, Any]:
    t = time.perf_counter()
    store = cold_start()
    stages["cold_start_ms"] = round((time.perf_counter() - t) * 1000, 1)
    if not skip_llm:
        t = time.perf_counter()
        stages["llm_reachable"] = probe_connectivity()
        stages["llm_probe_ms"] = round((time.perf_counter() - t) * 1000, 1)
        stages["llm_available"] = model_available()
    else:
        stages["llm_reachable"] = False
        stages["llm_available"] = False

    t = time.perf_counter()
    guard = guard_check(question)
    stages["guard_ms"] = round((time.perf_counter() - t) * 1000, 1)
    if guard.trivial in ("injection", "greeting", "thanks", "empty"):
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        stages["llm_trace"] = get_llm_trace()
        behavior = {
            "injection": "refuse",
            "greeting": "greeting",
            "thanks": "greeting",
            "empty": "greeting",
        }.get(guard.trivial or "", "refuse")
        return _result(guard.answer or templates.GREETING, behavior, stages=stages)

    q = guard.text
    base_slots = slots_from_history(history)
    lead_active = lead_active_from_history(history, base_slots)

    start_level = 2 if skip_llm else 0
    for degrade_level in range(start_level, 4):
        use_llm = not skip_llm and degrade_level < 2 and model_available()
        use_rerank = not skip_rerank and degrade_level == 0
        stages["degrade_level"] = degrade_level
        try:
            out = _cycle(
                store,
                q,
                history,
                base_slots=base_slots,
                lead_active=lead_active,
                skip_llm=not use_llm,
                skip_rerank=not use_rerank,
                stages=stages,
                t0=t0,
            )
            if out.get("answer"):
                return out
        except Exception as exc:  # noqa: BLE001
            stages[f"degrade_{degrade_level}_err"] = f"{type(exc).__name__}: {exc}"
            continue

    stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    stages["llm_trace"] = get_llm_trace()
    return _result(templates.NOT_IN_KB, "refuse", stages=stages)


def _cycle(
    store,
    question: str,
    history: list[dict],
    *,
    base_slots,
    lead_active: bool,
    skip_llm: bool,
    skip_rerank: bool,
    stages: dict[str, Any],
    t0: float,
) -> dict[str, Any]:
    llm_calls = 0

    t = time.perf_counter()
    u, n = understand(
        question,
        history,
        product_names=store.product_names,
        skip_llm=skip_llm,
    )
    llm_calls += n
    stages["understand_ms"] = round((time.perf_counter() - t) * 1000, 1)
    stages["route"] = getattr(u, "route", u.intent)
    stages["intent"] = u.intent
    stages["consult_intent"] = getattr(u, "consult_intent", "")
    stages["question_kind"] = getattr(u, "question_kind", "fact")

    slots = merge_slots(base_slots, u)

    rt = route(u, lead_active=lead_active, question=question)
    if rt.action in ("refuse", "off_topic", "greeting") and rt.answer:
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        stages["decision"] = rt.action
        stages["llm_trace"] = get_llm_trace()
        return _result(
            rt.answer,
            "refuse" if rt.action == "refuse" else rt.action,
            llm_calls=llm_calls,
            stages=stages,
        )

    if rt.action == "lead":
        products, chunk_ids = _history_products_and_cites(history, store)
        if getattr(u, "explicit_product", None) and u.explicit_product not in products:
            products.insert(0, u.explicit_product)
        answer, st, n = step_lead(
            question,
            history,
            slots,
            products=products,
            chunk_ids=chunk_ids,
        )
        llm_calls += n
        stages["lead_done"] = st.done
        stages["decision"] = "lead"
        stages["outcome"] = "lead"
        stages["cited_ids"] = chunk_ids[:8]
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        stages["llm_trace"] = get_llm_trace()
        return _result(
            answer,
            "lead",
            products=products,
            citations=chunk_ids[:5],
            llm_calls=llm_calls,
            stages=stages,
        )

    intent = getattr(u, "consult_intent", "energy_fact") or "energy_fact"
    scope = "catalog_overview" if intent == "catalog" else "specific"
    stages["query_scope"] = scope
    t = time.perf_counter()
    branches = retrieve(store, u, slots=None)
    ranked = fuse(store, branches, diversify_products=(intent == "catalog"))
    stages["retrieve_ms"] = round((time.perf_counter() - t) * 1000, 1)
    stages["has_dense"] = bool(getattr(store, "embeddings", None) is not None)

    used_rerank = False
    t = time.perf_counter()
    top_n = max(intent_k(u.intent), 12 if intent == "catalog" else intent_k(u.intent))
    do_rerank = not skip_rerank and should_rerank(u, ranked)
    if do_rerank:
        ranked, used_rerank = rerank(u.standalone_query, ranked, top_n=top_n)
    else:
        ranked = ranked[:top_n]
    stages["rerank_ms"] = round((time.perf_counter() - t) * 1000, 1)
    stages["used_rerank"] = used_rerank
    stages["degraded_retrieval"] = not used_rerank

    pack = build_context(store, u, ranked, used_rerank=used_rerank)
    top_score = float(ranked[0].score) if ranked else 0.0
    stages["top_score"] = round(top_score, 4)
    retrieved_ids = list(dict.fromkeys(pack.retrieved_ids or [c.chunk_id for c in ranked]))
    retrieved_scores = [round(c.score, 4) for c in ranked]
    stages["product_coverage"] = len(pack.products)

    decision = _consult_gate(
        intent,
        top_score,
        used_rerank,
        n_blocks=len(pack.blocks),
        n_products=len(pack.products) if intent != "catalog" else len(store.product_names),
    )
    if intent == "catalog" and store.cards:
        decision = "answer"
    stages["decision"] = decision
    stages["coverage"] = (
        "high" if decision == "answer" else "partial" if decision == "partial" else "none"
    )

    if decision == "not_in_kb" and not pack.blocks:
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        stages["llm_trace"] = get_llm_trace()
        return _result(
            templates.NOT_IN_KB,
            "refuse",
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            llm_calls=llm_calls,
            stages=stages,
        )

    mode_hint = "partial" if decision == "partial" else "answer"
    t = time.perf_counter()
    gen, evidence_blocks, used, n = generate(
        store,
        u,
        pack,
        history,
        skip_llm=skip_llm,
        mode_hint=mode_hint,
    )
    llm_calls += n
    stages["generate_ms"] = round((time.perf_counter() - t) * 1000, 1)

    gen, violations = verify(
        gen, used, store, query_scope=scope, consult_intent=intent
    )
    if violations:
        stages["violations"] = violations
    if (
        needs_repair(violations)
        and not skip_llm
        and model_available()
        and int(config.MAX_ATTEMPTS.get("P7", 0) or 0) >= 1
    ):
        repaired, n_rep = repair(
            gen, evidence_blocks, violations, consult_intent=intent
        )
        llm_calls += n_rep
        stages["repaired"] = n_rep > 0
        if n_rep:
            gen, violations2 = verify(
                repaired, used, store, query_scope=scope, consult_intent=intent
            )
            if violations2:
                stages["violations_after_repair"] = violations2
            violations = violations2

    if intent == "catalog":
        from agent.verify import catalog_gaps

        if catalog_gaps(gen.answer, list(gen.products), store):
            tmpl = catalog_template(store)
            gen = tmpl
            stages["catalog_template"] = True

    answer = clean_answer(strip_citations(gen.answer))
    if not answer.strip() or (gen.mode == "not_found" and "нет" in answer.lower()[:80]):
        if gen.mode == "not_found" or not answer.strip():
            fb = synthetic_answer(store, u, pack)
            answer = clean_answer(strip_citations(fb.answer))
            if not answer.strip():
                stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                stages["llm_trace"] = get_llm_trace()
                return _result(
                    templates.NOT_IN_KB,
                    "refuse",
                    retrieved_ids=retrieved_ids,
                    retrieved_scores=retrieved_scores,
                    citations=list(gen.cited_ids)[:5],
                    llm_calls=llm_calls,
                    stages=stages,
                )
            gen = fb

    kind = getattr(u, "question_kind", "fact") or "fact"
    if (
        kind == "recommendation"
        and decision == "answer"
        and templates.CTA not in answer
        and "?" not in answer[-50:]
    ):
        answer = f"{answer.rstrip()} {templates.CTA}".strip()

    products = list(gen.products or [])
    for c in used:
        for p in c.products:
            if p not in products:
                products.append(p)
    if getattr(u, "explicit_product", None) and u.explicit_product not in products:
        products.insert(0, u.explicit_product)
    for p in pack.products:
        if p not in products:
            products.append(p)

    if decision == "partial" or gen.mode == "partial":
        behavior = "partial"
    else:
        behavior = "answer"
    if decision == "answer" and (gen.answer or "").strip():
        behavior = "answer"
    if gen.next_action == "clarify" and decision != "answer":
        behavior = "clarify"

    cited = list(gen.cited_ids)[:5]
    stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    stages["llm_trace"] = get_llm_trace()
    stages["outcome"] = behavior
    stages["cited_ids"] = cited
    stages["doc_titles"] = [
        (store.get_chunk(cid).heading if store.get_chunk(cid) else cid) for cid in cited
    ]
    return _result(
        answer,
        behavior,
        products=products[:5],
        retrieved_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
        citations=cited,
        llm_calls=llm_calls,
        stages=stages,
    )
