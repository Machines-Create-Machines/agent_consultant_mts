"""Online orchestrator steps 0–15 + degradation ladder."""

from __future__ import annotations

import time
import traceback
from typing import Any

from agent import templates
from agent.clarity import compute_clarity
from agent.coverage import compute_coverage
from agent.decide import decide
from agent.fuse import fuse, normalize_scores_for_coverage
from agent.generate import generate
from agent.guard import check as guard_check
from agent.lead import lead_active_from_history, step_lead
from agent.llm import load_dotenv, model_available, probe_connectivity
from agent.policy import route
from agent.rerank import rerank, should_rerank
from agent.retrieve import intent_k, retrieve
from agent.slots import merge_slots, slots_as_plain, slots_from_history
from agent.store import cold_start, get_store
from agent.types import PipelineResult, ScoredChunk
from agent.understand import understand
from agent.verify import repair, strip_citations, verify


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
        answer=answer or templates.NOT_IN_KB,
        behavior=behavior,
        products=products or [],
        retrieved_ids=retrieved_ids or [],
        retrieved_scores=retrieved_scores or [],
        citations=citations or [],
        llm_calls=llm_calls,
        stages=stages or {},
    ).to_dict()


def _extractive(ranked: list[ScoredChunk], stages: dict[str, Any], llm_calls: int = 0) -> dict[str, Any]:
    if not ranked:
        return _result(templates.NOT_IN_KB, "refuse", llm_calls=llm_calls, stages=stages)
    import re

    text = re.sub(r"\s+", " ", ranked[0].text or "").strip()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    snippet = " ".join(parts[:2]) if parts else text[:400]
    answer = f"{snippet} {templates.SPECIALIST_TAIL}"
    return _result(
        answer,
        "answer",
        products=list(ranked[0].products)[:3],
        retrieved_ids=[c.chunk_id for c in ranked[:5]],
        retrieved_scores=[round(c.score, 4) for c in ranked[:5]],
        citations=[ranked[0].chunk_id],
        llm_calls=llm_calls,
        stages=stages,
    )


def run(
    question: str,
    history: list[dict] | None = None,
    *,
    skip_llm: bool = False,
    skip_rerank: bool = False,
) -> dict[str, Any]:
    """Full agent cycle. Never raises; never returns empty answer."""
    load_dotenv()
    t0 = time.perf_counter()
    stages: dict[str, Any] = {}
    llm_calls = 0
    history = history or []

    try:
        return _run_inner(
            question,
            history,
            skip_llm=skip_llm,
            skip_rerank=skip_rerank,
            stages=stages,
            t0=t0,
        )
    except Exception as exc:  # noqa: BLE001 — degradation step 4
        stages["error"] = f"{type(exc).__name__}: {exc}"
        stages["traceback"] = traceback.format_exc()[-1500:]
        try:
            store = get_store()
            from agent.understand import keyword_fallback

            u = keyword_fallback(question or "")
            branches = retrieve(store, u)
            ranked = fuse(store, branches)
            stages["degrade"] = "extractive"
            stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return _extractive(ranked, stages, llm_calls=llm_calls)
        except Exception:  # noqa: BLE001
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
    # Step 0 cold start
    t = time.perf_counter()
    store = cold_start()
    stages["cold_start_ms"] = round((time.perf_counter() - t) * 1000, 1)
    # Fail fast when PLATFORM_* endpoint is unreachable (avoids multi-minute CSV hangs)
    if not skip_llm:
        t = time.perf_counter()
        stages["llm_reachable"] = probe_connectivity()
        stages["llm_probe_ms"] = round((time.perf_counter() - t) * 1000, 1)
        avail = model_available()
        if not stages["llm_reachable"] or not avail:
            skip_llm = True
            stages["degrade"] = "no_llm"
    else:
        stages["llm_reachable"] = False

    # Step 1 guard
    t = time.perf_counter()
    guard = guard_check(question)
    stages["guard_ms"] = round((time.perf_counter() - t) * 1000, 1)
    if guard.trivial in ("injection", "greeting", "thanks", "empty"):
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
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

    # Degradation ladder: 0 full → 1 no rerank → 2 no LLM → 3 extractive
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
                llm_calls_start=0,
            )
            if out.get("answer"):
                return out
        except Exception as exc:  # noqa: BLE001
            stages[f"degrade_{degrade_level}_err"] = f"{type(exc).__name__}: {exc}"
            continue

    stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
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
    llm_calls_start: int,
) -> dict[str, Any]:
    llm_calls = llm_calls_start

    # Step 2 understand
    t = time.perf_counter()
    u, n = understand(
        question,
        history,
        product_names=store.product_names,
        skip_llm=skip_llm,
    )
    llm_calls += n
    stages["understand_ms"] = round((time.perf_counter() - t) * 1000, 1)
    stages["intent"] = u.intent

    slots = merge_slots(base_slots, u)

    # Step 3 policy
    rt = route(u, lead_active=lead_active)
    if rt.action in ("refuse", "off_topic", "greeting") and rt.answer:
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return _result(
            rt.answer,
            "refuse" if rt.action == "refuse" else rt.action,
            llm_calls=llm_calls,
            stages=stages,
        )

    if rt.action == "lead":
        answer, st, n = step_lead(
            question,
            history,
            slots,
            products=[],
            chunk_ids=[],
        )
        llm_calls += n
        stages["lead_done"] = st.done
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return _result(answer, "lead", llm_calls=llm_calls, stages=stages)

    # Steps 4–6 retrieve + fuse
    t = time.perf_counter()
    branches = retrieve(store, u, slots=slots_as_plain(slots))
    ranked = fuse(store, branches)
    stages["retrieve_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # Step 7 conditional rerank
    used_rerank = False
    t = time.perf_counter()
    if not skip_rerank and should_rerank(u, ranked):
        ranked, used_rerank = rerank(u.standalone_query, ranked, top_n=intent_k(u.intent))
    else:
        ranked = ranked[: intent_k(u.intent)]
    stages["rerank_ms"] = round((time.perf_counter() - t) * 1000, 1)
    stages["used_rerank"] = used_rerank

    scored = ranked if used_rerank else normalize_scores_for_coverage(ranked)

    # Steps 9–10 coverage / clarity / decide
    cov = compute_coverage(
        u, scored, used_rerank=used_rerank, catalog=store.product_names
    )
    clr = compute_clarity(u, slots)
    decision = decide(cov, clr, u, slots, history)
    stages["coverage"] = cov.level
    stages["clarity"] = clr.level
    stages["decision"] = decision.action

    retrieved_ids = [c.chunk_id for c in scored]
    retrieved_scores = [round(c.score, 4) for c in scored]

    if decision.action == "not_in_kb":
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return _result(
            templates.NOT_IN_KB,
            "refuse",
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            llm_calls=llm_calls,
            stages=stages,
        )

    if decision.action == "clarify" and decision.clarify_slot:
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return _result(
            templates.clarify_for(decision.clarify_slot),
            "clarify",
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            llm_calls=llm_calls,
            stages=stages,
        )

    # Steps 11–12 generate + verify (+ P7)
    t = time.perf_counter()
    gen, evidence_blocks, used, n = generate(
        store, u, slots, scored, history, skip_llm=skip_llm
    )
    llm_calls += n
    stages["generate_ms"] = round((time.perf_counter() - t) * 1000, 1)

    gen, violations = verify(gen, used, store)
    if violations and not skip_llm:
        gen, n = repair(gen, evidence_blocks, violations)
        llm_calls += n
        stages["repaired"] = True
        stages["violations"] = violations

    answer = strip_citations(gen.answer)
    if not answer.strip():
        return _extractive(scored, stages, llm_calls=llm_calls)

    # Attach clarify / CTA / partial tails (templates)
    if decision.action == "partial":
        missing = ", ".join(decision.missing_aspects[:2]) or "части деталей"
        answer = templates.partial(answer, missing)
    elif decision.clarify_slot and decision.action == "answer":
        # high + unclear: answer + one clarify
        qline = templates.clarify_for(decision.clarify_slot)
        if qline not in answer:
            answer = f"{answer.rstrip()} {qline}"
    elif decision.action == "answer" and decision.clarity == "clear":
        answer = templates.with_cta(answer)

    # Soft lead offer does not block
    products = gen.products or []
    for c in used:
        for p in c.products:
            if p not in products:
                products.append(p)

    stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    behavior = "partial" if decision.action == "partial" else "answer"
    return _result(
        answer,
        behavior,
        products=products[:5],
        retrieved_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
        citations=list(gen.cited_ids)[:5],
        llm_calls=llm_calls,
        stages=stages,
    )
