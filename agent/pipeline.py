"""Online orchestrator steps 0–15 + degradation ladder."""

from __future__ import annotations

import hashlib
import re
import threading
import time
import traceback
from typing import Any

from agent import config, templates
from agent.clarity import compute_clarity
from agent.coverage import compute_coverage
from agent.decide import decide
from agent.fuse import fuse, normalize_scores_for_coverage
from agent.generate import generate, safe_answer
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
from agent.slots import merge_slots, slots_as_plain, slots_from_history
from agent.store import cold_start, get_store
from agent.schemas import PipelineResult, ScoredChunk
from agent.understand import understand
from agent.verify import clean_answer, repair, strip_citations, verify

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
    raw = f"{(question or '').strip().lower()}\n{_hist_blob(hist)}\n{_kb_version()}"
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


def _extractive(ranked: list[ScoredChunk], stages: dict[str, Any], llm_calls: int = 0) -> dict[str, Any]:
    try:
        store = get_store()
    except Exception:  # noqa: BLE001
        store = None
    fb = safe_answer(ranked, store, specialist_tail=True)
    return _result(
        fb.answer,
        "answer",
        products=list(fb.products)[:3],
        retrieved_ids=[c.chunk_id for c in ranked[:5]],
        retrieved_scores=[round(c.score, 4) for c in ranked[:5]],
        citations=list(fb.cited_ids)[:5],
        llm_calls=llm_calls,
        stages=stages,
    )


def _ends_with_question(answer: str) -> bool:
    """True if a '?' appears in the last two sentences."""
    text = (answer or "").strip()
    if not text:
        return False
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    tail = parts[-2:] if len(parts) >= 2 else parts
    return any("?" in p for p in tail)


def _asked_aspects_missing(question: str, missing: list[str]) -> list[str]:
    """Aspects that were explicitly asked in the user question."""
    q = (question or "").lower().replace("ё", "е")
    cues = {
        "цена": ("цена", "стоимость", "скольк", "прайс", "тариф", "руб"),
        "срок": ("срок", "быстр", "когда", "дней", "внедр"),
        "гарантия": ("гарант",),
    }
    out: list[str] = []
    for aspect in missing:
        key = (aspect or "").lower()
        terms = cues.get(key)
        if terms and any(t in q for t in terms):
            out.append(aspect)
        elif key and key in q:
            out.append(aspect)
    return out


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
    reset_case_llm_state(deadline=None)
    stages: dict[str, Any] = {}
    llm_calls = 0
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
        # Authoritative per-case counter (TLS), not singleton ChatClient.calls.
        out["llm_calls"] = case_llm_calls()
        st = dict(out.get("stages") or {})
        st["llm_trace"] = get_llm_trace()
        out["stages"] = st
        # Never cache extractive / zero-LLM answers (poison later rows).
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
    except Exception as exc:  # noqa: BLE001 — degradation step 4
        stages["error"] = f"{type(exc).__name__}: {exc}"
        stages["traceback"] = traceback.format_exc()[-1500:]
        stages["llm_trace"] = get_llm_trace()
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
    # Probe is informational only — never latch skip_llm for the whole run.
    if not skip_llm:
        t = time.perf_counter()
        stages["llm_reachable"] = probe_connectivity()
        stages["llm_probe_ms"] = round((time.perf_counter() - t) * 1000, 1)
        stages["llm_available"] = model_available()
    else:
        stages["llm_reachable"] = False
        stages["llm_available"] = False

    # Step 1 guard
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

    # Degradation: 0 full → 1 no rerank → 2 no LLM → 3 extractive
    # Never force level-2 solely because probe failed once.
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
        stages["llm_trace"] = get_llm_trace()
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
        stages["llm_trace"] = get_llm_trace()
        return _result(answer, "lead", llm_calls=llm_calls, stages=stages)

    # Steps 4–6 retrieve + fuse
    t = time.perf_counter()
    branches = retrieve(store, u, slots=slots_as_plain(slots))
    ranked = fuse(store, branches)
    stages["retrieve_ms"] = round((time.perf_counter() - t) * 1000, 1)
    stages["has_dense"] = bool(getattr(store, "embeddings", None) is not None)

    # Step 7 conditional rerank
    used_rerank = False
    t = time.perf_counter()
    do_rerank = not skip_rerank and should_rerank(u, ranked)
    if do_rerank:
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

    thr = config.THRESHOLDS_RERANK if used_rerank else config.THRESHOLDS_NO_RERANK
    has_hit = bool(scored) and scored[0].score >= thr.T_floor
    empty_retrieve = not scored
    clarify_tail = None

    # Asymmetric refusal: never refuse on keyword coverage alone when we have a hit.
    # Refuse only on truly empty retrieve (or policy OOS already handled).
    if decision.action == "not_in_kb":
        if has_hit:
            decision.action = "answer"
            decision.reason = "answer_first_has_hit"
            stages["decision"] = "answer"
        elif empty_retrieve:
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
        else:
            # Weak coverage but some chunks — still answer-first
            decision.action = "answer"
            decision.reason = "answer_first_weak_cov"
            stages["decision"] = "answer"

    if decision.action == "clarify" and decision.clarify_slot and not has_hit:
        stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        stages["llm_trace"] = get_llm_trace()
        return _result(
            templates.clarify_for(decision.clarify_slot),
            "clarify",
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            llm_calls=llm_calls,
            stages=stages,
        )
    if decision.action == "clarify" and decision.clarify_slot and has_hit:
        clarify_tail = decision.clarify_slot
        decision.action = "answer"
        stages["decision"] = "answer"

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
    elif violations:
        stages["violations"] = violations

    answer = clean_answer(strip_citations(gen.answer))
    if not answer.strip():
        stages["llm_trace"] = get_llm_trace()
        return _extractive(scored, stages, llm_calls=llm_calls)

    # Asked-but-missing aspects: one short honesty line (no generic partial tail).
    if decision.missing_aspects:
        asked_miss = _asked_aspects_missing(question, decision.missing_aspects)
        if asked_miss and "в базе" not in answer.lower():
            miss = ", ".join(asked_miss[:2])
            answer = f"{answer.rstrip()} По {miss} в материалах точных данных нет."

    # At most one clarifying question if model asked and last sentences have no '?'.
    wants_clarify = (gen.next_action or "").lower() == "clarify"
    if decision.action != "partial" and wants_clarify and not _ends_with_question(answer):
        slot = (gen.missing_slot or clarify_tail or decision.clarify_slot or "").strip()
        filled = {
            k
            for k, v in slots.items()
            if getattr(v, "value", None) not in (None, "", [])
            and getattr(v, "source", "") != "missing"
        }
        if slot and slot in filled:
            slot = ""
        thematic_generic = {"pain", "object_type"}
        if slot and slot in thematic_generic and slot != (gen.missing_slot or "").strip():
            slot = ""
        if slot:
            qline = templates.clarify_for(slot)
            if qline and qline not in answer:
                answer = f"{answer.rstrip()} {qline}"
    answer = clean_answer(answer)

    products = gen.products or []
    for c in used:
        for p in c.products:
            if p not in products:
                products.append(p)

    # Honest behavior label for eval/debug (CSV still only uses answer).
    if decision.action == "partial":
        behavior = "partial"
    elif wants_clarify or clarify_tail:
        behavior = "clarify" if wants_clarify and not has_hit else "answer"
    else:
        behavior = "answer"

    stages["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    stages["llm_trace"] = get_llm_trace()
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
