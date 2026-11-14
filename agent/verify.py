"""Deterministic verifier + optional P7 repair + citation strip."""

from __future__ import annotations

import re
from typing import Any

from agent import config, prompts
from agent.llm import get_client, model_available
from agent.store import Store
from agent.types import GenerateResult, ScoredChunk

_CITE_RE = re.compile(r"\s*\[[\w#.\-/]+\]")
_NUMBER_RE = re.compile(r"\d[\d\s.,]*")
_TECH_SETUP = re.compile(
    r"(настройте|пропишите\s+адрес|modbus|holding\s+register|прошивк)",
    re.I,
)


def strip_citations(answer: str) -> str:
    return _CITE_RE.sub("", answer or "").strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _truncate_words(text: str, limit: int = config.MAX_ANSWER_WORDS) -> str:
    words = re.findall(r"\S+|\s+", text or "")
    count = 0
    out: list[str] = []
    for tok in words:
        if tok.isspace():
            out.append(tok)
            continue
        count += 1
        if count > limit:
            break
        out.append(tok)
    result = "".join(out).strip()
    # trim to sentence boundary if cut mid-way
    if count > limit and result and result[-1] not in ".!?":
        m = re.search(r"^(.*[.!?])\s", result + " ", re.S)
        if m and len(m.group(1)) > 40:
            result = m.group(1).strip()
    return result


def verify(
    result: GenerateResult,
    evidence: list[ScoredChunk],
    store: Store,
) -> tuple[GenerateResult, list[str]]:
    """Return (possibly fixed result still with cites, violations)."""
    violations: list[str] = []
    evidence_ids = {c.chunk_id for c in evidence}
    evidence_text = "\n".join(c.text for c in evidence)
    catalog = {p.lower().replace("ё", "е") for p in store.product_names}

    cited = [store.resolve_id(c) for c in result.cited_ids]
    bad_cites = [c for c in cited if c not in evidence_ids]
    if bad_cites:
        violations.append(f"cited_ids вне evidence: {bad_cites}")
        cited = [c for c in cited if c in evidence_ids]

    # numbers in answer must appear in cited evidence
    cited_text = "\n".join(c.text for c in evidence if c.chunk_id in set(cited)) or evidence_text
    for num in _NUMBER_RE.findall(result.answer or ""):
        compact = re.sub(r"\s+", "", num)
        if len(compact) < 2:
            continue
        if compact not in re.sub(r"\s+", "", cited_text):
            # allow years/percents soft; flag long money-like
            if len(compact) >= 3:
                violations.append(f"число {num.strip()} отсутствует в источниках")

    products_out: list[str] = []
    for p in result.products:
        pl = p.lower().replace("ё", "е")
        if pl in catalog or any(pl in c or c in pl for c in catalog):
            products_out.append(p)
        else:
            # also accept if mentioned in evidence
            if p in evidence_text:
                products_out.append(p)
            else:
                violations.append(f"продукт '{p}' не в каталоге")

    if _TECH_SETUP.search(result.answer or ""):
        violations.append("лексика настройки приборов")

    answer = result.answer or ""
    if _word_count(answer) > config.MAX_ANSWER_WORDS:
        violations.append(f"{_word_count(answer)} слов")
        answer = _truncate_words(answer)

    fixed = GenerateResult(
        answer=answer,
        cited_ids=cited,
        products=products_out,
        next_action=result.next_action,
        missing_slot=result.missing_slot,
        answer_with_cites=answer,
    )
    return fixed, violations


def repair(
    result: GenerateResult,
    evidence_blocks: str,
    violations: list[str],
) -> tuple[GenerateResult, int]:
    if not violations or not model_available():
        return result, 0
    client = get_client()
    if client is None:
        return result, 0
    answer_json: dict[str, Any] = {
        "answer": result.answer,
        "cited_ids": result.cited_ids,
        "products": result.products,
        "next_action": result.next_action,
        "missing_slot": result.missing_slot,
    }
    messages = [
        {"role": "system", "content": prompts.P7_SYSTEM},
        {
            "role": "user",
            "content": prompts.p7_user(
                evidence_blocks=evidence_blocks,
                answer_json=answer_json,
                violations=violations,
            ),
        },
    ]
    try:
        data = client.chat_json(
            messages, temperature=config.P7_TEMP, max_tokens=config.P7_MAX_TOKENS
        )
        if not data or not data.get("answer"):
            return result, 1
        answer = str(data["answer"])
        if _word_count(answer) > config.MAX_ANSWER_WORDS:
            answer = _truncate_words(answer)
        return (
            GenerateResult(
                answer=answer,
                cited_ids=[str(x) for x in (data.get("cited_ids") or result.cited_ids)],
                products=[str(x) for x in (data.get("products") or result.products)],
                next_action=str(data.get("next_action") or result.next_action),
                missing_slot=data.get("missing_slot", result.missing_slot),
                answer_with_cites=answer,
            ),
            1,
        )
    except Exception:  # noqa: BLE001
        return result, getattr(client, "calls", 0) or 0
