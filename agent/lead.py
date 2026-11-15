"""Lead FSM: name → phone → company; optional P8 summary on completion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent import config, prompts, templates
from agent.guard import extract_explicit_name, looks_like_name, looks_like_phone
from agent.llm import get_client, model_available
from agent.slots import slot_state
from agent.schemas import SlotValue

_PHONE_RE = re.compile(
    r"(?:\+?7|8)[\s\-]?(?:\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
)
_FIELDS = ("name", "phone", "company")

# Exact contact-collection prompts (not soft CTA tails).
_LEAD_ASK_MARKERS = (
    "укажите, пожалуйста, фио",
    "как к вам обращаться",
    "оставьте, пожалуйста, номер телефона",
    "номер телефона для связи",
    "назовите, пожалуйста, компанию",
)


@dataclass
class LeadState:
    name: str | None = None
    phone: str | None = None
    company: str | None = None
    next_field: str | None = "name"
    attempts: dict[str, int] | None = None
    done: bool = False

    def __post_init__(self) -> None:
        if self.attempts is None:
            self.attempts = {f: 0 for f in _FIELDS}


def _user_texts(history: list[dict] | None) -> list[str]:
    history = history or []
    out: list[str] = []
    for turn in history:
        if "user" in turn and isinstance(turn.get("user"), str):
            out.append(turn["user"])
            continue
        role = str(turn.get("role") or "").lower()
        if role in ("", "user"):
            content = turn.get("content") or turn.get("text")
            if content:
                out.append(str(content))
    return out


def _assistant_asked_contact(history: list[dict] | None) -> bool:
    history = history or []
    for turn in reversed(history[-6:]):
        text = ""
        if isinstance(turn.get("assistant"), str):
            text = turn["assistant"]
        elif str(turn.get("role") or "").lower() in ("assistant", "bot"):
            text = str(turn.get("content") or "")
        low = text.lower().replace("ё", "е")
        if any(m in low for m in _LEAD_ASK_MARKERS):
            return True
    return False


def lead_active_from_history(history: list[dict] | None, slots: dict[str, SlotValue]) -> bool:
    """Lead is active only when assistant already asked for a contact field.

    Partial heuristic contact slots alone must NOT latch lead mode.
    """
    del slots  # slots alone are not enough to activate lead
    return _assistant_asked_contact(history)


def derive_lead_state(
    history: list[dict] | None,
    slots: dict[str, SlotValue],
    question: str,
) -> LeadState:
    st = LeadState()
    # seed from verified slots only
    for f in _FIELDS:
        sv = slots.get(f)
        if sv and sv.value:
            if f == "name" and not extract_explicit_name(str(sv.value)) and not looks_like_name(str(sv.value)):
                # Ignore polluted name slots from old heuristics.
                continue
            setattr(st, f, str(sv.value))

    # Only parse current user reply + recent explicit contact lines when in lead.
    texts = _user_texts(history)[-3:] + [question]
    for text in texts:
        if not st.phone and looks_like_phone(text):
            m = _PHONE_RE.search(text)
            if m:
                st.phone = m.group(0).strip()
        if not st.name:
            explicit = extract_explicit_name(text)
            if explicit:
                st.name = explicit
            elif looks_like_name(text) and _assistant_asked_contact(history):
                st.name = text.strip()
        if st.name and st.phone and not st.company:
            low = text.lower()
            # Only the latest user reply can fill company (avoid polluting from consent).
            if text is not question:
                continue
            if not looks_like_phone(text) and text.strip() != st.name:
                if len(text.strip()) >= 2 and "?" not in text:
                    if not any(x in low for x in ("снт", "торгов", "производ", "менеджер", "передайте")):
                        if _assistant_asked_contact(history):
                            st.company = text.strip()

    for f in _FIELDS:
        if getattr(st, f):
            continue
        st.next_field = f
        break
    else:
        st.next_field = None
        st.done = True
    return st


def step_lead(
    question: str,
    history: list[dict] | None,
    slots: dict[str, SlotValue],
    *,
    products: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> tuple[str, LeadState, int]:
    """Advance FSM; return (answer, state, llm_calls)."""
    st = derive_lead_state(history, slots, question)
    llm_calls = 0

    # ingest current reply into the pending field
    if st.next_field == "name":
        explicit = extract_explicit_name(question)
        if explicit:
            st.name = explicit
        elif looks_like_name(question):
            st.name = question.strip()
        else:
            st.attempts = st.attempts or {}
            st.attempts["name"] = st.attempts.get("name", 0) + 1
            if st.attempts["name"] < config.LEAD_FIELD_ATTEMPTS:
                return templates.clarify_for("name"), st, 0
    elif st.next_field == "phone":
        m = _PHONE_RE.search(question)
        if m:
            st.phone = m.group(0).strip()
        else:
            st.attempts = st.attempts or {}
            st.attempts["phone"] = st.attempts.get("phone", 0) + 1
            if st.attempts["phone"] < config.LEAD_FIELD_ATTEMPTS:
                return templates.clarify_for("phone"), st, 0
    elif st.next_field == "company" and question.strip() and not looks_like_phone(question):
        st.company = question.strip()

    # recompute next
    for f in _FIELDS:
        if not getattr(st, f):
            st.next_field = f
            st.done = False
            return templates.clarify_for(f), st, 0

    st.done = True
    st.next_field = None
    summary, llm_calls = _maybe_summary(history, slots, question, products or [], chunk_ids or [], st)
    answer = templates.LEAD_DONE
    if summary:
        answer = f"{templates.LEAD_DONE}\n\nРезюме для менеджера:\n{summary}"
    return answer, st, llm_calls


def _maybe_summary(
    history: list[dict] | None,
    slots: dict[str, SlotValue],
    question: str,
    products: list[str],
    chunk_ids: list[str],
    st: LeadState,
) -> tuple[str, int]:
    plain = {k: {"value": v.value, "source": v.source} for k, v in slots.items()}
    plain.update(
        {
            "name": {"value": st.name, "source": "stated"},
            "phone": {"value": st.phone, "source": "stated"},
            "company": {"value": st.company, "source": "stated"},
        }
    )
    # deterministic fallback summary
    fallback = (
        f"Контакты: {st.name}, {st.phone}, {st.company}\n"
        f"Сегмент/объект: {plain.get('segment', {}).get('value') or 'не выяснено'} / "
        f"{plain.get('object_type', {}).get('value') or 'не выяснено'}\n"
        f"Продукты: {', '.join(products) or 'не выяснено'}\n"
        f"Источники: {', '.join(chunk_ids[:5]) or '—'}"
    )
    if not model_available():
        return fallback, 0
    client = get_client()
    if client is None:
        return fallback, 0
    transcript_parts: list[str] = []
    for turn in history or []:
        if "user" in turn:
            transcript_parts.append(f"user: {turn['user']}")
            if turn.get("assistant"):
                transcript_parts.append(f"assistant: {turn['assistant']}")
        else:
            transcript_parts.append(f"{turn.get('role')}: {turn.get('content')}")
    transcript_parts.append(f"user: {question}")
    messages = [
        {"role": "system", "content": prompts.P8_SYSTEM},
        {
            "role": "user",
            "content": prompts.p8_user(
                transcript="\n".join(transcript_parts),
                slots=plain,
                products=products,
            ),
        },
    ]
    try:
        text = client.chat(
            messages, temperature=config.P8_TEMP, max_tokens=config.P8_MAX_TOKENS
        )
        return (text or fallback).strip(), 1
    except Exception:  # noqa: BLE001
        return fallback, 0
