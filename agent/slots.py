"""Merge slot_delta into dialog slots derived from history."""

from __future__ import annotations

import re
from typing import Any

from agent import config
from agent.guard import extract_explicit_name, looks_like_name, looks_like_phone
from agent.schemas import SlotValue, UnderstandResult

_LEAD_ASK_MARKERS = (
    "укажите, пожалуйста, фио",
    "как к вам обращаться",
    "оставьте, пожалуйста, номер телефона",
    "номер телефона для связи",
    "назовите, пожалуйста, компанию",
)

_DECLINED = re.compile(
    r"(не\s+знаю|не\s+скажу|просто\s+скажите|без\s+уточнен|неважно)",
    re.I,
)


def parse_slot_delta(raw: dict[str, Any] | None) -> dict[str, SlotValue]:
    out: dict[str, SlotValue] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if isinstance(v, dict) and "value" in v:
            out[str(k)] = SlotValue(
                value=v.get("value"),
                source=str(v.get("source") or "stated"),
            )
        elif v is not None:
            out[str(k)] = SlotValue(value=v, source="stated")
    return out


def _hist_text(turn: dict) -> str:
    return str(turn.get("content") or turn.get("text") or turn.get("user") or "")


def _hist_role(turn: dict) -> str:
    role = str(turn.get("role") or "").lower()
    if role in ("user", "assistant", "bot"):
        return "user" if role == "user" else "assistant"
    if "user" in turn and "assistant" not in turn:
        return "user"
    if "assistant" in turn:
        return "assistant"
    return role or "user"


def _assistant_asked_contact(history: list[dict] | None) -> bool:
    for turn in reversed((history or [])[-8:]):
        text = ""
        if isinstance(turn.get("assistant"), str):
            text = turn["assistant"]
        elif _hist_role(turn) == "assistant":
            text = _hist_text(turn)
        low = (text or "").lower().replace("ё", "е")
        if any(m in low for m in _LEAD_ASK_MARKERS):
            return True
    return False


def slots_from_history(history: list[dict] | None) -> dict[str, SlotValue]:
    """Best-effort extraction of previously stated slots from history text.

    Contact fields (name/phone) are taken only from explicit patterns or
    bare replies after an assistant contact prompt — never from product Qs.
    """
    history = history or []
    slots: dict[str, SlotValue] = {}
    lead_ctx = _assistant_asked_contact(history)
    for turn in history:
        role = _hist_role(turn)
        text = _hist_text(turn)
        # nested formats: {"user": "...", "assistant": "..."}
        if "user" in turn and isinstance(turn.get("user"), str):
            text = turn["user"]
            role = "user"
        if role != "user" or not text:
            continue
        low = text.lower().replace("ё", "е")
        if any(x in low for x in ("снт", "садов", "дач")):
            slots["object_type"] = SlotValue(value="СНТ", source="stated")
            slots["segment"] = SlotValue(value="СНТ", source="stated")
        if any(x in low for x in ("торгов", "тц")):
            slots["object_type"] = SlotValue(value="ТЦ", source="stated")
        if any(x in low for x in ("производ", "завод")):
            slots["object_type"] = SlotValue(value="производство", source="stated")
        if looks_like_phone(text):
            m = re.search(r"(?:\+?7|8)[\d\s\-()]{10,}", text)
            if m:
                slots["phone"] = SlotValue(value=m.group(0).strip(), source="stated")
        name = extract_explicit_name(text)
        if name and "name" not in slots:
            slots["name"] = SlotValue(value=name, source="stated")
        elif lead_ctx and "name" not in slots and looks_like_name(text):
            slots["name"] = SlotValue(value=text.strip(), source="stated")
        elif (
            lead_ctx
            and slots.get("name")
            and slots.get("phone")
            and "company" not in slots
            and not looks_like_phone(text)
            and "?" not in text
            and len(text.strip()) >= 2
            and not any(x in low for x in ("менеджер", "передайте", "свяжите", "заявк"))
        ):
            slots["company"] = SlotValue(value=text.strip(), source="stated")
    return slots


def merge_slots(
    base: dict[str, SlotValue],
    understand: UnderstandResult,
) -> dict[str, SlotValue]:
    out = dict(base)
    if understand.topic_switch:
        for key in config.THEMATIC_SLOTS:
            out.pop(key, None)
    for key, sv in understand.slot_delta.items():
        prev = out.get(key)
        if prev and prev.value is not None and sv.value is not None and prev.value != sv.value:
            if prev.source == "stated" and sv.source == "stated":
                out[key] = SlotValue(value=sv.value, source="conflict")
                continue
        out[key] = sv
    return out


def slots_as_plain(slots: dict[str, SlotValue]) -> dict[str, Any]:
    return {k: {"value": v.value, "source": v.source} for k, v in slots.items()}


def slot_state(slots: dict[str, SlotValue], key: str) -> str:
    sv = slots.get(key)
    if sv is None or sv.value in (None, ""):
        return "missing"
    return sv.source if sv.source in ("stated", "inferred", "conflict") else "stated"


def declined_slot(slot: str, history: list[dict] | None) -> bool:
    history = history or []
    for turn in history:
        text = _hist_text(turn)
        if "user" in turn and isinstance(turn.get("user"), str):
            text = turn["user"]
        if _DECLINED.search(text or ""):
            return True
    return False


def declined_slots(history: list[dict] | None) -> set[str]:
    """Slots the user declined to answer (any decline phrase → all pending)."""
    if declined_slot("", history):
        return {"object_type", "pain", "scale", "segment", "metering_state", "region", "use_case"}
    return set()


def clarify_streak(history: list[dict] | None) -> int:
    """Count trailing assistant clarify turns from history."""
    history = history or []
    streak = 0
    clarify_markers = (
        "какой у вас объект",
        "что сейчас беспокоит",
        "сколько примерно",
        "уточните",
        "в каком регионе",
    )
    # Walk from end: user replies don't break if assistant asked before
    for turn in reversed(history):
        role = _hist_role(turn)
        text = _hist_text(turn).lower().replace("ё", "е")
        if "assistant" in turn and isinstance(turn.get("assistant"), str):
            role = "assistant"
            text = turn["assistant"].lower().replace("ё", "е")
        if role == "assistant":
            if any(m in text for m in clarify_markers) or text.strip().endswith("?"):
                streak += 1
            else:
                break
        elif role == "user":
            continue
        else:
            break
    return streak
