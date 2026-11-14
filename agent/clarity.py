"""Clarity / blocking slots / worth_asking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent import config
from agent.slots import clarify_streak, declined_slots
from agent.types import SlotValue, UnderstandResult

VALUE = {"stated": 1.0, "inferred": 0.6, "conflict": 0.0, "missing": 0.0}

BLOCKING: dict[str, list[tuple[str, float]]] = {
    "pricing": [],
    "product_consult": [("pain", 2), ("object_type", 2)],
    "task_diagnosis": [("object_type", 2)],
    "comparison": [("use_case", 2)],
    "lead_intent": [("name", 1), ("phone", 1), ("company", 1)],
    "other": [],
    "greeting": [],
    "off_topic": [],
    "out_of_scope_tech": [],
}


@dataclass
class ClarityResult:
    level: str  # clear | partial | unclear
    raw: float
    missing: list[str]
    blocked: bool


def _state(slots: dict[str, SlotValue], key: str) -> str:
    if key not in slots:
        return "missing"
    return slots[key].source if slots[key].source in VALUE else "stated"


def compute_clarity(
    understand: UnderstandResult,
    slots: dict[str, SlotValue],
) -> ClarityResult:
    intent = understand.intent or "other"
    blocking = BLOCKING.get(intent, [])
    if not blocking:
        return ClarityResult(level="clear", raw=1.0, missing=[], blocked=False)

    wsum = sum(w for _, w in blocking) or 1.0
    raw = 0.0
    missing: list[str] = []
    for key, w in blocking:
        st = _state(slots, key)
        raw += w * VALUE[st]
        if VALUE[st] < 1.0:
            missing.append(key)
    clarity_raw = raw / wsum

    blocked = (
        understand.unresolved_reference
        or any(_state(slots, k) == "conflict" for k, _ in blocking)
        or (understand.topic_switch and _state(slots, "object_type") == "missing")
    )

    if clarity_raw >= 0.99 and not blocked:
        level = "clear"
    elif clarity_raw >= 0.50 and not blocked:
        level = "partial"
    else:
        level = "unclear"
    if blocked and level == "clear":
        level = "unclear"
    return ClarityResult(level=level, raw=clarity_raw, missing=missing, blocked=blocked)


def worth_asking(
    intent: str,
    slots: dict[str, SlotValue],
    history: list[dict] | None,
    coverage_level: str,
) -> str | None:
    if coverage_level == "none":
        return None
    if clarify_streak(history) >= config.CLARIFY_STREAK_MAX:
        return None
    declined = declined_slots(history)
    blocking = BLOCKING.get(intent, [])
    missing = [
        (k, w)
        for k, w in blocking
        if VALUE[_state(slots, k)] < 1.0 and k not in declined
    ]
    if not missing:
        return None
    missing.sort(key=lambda x: -x[1])
    return missing[0][0]
