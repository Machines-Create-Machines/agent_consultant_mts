"""Deterministic pre-search routing."""

from __future__ import annotations

from dataclasses import dataclass

from agent import templates
from agent.schemas import UnderstandResult


@dataclass
class Route:
    action: str  # search | refuse | off_topic | lead | greeting
    answer: str | None = None
    reason: str = ""


def route(
    understand: UnderstandResult,
    *,
    lead_active: bool = False,
) -> Route:
    intent = (understand.intent or "").strip()
    if understand.out_of_scope or intent == "out_of_scope_tech":
        return Route(action="refuse", answer=templates.OUT_OF_SCOPE, reason="oos")
    if intent == "off_topic":
        return Route(action="off_topic", answer=templates.OFF_TOPIC, reason="off_topic")
    if intent == "greeting":
        return Route(action="greeting", answer=templates.GREETING, reason="greeting")
    # Do not hijack pricing/product Q&A on a soft lead_signal alone.
    consult_intents = {
        "pricing",
        "product_consult",
        "task_diagnosis",
        "comparison",
        "other",
    }
    if lead_active or intent == "lead_intent":
        return Route(action="lead", reason="lead")
    if understand.lead_signal and intent not in consult_intents:
        return Route(action="lead", reason="lead_signal")
    return Route(action="search", reason="search")
