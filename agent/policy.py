"""Deterministic pre-search routing (Consultant v2)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent import templates
from agent.guard import looks_like_name, looks_like_phone
from agent.schemas import PlannerResult, UnderstandResult

_PRODUCTISH = re.compile(
    r"(что\s+такое|как\s+работает|сколько\s+стоит|подходит|продукт|"
    r"аскуэ|орэм|солнеч|потер|показан|фото|energytool|энерджи|"
    r"учёт|учет|счётчик|счетчик|тариф|платформ|менеджер\s+не|"
    r"расскажите|подбер|предлож)",
    re.I,
)


@dataclass
class Route:
    action: str  # search | refuse | off_topic | lead | greeting
    answer: str | None = None
    reason: str = ""


def _is_product_question(text: str, understand: UnderstandResult | PlannerResult) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    if looks_like_phone(q) or looks_like_name(q):
        return False
    if len(q.split()) <= 4 and not _PRODUCTISH.search(q) and "?" not in q:
        # short contact-like reply during lead (company name etc.)
        return False
    if getattr(understand, "explicit_product", None):
        return True
    if getattr(understand, "question_kind", "") == "recommendation":
        return True
    return bool(_PRODUCTISH.search(q)) or "?" in q


def route(
    understand: UnderstandResult | PlannerResult,
    *,
    lead_active: bool = False,
    question: str = "",
) -> Route:
    route_name = getattr(understand, "route", None) or getattr(understand, "intent", "") or ""
    if route_name in ("out_of_scope", "out_of_scope_tech") or getattr(understand, "out_of_scope", False):
        return Route(action="refuse", answer=templates.OUT_OF_SCOPE, reason="oos")
    if route_name == "off_topic":
        return Route(action="off_topic", answer=templates.OFF_TOPIC, reason="off_topic")
    if route_name == "greeting":
        return Route(action="greeting", answer=templates.GREETING, reason="greeting")
    if route_name in ("lead", "lead_intent"):
        return Route(action="lead", reason="lead")
    if lead_active:
        # Product questions interrupt lead; contact replies stay in FSM.
        if _is_product_question(question, understand):
            return Route(action="search", reason="lead_interrupt_consult")
        return Route(action="lead", reason="lead_active")
    return Route(action="search", reason="search")
