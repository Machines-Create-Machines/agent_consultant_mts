"""P5 understand call + keyword fallback."""

from __future__ import annotations

import re
from typing import Any

from agent import config, prompts
from agent.llm import get_client
from agent.slots import parse_slot_delta
from agent.schemas import Aspect, UnderstandResult

_OOS = re.compile(
    r"(modbus|настро(ить|йка)|конфигур|прошивк|схем[аы]\s+подключ|"
    r"ремонт|адрес\s+сч[её]тчик|регистр[аы]\s+сч)",
    re.I,
)
_PRICING = re.compile(r"(скольк\w*\s+сто(ит|ят)|цена|стоимость|прайс|тариф)", re.I)
_COMPARE = re.compile(r"(сравн|чем\s+отлича|что\s+лучше|или\s+)", re.I)
_LEAD = re.compile(
    r"(связ(ать|итесь|аться)|оставь\w*\s+контакт|заявк|менеджер|"
    r"перезвон|оставьте\s+(фио|телефон))",
    re.I,
)
_TASK = re.compile(
    r"(сниз(ить|ить)|эконом|потер|хищен|учёт|учет|счета|счет[аы]|"
    r"неплател|солнеч|панел|орэм|оптов)",
    re.I,
)
_PRODUCT = re.compile(
    r"(аскуэ|платформ|солнечн|орэм|потер|фото|показан|energytool|энерджи)",
    re.I,
)


def _keyword_intent(question: str) -> str:
    q = question or ""
    if _OOS.search(q):
        return "out_of_scope_tech"
    if _LEAD.search(q):
        return "lead_intent"
    if _COMPARE.search(q):
        return "comparison"
    if _PRICING.search(q):
        return "pricing"
    if _PRODUCT.search(q):
        return "product_consult"
    if _TASK.search(q):
        return "task_diagnosis"
    return "other"


def _default_aspects(query: str) -> list[Aspect]:
    words = [w for w in re.findall(r"[a-zа-яё0-9]+", (query or "").lower()) if len(w) > 2]
    attr = "цена" if _PRICING.search(query or "") else "механизм"
    return [Aspect(name="основной вопрос", attr=attr, keywords=words[:8] or [query], weight=1)]


def _parse_aspects(raw: Any) -> list[Aspect]:
    out: list[Aspect] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            Aspect(
                name=str(item.get("name") or "aspect"),
                attr=str(item.get("attr") or "механизм"),
                keywords=[str(x) for x in (item.get("keywords") or [])],
                weight=float(item.get("weight") or 1),
            )
        )
    return out


def _from_dict(data: dict[str, Any], question: str) -> UnderstandResult:
    intent = str(data.get("intent") or _keyword_intent(question))
    standalone = str(data.get("standalone_query") or question).strip() or question
    aspects = _parse_aspects(data.get("aspects"))
    if not aspects:
        aspects = _default_aspects(standalone)
    return UnderstandResult(
        intent=intent,
        out_of_scope=bool(data.get("out_of_scope")) or intent == "out_of_scope_tech",
        standalone_query=standalone,
        query_variants=[str(x) for x in (data.get("query_variants") or [])],
        entities=[str(x) for x in (data.get("entities") or [])],
        aspects=aspects,
        slot_delta=parse_slot_delta(data.get("slot_delta") if isinstance(data.get("slot_delta"), dict) else {}),
        topic_switch=bool(data.get("topic_switch")),
        unresolved_reference=bool(data.get("unresolved_reference")),
        lead_signal=bool(data.get("lead_signal")) or intent == "lead_intent",
        raw=data,
    )


def fallback_understand(question: str, history: list[dict] | None = None) -> UnderstandResult:
    intent = _keyword_intent(question)
    prior = [
        str(t.get("content") or "").strip()
        for t in (history or [])[-4:]
        if (t.get("role") or "") == "user" and str(t.get("content") or "").strip()
    ]
    standalone = " ".join(prior + [question]).strip() or question
    return UnderstandResult(
        intent=intent,
        out_of_scope=intent == "out_of_scope_tech",
        standalone_query=standalone,
        query_variants=[],
        entities=[],
        aspects=_default_aspects(question),
        slot_delta={},
        topic_switch=False,
        unresolved_reference=False,
        lead_signal=intent == "lead_intent",
        raw={},
    )


keyword_fallback = fallback_understand


def understand(
    question: str,
    history: list[dict] | None = None,
    product_names: list[str] | None = None,
    *,
    use_llm: bool = True,
    skip_llm: bool | None = None,
) -> tuple[UnderstandResult, int]:
    """Return (result, llm_calls)."""
    history = history or []
    products = product_names or list(config.PRODUCT_NAMES)
    if skip_llm is not None:
        use_llm = not skip_llm
    if not use_llm:
        return fallback_understand(question, history), 0
    client = get_client()
    if client is None:
        return fallback_understand(question, history), 0
    messages = [
        {"role": "system", "content": prompts.P5_SYSTEM},
        {"role": "user", "content": prompts.p5_user(history, products, question)},
    ]
    before = client.calls
    try:
        data = client.chat_json(
            messages,
            temperature=config.P5_TEMP,
            max_tokens=config.P5_MAX_TOKENS,
            stage="P5",
            max_json_attempts=int(config.MAX_ATTEMPTS.get("P5", 2)),
        )
        n = max(0, client.calls - before)
        if not data:
            return fallback_understand(question, history), n
        return _from_dict(data, question), n
    except Exception:  # noqa: BLE001
        return fallback_understand(question, history), max(0, client.calls - before)
