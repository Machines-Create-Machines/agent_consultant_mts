"""Planner (Consultant v4): route, consult_intent, products, facets."""

from __future__ import annotations

import re
from typing import Any

from agent import config, prompts
from agent.llm import get_client
from agent.schemas import PlannerResult
from agent.products import PRODUCT_ALIASES, extract_products

_OOS = re.compile(
    r"(modbus|настро(ить|йка)|конфигур|прошивк|схем[аы]\s+подключ|"
    r"ремонт|адрес\s+сч[её]тчик|регистр[аы]\s+сч|holding\s+register)",
    re.I,
)
_LEAD = re.compile(
    r"(связ(ать|итесь|аться)|оставь\w*\s+контакт|заявк|менеджер|"
    r"перезвон|оставьте\s+(фио|телефон)|передайте\s+менеджер|"
    r"да[,.]?\s*(передайте|свяжите)|хочу\s+с\s+менеджер)",
    re.I,
)
_OFF = re.compile(
    r"(курс\s+доллар|погод[аы]|рецепт|футбол|анекдот|политик)",
    re.I,
)
_RECOMMEND = re.compile(
    r"(что\s+предлож|подбер|какой\s+продукт\s+подходит|чем\s+поможете|"
    r"снизить\s+счет|снизить\s+счёт|хотим\s+|нужно\s+решени|"
    r"что\s+лучше|посовет)",
    re.I,
)
_COMPARE = re.compile(
    r"(чем\s+отлича|сравн|разниц[аы]\s+между|vs\b|против)",
    re.I,
)
_CATALOG = re.compile(
    r"(расскаж\w*\s+о\s+продукт|какие\s+у\s+вас\s+(продукт|решени)|"
    r"что\s+предлагает\s+energy|что\s+предлагаете|"
    r"какие\s+есть\s+(продукт|решени)|перечень\s+продукт|"
    r"обзор\s+продукт|ваши\s+продукт|ваши\s+решени|"
    r"что\s+у\s+вас\s+есть|каталог\s+продукт|"
    r"какие\s+продукты|какие\s+решения)",
    re.I,
)
_ANAPHORA = re.compile(
    r"^\s*(а\s+)?(скольк|а\s+цена|а\s+стоимость|а\s+срок|а\s+это|"
    r"а\s+там|а\s+как|и\s+сколько|это\s+сколько|а\s+можно|"
    r"подробнее)\b",
    re.I,
)
_GREETING = re.compile(
    r"^\s*(здравствуй(те)?|привет|добрый\s+(день|вечер|утро)|hi|hello)\b",
    re.I,
)
_PRODUCT_FACT = re.compile(
    r"(что\s+такое|как\s+работает|расскаж\w*\s+про|подробнее\s+про|"
    r"для\s+чего|кому\s+подходит|сколько\s+стоит)",
    re.I,
)


def _match_products(text: str) -> list[str]:
    found = extract_products(text or "")
    if found:
        # Stable canonical order
        order = list(config.PRODUCT_NAMES)
        return [p for p in order if p in found]
    low = (text or "").lower().replace("ё", "е")
    hits: list[tuple[int, str]] = []
    for canonical, aliases in PRODUCT_ALIASES.items():
        for a in [canonical] + list(aliases):
            al = a.lower().replace("ё", "е")
            if al and al in low:
                hits.append((len(al), canonical))
                break
    hits.sort(reverse=True)
    out: list[str] = []
    for _, p in hits:
        if p not in out:
            out.append(p)
    return out


def _standalone(question: str, history: list[dict] | None) -> str:
    q = (question or "").strip()
    if not q:
        return q
    if not _ANAPHORA.search(q):
        return q
    prior = [
        str(t.get("content") or "").strip()
        for t in (history or [])[-4:]
        if (t.get("role") or "") == "user" and str(t.get("content") or "").strip()
    ]
    if not prior:
        return q
    return f"{prior[-1]} {q}".strip()


def _consult_intent(standalone: str, products: list[str]) -> str:
    if _CATALOG.search(standalone) and len(products) != 1:
        return "catalog"
    if _COMPARE.search(standalone) or len(products) >= 2:
        return "comparison"
    if _RECOMMEND.search(standalone) and not products:
        return "recommendation"
    if products and (_PRODUCT_FACT.search(standalone) or len(products) == 1):
        if _RECOMMEND.search(standalone):
            return "recommendation"
        return "product_fact"
    if _RECOMMEND.search(standalone):
        return "recommendation"
    return "energy_fact"


def _keyword_plan(question: str, history: list[dict] | None = None) -> PlannerResult:
    q = question or ""
    if _GREETING.match(q.strip()):
        return PlannerResult(route="greeting", standalone_query=q, question_kind="fact")
    if _OOS.search(q):
        return PlannerResult(
            route="out_of_scope",
            standalone_query=_standalone(q, history),
            question_kind="fact",
        )
    if _LEAD.search(q):
        return PlannerResult(
            route="lead",
            standalone_query=_standalone(q, history),
            question_kind="fact",
        )
    if _OFF.search(q):
        return PlannerResult(
            route="off_topic",
            standalone_query=q,
            question_kind="fact",
        )
    standalone = _standalone(q, history)
    products = _match_products(standalone)
    intent = _consult_intent(standalone, products)
    kind = "recommendation" if intent in ("recommendation", "comparison") else "fact"
    scope = "catalog_overview" if intent == "catalog" else "specific"
    return PlannerResult(
        route="consult",
        standalone_query=standalone,
        question_kind=kind,
        explicit_product=products[0] if len(products) == 1 else None,
        explicit_products=products,
        consult_intent=intent,
        query_scope=scope,
    )


def _from_dict(data: dict[str, Any], question: str, history: list[dict] | None) -> PlannerResult:
    fb = _keyword_plan(question, history)
    route = str(data.get("route") or "consult").strip()
    if route not in ("consult", "off_topic", "out_of_scope", "lead", "greeting"):
        route = "consult"
    if fb.route in ("out_of_scope", "off_topic", "lead", "greeting") and route == "consult":
        route = fb.route
    standalone = str(data.get("standalone_query") or question).strip() or question
    products = _match_products(standalone)
    raw_prods = data.get("explicit_products") or []
    if isinstance(raw_prods, list):
        for p in raw_prods:
            ps = str(p)
            if ps in config.PRODUCT_NAMES and ps not in products:
                products.append(ps)
    one = data.get("explicit_product")
    if one:
        one = str(one)
        if one not in config.PRODUCT_NAMES:
            one = products[0] if products else None
        elif one not in products:
            products.insert(0, one)
    intent = str(data.get("consult_intent") or data.get("intent") or "").strip()
    if intent not in ("catalog", "product_fact", "energy_fact", "recommendation", "comparison"):
        intent = fb.consult_intent
    if fb.consult_intent == "catalog" and intent != "catalog":
        intent = "catalog"
    kind = "recommendation" if intent in ("recommendation", "comparison") else "fact"
    facets = data.get("required_facets") or []
    if not isinstance(facets, list):
        facets = []
    missing = data.get("missing_critical_fact")
    missing_s = str(missing).strip() if missing else None
    return PlannerResult(
        route=route,
        standalone_query=standalone,
        question_kind=kind if route == "consult" else "fact",
        explicit_product=products[0] if len(products) == 1 else None,
        explicit_products=products,
        consult_intent=intent if route == "consult" else "energy_fact",
        required_facets=[str(x) for x in facets[:6]],
        missing_critical_fact=missing_s,
        query_scope="catalog_overview" if intent == "catalog" else "specific",
        raw=data,
    )


def fallback_understand(question: str, history: list[dict] | None = None) -> PlannerResult:
    return _keyword_plan(question, history)


keyword_fallback = fallback_understand


def understand(
    question: str,
    history: list[dict] | None = None,
    product_names: list[str] | None = None,
    *,
    use_llm: bool = True,
    skip_llm: bool | None = None,
) -> tuple[PlannerResult, int]:
    history = history or []
    products = product_names or list(config.PRODUCT_NAMES)
    fb = fallback_understand(question, history)
    if skip_llm is not None:
        use_llm = not skip_llm
    # Fast path: obvious catalog / named-product / comparison don't need P5.
    if fb.route != "consult" or fb.consult_intent == "catalog":
        return fb, 0
    if fb.consult_intent == "product_fact" and fb.explicit_product:
        return fb, 0
    if fb.consult_intent == "comparison" and len(fb.explicit_products) >= 2:
        return fb, 0
    if not use_llm:
        return fb, 0
    client = get_client()
    if client is None:
        return fb, 0
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
            max_json_attempts=int(config.MAX_ATTEMPTS.get("P5", 1)),
        )
        n = max(0, client.calls - before)
        if not data:
            return fb, n
        return _from_dict(data, question, history), n
    except Exception:  # noqa: BLE001
        return fb, max(0, client.calls - before)
