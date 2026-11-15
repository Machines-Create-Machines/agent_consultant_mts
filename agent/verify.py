"""Grounding and style validator with rare conditional P7 repair."""

from __future__ import annotations

import re
from typing import Any

from agent import config, prompts
from agent.store import Store
from agent.schemas import GenerateResult, ScoredChunk, SourceBlock

_CITE_RE = re.compile(r"\s*\[[^\[\]]*\.md#[^\[\]]*\]")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_HEADING = re.compile(r"^#{1,6}\s*", re.M)
_MD_BULLET = re.compile(r"^[\-\*]\s+", re.M)
_MD_HR = re.compile(r"^[\s\-—–]{3,}\s*$", re.M)
_DASH_RUN = re.compile(r"(?:\s*[-—–]{3,}\s*)+")
_NUMBER_RE = re.compile(r"\d[\d\s.,]*")
_TECH_SETUP = re.compile(
    r"(настройте|пропишите\s+адрес|modbus|holding\s+register|прошивк)",
    re.I,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\S+")
_DUMP_START = (
    "таблица в разделе",
    "основной вопрос",
    "документ:",
    "раздел:",
    "назначение:",
    "согласно разделу",
)
_SUBSTANTIAL = (
    "unsupported_number",
    "unknown product",
    "catalog_missing",
    "copy_overlap",
    "tech_setup",
    "dump_style",
    "cited_ids вне",
)

_PRODUCT_ALIASES = {
    "выход на оптовый рынок": ("оптов", "орэм"),
    "облачная платформа учёта": ("платформ", "аскуэ"),
    "солнечные электростанции": ("солнеч",),
    "поиск коммерческих потерь": ("потер", "хищен"),
    "распознавание показаний по фотографии": ("фото", "распозна"),
}


def strip_citations(answer: str) -> str:
    return _CITE_RE.sub("", answer or "").strip()


def clean_answer(answer: str) -> str:
    text = answer or ""
    text = strip_citations(text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_HR.sub("", text)
    text = _DASH_RUN.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = text.strip()
    text = re.sub(r"\s*…\s*$", "", text).strip()
    return text


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


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
    if count > limit and result and result[-1] not in ".!?":
        m = re.search(r"^(.*[.!?])\s", result + " ", re.S)
        if m and len(m.group(1)) > 40:
            result = m.group(1).strip()
    return result


def _copy_overlap(answer: str, source: str, n: int = config.MAX_COPY_NGRAM) -> float:
    aw = [w.lower().replace("ё", "е") for w in _WORD_RE.findall(answer or "")]
    sw = [w.lower().replace("ё", "е") for w in _WORD_RE.findall(source or "")]
    if len(aw) < n or len(sw) < n:
        return 0.0
    src = {" ".join(sw[i : i + n]) for i in range(len(sw) - n + 1)}
    hits = 0
    total = 0
    for i in range(len(aw) - n + 1):
        total += 1
        if " ".join(aw[i : i + n]) in src:
            hits += 1
    return hits / float(total) if total else 0.0


def _evidence_blob(evidence: list[ScoredChunk] | list[SourceBlock]) -> tuple[set[str], str]:
    ids: set[str] = set()
    texts: list[str] = []
    for c in evidence:
        cid = getattr(c, "chunk_id", "") or ""
        if cid:
            ids.add(cid)
        texts.append(getattr(c, "text", "") or "")
    return ids, "\n".join(texts)


def catalog_gaps(answer: str, products_out: list[str], store: Store) -> list[str]:
    low = ((answer or "") + " " + " ".join(products_out)).lower().replace("ё", "е")
    missing: list[str] = []
    for p in store.product_names:
        plow = p.lower().replace("ё", "е")
        short = plow.split()[0]
        if plow in low or short in low:
            continue
        if any(a in low for a in _PRODUCT_ALIASES.get(plow, ())):
            continue
        missing.append(p)
    return missing


def map_products(names: list[str], store: Store, violations: list[str]) -> list[str]:
    catalog = {p.lower().replace("ё", "е"): p for p in store.product_names}
    out: list[str] = []
    for p in names:
        pl = p.lower().replace("ё", "е")
        if pl in catalog:
            if catalog[pl] not in out:
                out.append(catalog[pl])
            continue
        hit = next((c for c in store.product_names if pl in c.lower().replace("ё", "е")), None)
        if hit:
            if hit not in out:
                out.append(hit)
        else:
            violations.append(f"unknown product {p}")
    return out


def needs_repair(violations: list[str]) -> bool:
    return any(any(k in v for k in _SUBSTANTIAL) for v in violations)


def verify(
    result: GenerateResult,
    evidence: list[ScoredChunk] | list[SourceBlock],
    store: Store,
    *,
    query_scope: str = "specific",
    consult_intent: str = "",
) -> tuple[GenerateResult, list[str]]:
    """Flag grounding/style issues. Do not silently drop sentences."""
    violations: list[str] = []
    intent = consult_intent or ("catalog" if query_scope == "catalog_overview" else "")
    evidence_ids, evidence_text = _evidence_blob(evidence)

    cited = [store.resolve_id(c) for c in result.cited_ids]
    bad_cites = [c for c in cited if c not in evidence_ids]
    if bad_cites:
        violations.append(f"cited_ids вне evidence: {bad_cites}")
        cited = [c for c in cited if c in evidence_ids] or cited

    answer = clean_answer(result.answer)
    compact_src = re.sub(r"\s+", "", evidence_text)
    for num in _NUMBER_RE.findall(answer):
        compact = re.sub(r"\s+", "", num)
        if len(compact) < 3:
            continue
        if compact not in compact_src:
            violations.append(f"unsupported_number {num.strip()}")

    if _TECH_SETUP.search(answer):
        violations.append("tech_setup")

    low = answer.lower().replace("ё", "е")
    if any(low.startswith(s) for s in _DUMP_START) or "таблица в разделе" in low[:80]:
        violations.append("dump_style")

    overlap = _copy_overlap(answer, evidence_text, n=config.MAX_COPY_NGRAM)
    if overlap >= 0.45:
        violations.append(f"copy_overlap {overlap:.2f}")

    if _word_count(answer) > config.MAX_ANSWER_WORDS:
        answer = _truncate_words(answer)
        violations.append("truncated_words")

    products_out = map_products(list(result.products), store, violations)

    if intent == "catalog" or query_scope == "catalog_overview":
        missing = catalog_gaps(answer, products_out, store)
        for p in store.product_names:
            plow = p.lower().replace("ё", "е")
            short = plow.split()[0]
            blob = low + " " + " ".join(products_out).lower().replace("ё", "е")
            if plow in blob or short in blob or any(
                a in blob for a in _PRODUCT_ALIASES.get(plow, ())
            ):
                if p not in products_out:
                    products_out.append(p)
        if missing:
            violations.append("catalog_missing:" + ",".join(missing))

    fixed = GenerateResult(
        answer=answer,
        cited_ids=cited,
        products=products_out,
        next_action=result.next_action,
        missing_slot=result.missing_slot,
        answer_with_cites=answer,
        mode=result.mode if intent != "catalog" else "answer",
        follow_up=result.follow_up,
    )
    return fixed, violations


def repair(
    result: GenerateResult,
    evidence_blocks: str,
    violations: list[str],
    *,
    consult_intent: str = "energy_fact",
) -> tuple[GenerateResult, int]:
    """One conditional P7 call. Returns original if LLM unavailable."""
    from agent.generate import _parse_sentences
    from agent.llm import get_client, model_available

    if not violations or not model_available():
        return result, 0
    client = get_client()
    if client is None:
        return result, 0
    payload = {
        "answer": result.answer,
        "sentences": [{"text": result.answer, "citations": list(result.cited_ids)}],
        "products": list(result.products),
        "follow_up": result.follow_up,
        "mode": result.mode,
        "consult_intent": consult_intent,
    }
    messages = [
        {"role": "system", "content": prompts.P7_SYSTEM},
        {"role": "user", "content": prompts.p7_user(evidence_blocks, payload, violations)},
    ]
    before = client.calls
    try:
        data = client.chat_json(
            messages,
            temperature=config.P7_TEMP,
            max_tokens=config.P7_MAX_TOKENS,
            stage="P7",
            max_json_attempts=int(config.MAX_ATTEMPTS.get("P7", 1)),
        )
        n = max(0, client.calls - before)
        if not data:
            return result, n
        fixed = _parse_sentences(data)
        if not (fixed.answer or "").strip():
            return result, n
        return fixed, n
    except Exception:  # noqa: BLE001
        return result, max(0, client.calls - before)
