"""LLM judge: nugget yes/no and pairwise with position swap. Cached by hash."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from eval.loader import Case
from eval.model_client import ChatClient

PROMPT_VERSION_DEFAULT = "v1"

NUGGET_SYSTEM = (
    "Ты судья качества ответа консультанта МТС Energytool. "
    "Отвечай строго JSON без пояснений."
)

PAIRWISE_SYSTEM = (
    "Ты судья. Сравни два ответа консультанта на один вопрос клиента. "
    "Критерии: фактическая точность по смыслу, полезность, отсутствие запрещённых обещаний, "
    "ясность. Ответь строго одним JSON-объектом без markdown. "
    "reason — максимум 20 слов. Формат: {\"winner\":\"A\"|\"B\"|\"tie\",\"reason\":\"...\"}."
)


def _cache_key(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class JudgeCache:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if enabled:
            root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self.root / f"{key}.json"
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self.root / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_json(text: str) -> dict[str, Any]:
    """Best-effort parse of judge JSON. Tolerates fences and truncated reason strings."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        # Truncated / prose responses: pull fields we need.
        recovered: dict[str, Any] = {}
        mw = re.search(r'"winner"\s*:\s*"?(A|B|tie)"?', text, re.I)
        if mw:
            recovered["winner"] = mw.group(1).lower()
        my = re.search(r'"yes"\s*:\s*(true|false)', text, re.I)
        if my:
            recovered["yes"] = my.group(1).lower() == "true"
        if recovered:
            return recovered
        raise


def judge_nuggets(
    client: ChatClient,
    cache: JudgeCache,
    question: str,
    answer: str,
    nuggets: list[dict[str, Any]],
    prompt_version: str,
) -> dict[str, Any]:
    if not nuggets:
        return {"nugget_recall": 1.0, "hits": [], "details": []}

    details: list[dict[str, Any]] = []
    weighted_hit = 0.0
    weight_sum = 0.0

    for nug in nuggets:
        text = nug["text"]
        w = float(nug.get("w", 1))
        key = _cache_key([prompt_version, "nugget", question, answer, text, str(w)])
        cached = cache.get(key)
        if cached is not None:
            confirmed = bool(cached.get("yes"))
        else:
            messages = [
                {"role": "system", "content": NUGGET_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Вопрос клиента:\n{question}\n\n"
                        f"Ответ бота:\n{answer}\n\n"
                        f"Утверждение (nugget):\n{text}\n\n"
                        'Подтверждается ли утверждение ответом? JSON: {"yes": true|false}.'
                    ),
                },
            ]
            raw = client.chat(messages, temperature=0)
            parsed = _parse_json(raw)
            confirmed = bool(parsed.get("yes"))
            cache.set(key, {"yes": confirmed, "raw": raw})
        details.append({"text": text, "w": w, "yes": confirmed})
        weight_sum += w
        if confirmed:
            weighted_hit += w

    recall = (weighted_hit / weight_sum) if weight_sum else 1.0
    return {
        "nugget_recall": recall,
        "hits": [d["text"] for d in details if d["yes"]],
        "details": details,
    }


def _pairwise_once(
    client: ChatClient,
    cache: JudgeCache,
    question: str,
    answer_a: str,
    answer_b: str,
    prompt_version: str,
    tag: str,
) -> str:
    key = _cache_key([prompt_version, "pair", tag, question, answer_a, answer_b])
    cached = cache.get(key)
    if cached is not None:
        return cached["winner"]
    messages = [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Вопрос:\n{question}\n\n"
                f"Ответ A:\n{answer_a}\n\n"
                f"Ответ B:\n{answer_b}\n"
            ),
        },
    ]
    raw = client.chat(messages, temperature=0)
    try:
        parsed = _parse_json(raw)
        winner = str(parsed.get("winner", "tie")).lower()
    except (json.JSONDecodeError, TypeError, AttributeError):
        winner = "tie"
    if winner not in {"a", "b", "tie"}:
        winner = "tie"
    cache.set(key, {"winner": winner, "raw": raw})
    return winner


def pairwise_score(
    client: ChatClient,
    cache: JudgeCache,
    question: str,
    candidate: str,
    reference: str,
    prompt_version: str,
    tag: str,
) -> float:
    """Position-swapped pairwise. Returns win score in {0, 0.5, 1} for candidate."""
    if not candidate.strip():
        return 0.0
    if not reference.strip():
        return 1.0
    w1 = _pairwise_once(client, cache, question, candidate, reference, prompt_version, tag + "|ab")
    w2 = _pairwise_once(client, cache, question, reference, candidate, prompt_version, tag + "|ba")
    # Map to candidate wins
    score = 0.0
    n = 0
    if w1 == "a":
        score += 1.0
    elif w1 == "tie":
        score += 0.5
    n += 1
    if w2 == "b":
        score += 1.0
    elif w2 == "tie":
        score += 0.5
    n += 1
    return score / n


def soft_violation(
    client: ChatClient,
    cache: JudgeCache,
    answer: str,
    forbidden: list[str],
    prompt_version: str,
) -> list[str]:
    hits: list[str] = []
    for item in forbidden or []:
        key = _cache_key([prompt_version, "forbid", answer, item])
        cached = cache.get(key)
        if cached is not None:
            if cached.get("yes"):
                hits.append(item)
            continue
        messages = [
            {"role": "system", "content": NUGGET_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Ответ бота:\n{answer}\n\n"
                    f"Запрещённый смысл:\n{item}\n\n"
                    'Есть ли в ответе этот запрещённый смысл (прямо или перифразом)? '
                    'JSON: {"yes": true|false}.'
                ),
            },
        ]
        raw = client.chat(messages, temperature=0)
        parsed = _parse_json(raw)
        yes = bool(parsed.get("yes"))
        cache.set(key, {"yes": yes, "raw": raw})
        if yes:
            hits.append(item)
    return hits


def enrich_with_judge(
    cases: list[Case],
    answers_by_id: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    baseline_by_id: dict[str, str],
    client: ChatClient,
    cache: JudgeCache,
    prompt_version: str,
    soft_violation_check: bool,
) -> list[dict[str, Any]]:
    case_by_id = {c.id: c for c in cases}
    for row in rows:
        cid = row["id"]
        case = case_by_id[cid]
        ans = answers_by_id.get(cid) or {}
        answer_text = ans.get("answer") or ""
        if row.get("error"):
            row["nugget_recall"] = 0.0
            row["nugget_hits"] = ""
            row["judge_vs_gold"] = 0.0
            row["judge_vs_baseline"] = None
            row["judge_win_score"] = 0.0
            continue

        try:
            nug = judge_nuggets(
                client,
                cache,
                case.raw["question"],
                answer_text,
                case.raw.get("nuggets") or [],
                prompt_version,
            )
            row["nugget_recall"] = nug["nugget_recall"]
            row["nugget_hits"] = "|".join(nug["hits"])

            vs_gold = pairwise_score(
                client,
                cache,
                case.raw["question"],
                answer_text,
                case.raw.get("gold_answer") or "",
                prompt_version,
                f"gold:{cid}",
            )
            row["judge_vs_gold"] = vs_gold

            base = baseline_by_id.get(cid)
            if base is not None:
                vs_base = pairwise_score(
                    client,
                    cache,
                    case.raw["question"],
                    answer_text,
                    base,
                    prompt_version,
                    f"base:{cid}",
                )
                row["judge_vs_baseline"] = vs_base
                row["judge_win_score"] = 0.5 * (vs_gold + vs_base)
            else:
                row["judge_vs_baseline"] = None
                row["judge_win_score"] = vs_gold

            if soft_violation_check:
                soft = soft_violation(
                    client,
                    cache,
                    answer_text,
                    case.raw.get("forbidden") or [],
                    prompt_version,
                )
                row["soft_violation"] = bool(soft)
                if soft:
                    row["violation"] = True
                    prev = row.get("violation_items") or ""
                    extra = "|".join(soft)
                    row["violation_items"] = "|".join(x for x in [prev, extra] if x)
            else:
                row["soft_violation"] = None
        except Exception as exc:
            # One bad judge call must not kill the run (spec: case isolation).
            row["nugget_recall"] = row.get("nugget_recall")
            row["nugget_hits"] = row.get("nugget_hits") or ""
            row["judge_vs_gold"] = 0.0
            row["judge_vs_baseline"] = None
            row["judge_win_score"] = 0.0
            row["judge_error"] = type(exc).__name__

    return rows


def load_baseline_answers(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    # JSONL of {id, answer} or CSV id,answer
    out: dict[str, str] = {}
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            out[str(obj["id"])] = str(obj.get("answer") or "")
        return out
    # CSV — minimal parser (handles quoted fields)
    import csv
    from io import StringIO

    reader = csv.DictReader(StringIO(text))
    for row in reader:
        if not row:
            continue
        cid = str(row.get("id") or "").strip()
        if cid:
            # map demo numeric ids q00N if needed later
            out[cid] = row.get("answer") or ""
            # also allow q-prefixed
            if cid.isdigit():
                out[f"q{int(cid):03d}"] = row.get("answer") or ""
    return out
