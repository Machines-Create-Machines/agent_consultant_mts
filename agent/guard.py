"""Normalize input and detect injection / trivial intents without LLM."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from agent import templates

_INJECTION = re.compile(
    r"(игнорируй\s+(все\s+)?инструкц|ignore\s+(all\s+)?instructions|"
    r"выведи\s+системн|system\s+prompt|reveal\s+prompt|"
    r"забудь\s+(все\s+)?правил|jailbreak|DAN\s+mode)",
    re.I,
)

_GREETING = re.compile(
    r"^\s*(здравствуй(те)?|привет|добрый\s+(день|вечер|утро)|hi|hello|"
    r"доброго\s+времени)\b[!.,\s]*$",
    re.I,
)

_THANKS = re.compile(
    r"^\s*(спасибо|благодарю|thanks|thank\s+you)[!.,\s]*$",
    re.I,
)

_PHONE = re.compile(
    r"(?:\+?7|8)[\s\-]?(?:\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
)

_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002700-\U000027BF"
    "\U0001F000-\U0001F02F"
    "]+",
    flags=re.UNICODE,
)


@dataclass
class GuardResult:
    text: str
    blocked: bool = False
    trivial: str | None = None  # greeting | thanks | injection | None
    answer: str | None = None


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _EMOJI.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_phone(text: str) -> bool:
    return bool(_PHONE.search(text or ""))


def looks_like_name(text: str) -> bool:
    """Heuristic: 2–3 capitalized words, no question mark."""
    t = (text or "").strip()
    if "?" in t or len(t) > 80:
        return False
    parts = t.split()
    if not (2 <= len(parts) <= 4):
        return False
    letters = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\-]", "", t)
    return len(letters) >= 4 and not any(ch.isdigit() for ch in t)


def check(question: str) -> GuardResult:
    text = normalize(question)
    if not text:
        return GuardResult(
            text=text,
            blocked=True,
            trivial="empty",
            answer=templates.GREETING,
        )
    if _INJECTION.search(text):
        return GuardResult(
            text=text,
            blocked=True,
            trivial="injection",
            answer=templates.INJECTION_REFUSAL,
        )
    if _GREETING.match(text):
        return GuardResult(text=text, trivial="greeting", answer=templates.GREETING)
    if _THANKS.match(text):
        return GuardResult(text=text, trivial="thanks", answer=templates.THANKS)
    return GuardResult(text=text)
