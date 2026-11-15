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

_EXPLICIT_NAME = re.compile(
    r"(?:меня\s+зовут|мое\s+имя|зовут\s+меня|я\s*[—\-]\s*|фио\s*[:—]\s*)\s*(.+)$",
    re.I,
)

_QUESTION_STEMS = (
    "что",
    "как",
    "где",
    "зачем",
    "почему",
    "можно",
    "есть",
    "чем",
    "когда",
    "какой",
    "какая",
    "какие",
    "сколько",
    "нужно",
    "подскаж",
    "расскаж",
    "интересно",
    "хочу",
    "нужен",
    "нужна",
)

_PRODUCT_TOKENS = (
    "energytool",
    "энерджи",
    "аскуэ",
    "счётчик",
    "счетчик",
    "тариф",
    "фото",
    "показан",
    "платформ",
    "орэм",
    "солнеч",
    "потери",
    "учёт",
    "учет",
    "сервис",
    "продукт",
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


def extract_explicit_name(text: str) -> str | None:
    """Return a name only from explicit contact phrasing."""
    t = (text or "").strip()
    if not t or "?" in t:
        return None
    m = _EXPLICIT_NAME.search(t)
    if not m:
        return None
    name = m.group(1).strip(" .,:;—-")
    parts = name.split()
    if not (1 <= len(parts) <= 4):
        return None
    letters = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\-]", "", name)
    if len(letters) < 2 or any(ch.isdigit() for ch in name):
        return None
    low = name.lower().replace("ё", "е")
    if any(tok in low for tok in _PRODUCT_TOKENS):
        return None
    return name


def looks_like_name(text: str) -> bool:
    """True only for explicit name phrases or short TitleCase FIO without product/Q words."""
    t = (text or "").strip()
    if not t or "?" in t or "…" in t or "..." in t or len(t) > 80:
        return False
    if extract_explicit_name(t):
        return True
    parts = t.split()
    if not (2 <= len(parts) <= 3):
        return False
    low = t.lower().replace("ё", "е")
    first = parts[0].lower().replace("ё", "е")
    if first in _QUESTION_STEMS or any(first.startswith(s) for s in _QUESTION_STEMS):
        return False
    if any(tok in low for tok in _PRODUCT_TOKENS):
        return False
    # Prefer capitalized tokens (FIO-like), allow all-lowercase only if no verbs/stems.
    titled = sum(1 for p in parts if p[:1].isupper())
    if titled < max(2, len(parts) - 0):
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
