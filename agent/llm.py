"""OpenAI-compatible client with JSON parse + retry (LLM_*/OPENAI_* first)."""

from __future__ import annotations

import json
import os
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from agent import config

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_ENV_LOADED = False
_CIRCUIT_OPEN_UNTIL = 0.0
_CIRCUIT_REASON = ""
_LLM_SEM = threading.Semaphore(int(getattr(config, "LLM_HTTP_CONCURRENCY", 2)))
_TLS = threading.local()


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency). Does not override existing env."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = path or (config.ROOT / ".env")
    if not env_path.exists():
        _ENV_LOADED = True
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass
    _ENV_LOADED = True


def _ssl_verify_flag() -> bool:
    raw = (
        os.environ.get("EMB_SSL_VERIFY")
        or os.environ.get("LLM_SSL_VERIFY")
        or os.environ.get("PLATFORM_SSL_VERIFY")
        or "true"
    )
    return raw.lower() not in ("0", "false", "no")


def resolve_model_env() -> dict[str, str]:
    """Chat LLM: platform contract LLM_*/OPENAI_* first, then local PLATFORM_*."""
    load_dotenv()
    key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("PLATFORM_API_KEY")
        or ""
    )
    base = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("PLATFORM_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = (
        os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("GEN_MODEL")
        or os.environ.get("PLATFORM_MODEL")
        or ""
    )
    return {
        "api_key": key,
        "base_url": base.rstrip("/"),
        "model": model,
        "ssl_verify": _ssl_verify_flag(),
        "source": "llm",
    }


def resolve_embed_env() -> dict[str, str]:
    """Embeddings endpoint + model.

    Priority:
    1. EMB_URL + EMB_KEY — local (e.g. Timeweb) for index build / offline
    2. Same host as chat (LLM_*/OPENAI_*/PLATFORM_*) — platform auto-eval contract

    Default model: EMB_MODEL, else BAAI/bge-m3 (reglament). Local aliases tried in embed_query.
    """
    load_dotenv()
    emb_url = (os.environ.get("EMB_URL") or os.environ.get("EMBEDDINGS_BASE_URL") or "").strip()
    emb_key = (os.environ.get("EMB_KEY") or os.environ.get("EMBEDDINGS_API_KEY") or "").strip()
    if emb_url and emb_key:
        base = emb_url.rstrip("/")
        if base.endswith("/embeddings"):
            base = base[: -len("/embeddings")]
        # Timeweb catalog id; override with EMB_MODEL if needed.
        model = (os.environ.get("EMB_MODEL") or "timeweb/bge-m3").strip() or "timeweb/bge-m3"
        return {
            "api_key": emb_key,
            "base_url": base,
            "model": model,
            "ssl_verify": _ssl_verify_flag(),
            "source": "emb",
        }
    chat = resolve_model_env()
    model = (os.environ.get("EMB_MODEL") or "BAAI/bge-m3").strip() or "BAAI/bge-m3"
    return {
        "api_key": chat["api_key"],
        "base_url": chat["base_url"],
        "model": model,
        "ssl_verify": chat["ssl_verify"],
        "source": "llm",
    }


def resolve_rerank_env() -> dict[str, str] | None:
    """Rerank endpoint. Platform: same LLM host /rerank. Local: EMB_URL or RERANK_URL."""
    load_dotenv()
    flag = (os.environ.get("PLATFORM_RERANK") or "auto").lower()
    if flag in ("0", "false", "no", "off"):
        return None
    explicit = (
        os.environ.get("PLATFORM_RERANK_URL")
        or os.environ.get("RERANK_URL")
        or ""
    ).strip()
    emb = resolve_embed_env()
    # Local Timeweb reranker id; platform reglament: Qwen3 Reranker 8B.
    default_rerank = (
        "timeweb/bge-reranker-v2-m3"
        if emb.get("source") == "emb"
        else "qwen3-reranker-8b"
    )
    model = (
        os.environ.get("PLATFORM_RERANK_MODEL")
        or os.environ.get("RERANK_MODEL")
        or default_rerank
    ).strip()
    if explicit:
        return {
            "api_key": (
                os.environ.get("EMB_KEY")
                or os.environ.get("RERANK_API_KEY")
                or resolve_model_env()["api_key"]
            ),
            "base_url": explicit.rstrip("/"),
            "endpoint": explicit.rstrip("/"),
            "model": model,
            "ssl_verify": _ssl_verify_flag(),
            "source": "explicit",
        }
    if not emb["api_key"]:
        return None
    # If dedicated emb host — prefer it for rerank too (Timeweb).
    if emb.get("source") == "emb":
        endpoint = f"{emb['base_url']}/rerank"
        return {
            "api_key": emb["api_key"],
            "base_url": emb["base_url"],
            "endpoint": endpoint,
            "model": model,
            "ssl_verify": emb["ssl_verify"],
            "source": "emb",
        }
    chat = resolve_model_env()
    if not chat["api_key"]:
        return None
    return {
        "api_key": chat["api_key"],
        "base_url": chat["base_url"],
        "endpoint": f"{chat['base_url']}/rerank",
        "model": model,
        "ssl_verify": chat["ssl_verify"],
        "source": "llm",
    }


def embed_model_candidates(preferred: str | None = None) -> list[str]:
    """Local Timeweb ids + short name + platform reglament BAAI/bge-m3."""
    pref = (preferred or "").strip()
    out: list[str] = []
    for m in (pref, "timeweb/bge-m3", "bge-m3", "BAAI/bge-m3"):
        if m and m not in out:
            out.append(m)
    return out


def _circuit_open() -> bool:
    return time.monotonic() < _CIRCUIT_OPEN_UNTIL


def _trip_circuit(reason: str) -> None:
    global _CIRCUIT_OPEN_UNTIL, _CIRCUIT_REASON
    _CIRCUIT_OPEN_UNTIL = time.monotonic() + float(config.LLM_CIRCUIT_COOLDOWN_SEC)
    _CIRCUIT_REASON = reason


def _clear_circuit() -> None:
    global _CIRCUIT_OPEN_UNTIL, _CIRCUIT_REASON
    _CIRCUIT_OPEN_UNTIL = 0.0
    _CIRCUIT_REASON = ""


def model_available() -> bool:
    if _circuit_open():
        return False
    env = resolve_model_env()
    return bool(env["api_key"] and env["model"])


def reset_case_llm_state(*, deadline: float | None = None) -> None:
    """Per-case: clear trace and set wall-clock deadline (monotonic)."""
    _TLS.llm_trace = []
    _TLS.deadline = deadline
    _TLS.case_calls = 0


def get_llm_trace() -> list[dict[str, Any]]:
    return list(getattr(_TLS, "llm_trace", []) or [])


def case_llm_calls() -> int:
    return int(getattr(_TLS, "case_calls", 0) or 0)


def _trace(stage: str, attempt: int, reason: str) -> None:
    buf = getattr(_TLS, "llm_trace", None)
    if buf is None:
        buf = []
        _TLS.llm_trace = buf
    buf.append({"stage": stage, "attempt": attempt, "reason": reason})


def time_left() -> float | None:
    deadline = getattr(_TLS, "deadline", None)
    if deadline is None:
        return None
    return float(deadline) - time.monotonic()


def effective_timeout(default: float | None = None, *, reserve: float = 0.0) -> float:
    """Urllib timeout; optional reserve only if a case deadline was set."""
    base = float(default if default is not None else config.LLM_TIMEOUT_SEC)
    left = time_left()
    if left is None:
        return base
    avail = max(0.5, left - reserve)
    return max(0.5, min(base, avail))


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(cleaned)
    if not m:
        return None
    blob = m.group(0)
    blob = blob.replace("\n", " ")
    blob = re.sub(r",\s*}", "}", blob)
    blob = re.sub(r",\s*]", "]", blob)
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _is_hard_connect_error(exc: BaseException) -> bool:
    """True only for refused/DNS/unreachable — not read timeouts."""
    text = str(exc).lower()
    hard_tokens = (
        "connection refused",
        "name or service not known",
        "nodename nor servname",
        "getaddrinfo failed",
        "failed to resolve",
        "network is unreachable",
        "no route to host",
    )
    if any(token in text for token in hard_tokens):
        return True
    if isinstance(exc, ConnectionRefusedError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return False
        reason_text = str(reason or "").lower()
        return any(token in reason_text for token in hard_tokens)
    return False


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


@contextmanager
def _http_slot() -> Iterator[None]:
    _LLM_SEM.acquire()
    try:
        yield
    finally:
        _LLM_SEM.release()


class ChatClient:
    def __init__(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self.temperature = (
            float(temperature) if temperature is not None else _env_float("TEMPERATURE", 0.0)
        )
        self.max_tokens = (
            int(max_tokens) if max_tokens is not None else _env_int("MAX_TOKENS", 512)
        )
        self.timeout_sec = timeout_sec if timeout_sec is not None else config.LLM_TIMEOUT_SEC
        self.env = resolve_model_env()
        self.calls = 0
        if not self.env["api_key"] or not self.env["model"]:
            raise RuntimeError(
                "LLM env missing. Set LLM_API_KEY + LLM_MODEL "
                "(or OPENAI_* / PLATFORM_* equivalents)."
            )

    def _timeout(self) -> float:
        return effective_timeout(float(self.timeout_sec))

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str = "chat",
        timeout_cap: float | None = None,
        budget_reserve: float | None = None,
    ) -> str:
        if _circuit_open():
            raise RuntimeError(f"LLM circuit open: {_CIRCUIT_REASON or 'unavailable'}")
        # timeout_cap / budget_reserve kept for API compat; unused (no case budget).
        _ = timeout_cap, budget_reserve

        payload: dict[str, Any] = {
            "model": self.env["model"],
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.env['base_url']}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.env['api_key']}",
        }
        ctx = None
        if not self.env.get("ssl_verify", True):
            ctx = ssl._create_unverified_context()
        last_err: Exception | None = None
        http_attempts = min(config.LLM_RETRIES + 1, 2)
        for attempt in range(http_attempts):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with _http_slot():
                    with urllib.request.urlopen(
                        req,
                        timeout=self._timeout(),
                        context=ctx,
                    ) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                self.calls += 1
                _TLS.case_calls = int(getattr(_TLS, "case_calls", 0) or 0) + 1
                _clear_circuit()
                _trace(stage, attempt + 1, "ok")
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_err = RuntimeError(f"LLM HTTP {exc.code}: {detail}")
                _trace(stage, attempt + 1, f"http_{exc.code}")
                if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < http_attempts:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                _trace(stage, attempt + 1, type(exc).__name__)
                if _is_hard_connect_error(exc):
                    _trip_circuit(str(exc)[:200])
                    break
                if attempt + 1 < http_attempts:
                    time.sleep(0.4 * (attempt + 1))
        raise RuntimeError(str(last_err) if last_err else "LLM failed")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str = "chat",
        max_json_attempts: int = 1,
        timeout_cap: float | None = None,
        budget_reserve: float | None = None,
    ) -> dict[str, Any] | None:
        attempts = max(1, int(max_json_attempts))
        last: dict[str, Any] | None = None
        for i in range(attempts):
            text = self.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stage=stage,
                timeout_cap=timeout_cap,
                budget_reserve=budget_reserve,
            )
            last = extract_json(text)
            if last is not None:
                return last
            _trace(stage, i + 1, "json_invalid")
        return last


_CLIENT: ChatClient | None = None
_CLIENT_LOCK = threading.Lock()

_EMB_CACHE: dict[str, list[float]] = {}
_EMB_LAST_ERROR = ""
_EMB_CACHE_LOCK = threading.Lock()


def last_embed_error() -> str:
    return _EMB_LAST_ERROR


def _l2_normalize(vec: list[float]) -> list[float]:
    import math

    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def embed_query(text: str, *, model: str | None = None, timeout_sec: float | None = None) -> list[float] | None:
    """Query embedding. Uses EMB_URL/EMB_KEY locally; platform LLM_* + BAAI/bge-m3 in prod."""
    global _EMB_LAST_ERROR
    key = (text or "").strip()
    if not key:
        return None
    with _EMB_CACHE_LOCK:
        cached = _EMB_CACHE.get(key)
        if cached is not None:
            return cached
    env = resolve_embed_env()
    if not env["api_key"]:
        _EMB_LAST_ERROR = "no api key"
        return None
    url = f"{env['base_url']}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {env['api_key']}",
    }
    ctx = None
    if not env.get("ssl_verify", True):
        ctx = ssl._create_unverified_context()
    timeout = float(
        timeout_sec if timeout_sec is not None else effective_timeout(float(config.LLM_TIMEOUT_SEC))
    )
    errors: list[str] = []
    # Prefer pinned model; only fall back on 401/403/404 model errors.
    preferred = (model or env.get("model") or "").strip()
    models = [preferred] if preferred else embed_model_candidates(None)[:1]
    fallback_tried = False
    for mid in models:
        payload = json.dumps(
            {"model": mid, "input": key}, ensure_ascii=False
        ).encode("utf-8")
        for attempt in range(2):
            req = urllib.request.Request(
                url, data=payload, headers=headers, method="POST"
            )
            try:
                with _http_slot():
                    with urllib.request.urlopen(
                        req, timeout=timeout, context=ctx
                    ) as resp:
                        raw = resp.read().decode("utf-8")
                body = json.loads(raw)
                vec: list[float] | None = None
                if isinstance(body, dict):
                    data = body.get("data")
                    if isinstance(data, list) and data:
                        emb = (
                            data[0].get("embedding")
                            if isinstance(data[0], dict)
                            else None
                        )
                        if isinstance(emb, list) and emb:
                            vec = [float(x) for x in emb]
                    elif isinstance(body.get("embedding"), list):
                        vec = [float(x) for x in body["embedding"]]
                if not vec:
                    errors.append(f"model={mid} empty embedding")
                    break
                vec = _l2_normalize(vec)
                with _EMB_CACHE_LOCK:
                    _EMB_CACHE[key] = vec
                _EMB_LAST_ERROR = ""
                _trace("embed", attempt + 1, f"ok:{mid}")
                return vec
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                errors.append(f"HTTP {exc.code} model={mid}: {detail}")
                _trace("embed", attempt + 1, f"http_{exc.code}")
                detail_l = detail.lower()
                if (
                    not fallback_tried
                    and exc.code in (401, 403, 404)
                    and (
                        "модел" in detail_l
                        or "model" in detail_l
                        or "не найден" in detail_l
                        or "not found" in detail_l
                    )
                ):
                    fallback_tried = True
                    for alt in embed_model_candidates(preferred):
                        if alt != mid:
                            models.append(alt)
                            break
                    break
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                _trace("embed", attempt + 1, type(exc).__name__)
                if _is_hard_connect_error(exc):
                    _EMB_LAST_ERROR = " | ".join(errors)
                    return None
                time.sleep(0.6 * (attempt + 1))
    _EMB_LAST_ERROR = " | ".join(errors) if errors else "unknown"
    return None


def get_client() -> ChatClient | None:
    global _CLIENT
    if not model_available():
        return None
    with _CLIENT_LOCK:
        if _CLIENT is None:
            try:
                _CLIENT = ChatClient()
            except RuntimeError:
                return None
        return _CLIENT


def probe_connectivity(timeout_sec: float | None = None) -> bool:
    """Best-effort TCP connect check. Does NOT trip circuit (non-latching)."""
    env = resolve_model_env()
    if not env["api_key"] or not env["model"]:
        return False
    parsed = urlparse(env["base_url"])
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connect_t = float(timeout_sec or getattr(config, "LLM_CONNECT_TIMEOUT_SEC", 2.5))
    try:
        with socket.create_connection((host, port), timeout=connect_t):
            return True
    except OSError:
        return False
