"""OpenAI-compatible client with JSON parse + retry (PLATFORM_* first)."""

from __future__ import annotations

import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent import config

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_ENV_LOADED = False
_CIRCUIT_OPEN_UNTIL = 0.0
_CIRCUIT_REASON = ""


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


def resolve_model_env() -> dict[str, str]:
    load_dotenv()
    key = (
        os.environ.get("PLATFORM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    )
    base = (
        os.environ.get("PLATFORM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = (
        os.environ.get("PLATFORM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("LLM_MODEL")
        or ""
    )
    ssl_verify = (
        os.environ.get("PLATFORM_SSL_VERIFY")
        or os.environ.get("LLM_SSL_VERIFY")
        or "true"
    )
    return {
        "api_key": key,
        "base_url": base.rstrip("/"),
        "model": model,
        "ssl_verify": ssl_verify.lower() not in ("0", "false", "no"),
    }


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


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
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


def _is_connectivity_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionError)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return True
    return any(
        token in text
        for token in ("timed out", "timeout", "connection refused", "name or service not known", "network is unreachable")
    )


class ChatClient:
    def __init__(
        self,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_sec: float | None = None,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec if timeout_sec is not None else config.LLM_TIMEOUT_SEC
        self.env = resolve_model_env()
        self.calls = 0
        if not self.env["api_key"] or not self.env["model"]:
            raise RuntimeError(
                "LLM env missing. Set PLATFORM_API_KEY + PLATFORM_MODEL "
                "(optional PLATFORM_BASE_URL), or OPENAI_* equivalents."
            )

    def _timeout(self) -> float:
        # urllib.request.urlopen timeout must be a single float (not requests-style tuple)
        return float(self.timeout_sec)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if _circuit_open():
            raise RuntimeError(f"LLM circuit open: {_CIRCUIT_REASON or 'unavailable'}")

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
        for attempt in range(config.LLM_RETRIES + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout(), context=ctx) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self.calls += 1
                _clear_circuit()
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_err = RuntimeError(f"LLM HTTP {exc.code}: {detail}")
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(0.4 * (attempt + 1))
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if _is_connectivity_error(exc):
                    _trip_circuit(str(exc)[:200])
                    break
                time.sleep(0.4 * (attempt + 1))
        raise RuntimeError(str(last_err) if last_err else "LLM failed")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        text = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return extract_json(text)


_CLIENT: ChatClient | None = None


def get_client() -> ChatClient | None:
    global _CLIENT
    if not model_available():
        return None
    if _CLIENT is None:
        try:
            _CLIENT = ChatClient()
        except RuntimeError:
            return None
    return _CLIENT


def probe_connectivity(timeout_sec: float | None = None) -> bool:
    """Best-effort TCP connect check; trips circuit on failure."""
    if _circuit_open():
        return False
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
    except OSError as exc:
        _trip_circuit(f"tcp {host}:{port}: {exc}")
        return False
