"""OpenAI-compatible chat client (stdlib). Secrets from env only."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_sec: float = 60.0


def resolve_model_env() -> dict[str, str]:
    try:
        from agent.llm import load_dotenv

        load_dotenv()
    except Exception:
        pass
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
    return {"api_key": key, "base_url": base.rstrip("/"), "model": model}


def model_available() -> bool:
    env = resolve_model_env()
    return bool(env["api_key"] and env["model"])


class ChatClient:
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig()
        self.env = resolve_model_env()
        if not self.env["api_key"] or not self.env["model"]:
            raise RuntimeError(
                "LLM env missing. Set OPENAI_API_KEY + OPENAI_MODEL "
                "(optional OPENAI_BASE_URL), or PLATFORM_* equivalents."
            )
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self.env["model"]

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.env["model"],
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.env['base_url']}/chat/completions"
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.env['api_key']}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        self.calls += 1
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected LLM response: {body!r}") from exc
