#!/usr/bin/env python3
"""Build build/embeddings.npy for chunk text_ctx via EMB_URL or platform LLM_*."""

from __future__ import annotations

import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import (  # noqa: E402
    load_dotenv,
    reset_case_llm_state,
    resolve_embed_env,
)

CHECKPOINT = ROOT / "build" / "embeddings.partial.json"
BATCH = 8
TIMEOUT = 60.0


def _chunk_embed_text(obj: dict) -> str:
    return (obj.get("text_indexed") or obj.get("text_ctx") or "").strip()


def _l2(vec: list[float]) -> list[float]:
    import math

    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _warm_dns(base_url: str) -> None:
    host = urlparse(base_url).hostname
    if not host:
        return
    import socket

    for _ in range(3):
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            with socket.create_connection((host, 443), timeout=10):
                return
        except OSError:
            time.sleep(1.0)


def _post_embeddings(
    *,
    base_url: str,
    api_key: str,
    model: str,
    inputs: list[str],
    ssl_verify: bool,
    timeout: float = TIMEOUT,
) -> list[list[float]]:
    url = f"{base_url.rstrip('/')}/embeddings"
    payload = json.dumps(
        {"model": model, "input": inputs if len(inputs) > 1 else inputs[0]},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    ctx = None if ssl_verify else ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"bad embeddings response keys={list(body) if isinstance(body, dict) else type(body)}")
    # OpenAI-style: each item has index + embedding
    ordered = sorted(data, key=lambda x: int(x.get("index", 0)))
    out: list[list[float]] = []
    for item in ordered:
        emb = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(emb, list) or not emb:
            raise RuntimeError("empty embedding in batch")
        out.append(_l2([float(x) for x in emb]))
    if len(out) != len(inputs):
        raise RuntimeError(f"batch size mismatch got={len(out)} want={len(inputs)}")
    return out


def _post_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    inputs: list[str],
    ssl_verify: bool,
    tries: int = 5,
) -> list[list[float]]:
    last: Exception | None = None
    for i in range(tries):
        try:
            return _post_embeddings(
                base_url=base_url,
                api_key=api_key,
                model=model,
                inputs=inputs,
                ssl_verify=ssl_verify,
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = 1.5 * (i + 1)
            print(f"  retry {i + 1}/{tries} after {type(exc).__name__}: {exc} sleep={wait:.1f}s", flush=True)
            time.sleep(wait)
            _warm_dns(base_url)
    raise RuntimeError(str(last) if last else "embed failed")


def main() -> None:
    load_dotenv()
    reset_case_llm_state(deadline=None)
    env = resolve_embed_env()
    if not env["api_key"]:
        raise SystemExit(
            "no embeddings credentials: set EMB_URL+EMB_KEY (local) "
            "or LLM_*/PLATFORM_* (platform contract)"
        )
    model = env["model"]
    print(
        f"embeddings via {env['base_url']}/embeddings model={model} "
        f"source={env.get('source')} batch={BATCH} timeout={TIMEOUT}s",
        flush=True,
    )
    print("warming DNS/TCP…", flush=True)
    _warm_dns(env["base_url"])

    chunks_path = ROOT / "build" / "chunks.jsonl"
    ids: list[str] = []
    texts: list[str] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("chunk_id")
        text = _chunk_embed_text(obj) or (obj.get("text_raw") or "").strip()
        if cid and text:
            ids.append(cid)
            texts.append(text[:2000])

    done: dict[str, list[float]] = {}
    if CHECKPOINT.exists():
        try:
            raw = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                done = {k: v for k, v in raw.items() if isinstance(v, list) and v}
            print(f"resuming checkpoint n={len(done)}", flush=True)
        except Exception:  # noqa: BLE001
            done = {}

    print("probe…", flush=True)
    probe = _post_with_retries(
        base_url=env["base_url"],
        api_key=env["api_key"],
        model=model,
        inputs=["тест эмбеддингов Energytool"],
        ssl_verify=bool(env.get("ssl_verify", True)),
    )[0]
    print(f"probe ok dim={len(probe)} n_chunks={len(texts)}", flush=True)

    vectors: list[list[float] | None] = [None] * len(ids)
    pending_idx: list[int] = []
    for i, cid in enumerate(ids):
        if cid in done and len(done[cid]) == len(probe):
            vectors[i] = done[cid]
        else:
            pending_idx.append(i)

    print(f"to embed: {len(pending_idx)} (cached {len(ids) - len(pending_idx)})", flush=True)

    for start in range(0, len(pending_idx), BATCH):
        batch_i = pending_idx[start : start + BATCH]
        batch_texts = [texts[i] for i in batch_i]
        vecs = _post_with_retries(
            base_url=env["base_url"],
            api_key=env["api_key"],
            model=model,
            inputs=batch_texts,
            ssl_verify=bool(env.get("ssl_verify", True)),
        )
        for i, vec in zip(batch_i, vecs):
            vectors[i] = vec
            done[ids[i]] = vec
        CHECKPOINT.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        done_n = sum(1 for v in vectors if v is not None)
        print(f"embedded {done_n}/{len(texts)}", flush=True)

    if any(v is None for v in vectors):
        raise SystemExit("internal: missing vectors")

    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arr = (arr / norms).astype(np.float16)

    out_npy = ROOT / "build" / "embeddings.npy"
    out_ids = ROOT / "build" / "embedding_ids.json"
    np.save(out_npy, arr)
    out_ids.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
    print(f"wrote {out_npy} shape={arr.shape} dtype={arr.dtype} ids={len(ids)}")


if __name__ == "__main__":
    main()
