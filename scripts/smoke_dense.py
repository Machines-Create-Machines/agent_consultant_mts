#!/usr/bin/env python3
"""Smoke: embed(chunk.text_ctx) must retrieve that chunk in top-k.

Near-duplicate headings (comma / title variants) can tie at cosine≈1.0;
we require the source id in top-3, not strict top-1 equality.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import embed_query, load_dotenv, resolve_embed_env  # noqa: E402

TOP_K = 3


def main() -> None:
    load_dotenv()
    npy = ROOT / "build" / "embeddings.npy"
    ids_path = ROOT / "build" / "embedding_ids.json"
    chunks_path = ROOT / "build" / "chunks.jsonl"
    if not npy.is_file() or not ids_path.is_file():
        raise SystemExit("missing embeddings.npy or embedding_ids.json — run embed_index.py")

    env = resolve_embed_env()
    print(f"smoke via {env['base_url']} model={env['model']} source={env.get('source')}", flush=True)

    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    mat = np.load(str(npy))
    if len(ids) != len(mat):
        raise SystemExit(f"misaligned: ids={len(ids)} emb={len(mat)}")

    by_id: dict[str, str] = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("chunk_id")
        text = (obj.get("text_indexed") or obj.get("text_ctx") or "").strip()
        if cid and text:
            by_id[cid] = text[:2000]

    mat_f = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat_f, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat_n = mat_f / norms
    id_to_i = {cid: i for i, cid in enumerate(ids)}

    sample = random.sample(ids, min(20, len(ids)))
    fails = 0
    for cid in sample:
        text = by_id.get(cid)
        if not text:
            print(f"SKIP {cid}: no text_ctx")
            continue
        vec = embed_query(text, model=env["model"], timeout_sec=60.0)
        if not vec:
            raise SystemExit(f"embed failed for {cid}")
        q = np.asarray(vec, dtype=np.float32)
        q = q / (float(np.linalg.norm(q)) or 1.0)
        sims = mat_n @ q
        top_idx = np.argpartition(-sims, min(TOP_K, len(sims) - 1))[:TOP_K]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        top_ids = [ids[int(i)] for i in top_idx]
        self_i = id_to_i.get(cid)
        self_score = float(sims[self_i]) if self_i is not None else -1.0
        ok = cid in top_ids
        rank = top_ids.index(cid) + 1 if ok else -1
        print(
            f"{'OK' if ok else 'FAIL'} {cid} -> {top_ids[0]} "
            f"rank={rank} self={self_score:.4f} top={float(sims[top_idx[0]]):.4f}"
        )
        if not ok:
            fails += 1
    if fails:
        raise SystemExit(f"smoke_dense FAILED: {fails}/{len(sample)} not in top-{TOP_K}")
    print(f"smoke_dense PASS n={len(sample)} top_k={TOP_K}")


if __name__ == "__main__":
    main()
