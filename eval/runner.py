"""Per-case solution runner with timeout, retries, concurrency."""

from __future__ import annotations

import concurrent.futures
import importlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from eval.loader import Case


def _normalize_result(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"answer": ""}
    if isinstance(raw, str):
        return {"answer": raw}
    if isinstance(raw, dict):
        out = dict(raw)
        if "answer" not in out and "text" in out:
            out["answer"] = out["text"]
        out.setdefault("answer", "")
        # normalize retrieved shapes
        retrieved = out.get("retrieved_ids")
        if retrieved is None and out.get("retrieved"):
            items = out["retrieved"]
            if items and isinstance(items[0], dict):
                out["retrieved_ids"] = [x.get("id") for x in items if x.get("id")]
                out["retrieved_scores"] = [x.get("score") for x in items]
            else:
                out["retrieved_ids"] = list(items)
        return out
    return {"answer": str(raw)}


def load_predict_fn(module_name: str = "solution") -> Callable[..., Any]:
    mod = importlib.import_module(module_name)
    for name in ("predict", "answer", "run_case", "solve", "generate"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise AttributeError(
        f"{module_name}.py must expose predict(question, history=None) "
        "or answer/run_case/solve/generate"
    )


def _call_predict(fn: Callable[..., Any], public: dict[str, Any]) -> dict[str, Any]:
    # Prefer (question, history); fall back to single case dict.
    try:
        result = fn(public["question"], public.get("history") or [])
    except TypeError:
        try:
            result = fn(question=public["question"], history=public.get("history") or [])
        except TypeError:
            result = fn(public)
    return _normalize_result(result)


def _run_subprocess(project_root: Path, public: dict[str, Any], timeout: float) -> dict[str, Any]:
    helper = r"""
import json, sys
from importlib import import_module
payload = json.loads(sys.stdin.read())
mod = import_module("solution")
fn = None
for name in ("predict", "answer", "run_case", "solve", "generate"):
    cand = getattr(mod, name, None)
    if callable(cand):
        fn = cand
        break
if fn is None:
    raise SystemExit("no predict-like function in solution")
try:
    out = fn(payload["question"], payload.get("history") or [])
except TypeError:
    try:
        out = fn(question=payload["question"], history=payload.get("history") or [])
    except TypeError:
        out = fn(payload)
if not isinstance(out, dict):
    out = {"answer": out}
print(json.dumps(out, ensure_ascii=False))
"""
    proc = subprocess.run(
        [sys.executable, "-c", helper],
        input=json.dumps(public, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(project_root),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()})},
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"subprocess exit {proc.returncode}")
    line = proc.stdout.strip().splitlines()[-1]
    return _normalize_result(json.loads(line))


def run_one(
    case: Case,
    predict_fn: Callable[..., Any] | None,
    project_root: Path,
    timeout_sec: float,
    retries: int,
    isolation: str,
    cold_start: bool,
) -> dict[str, Any]:
    public = case.public_input()
    # Hard guarantee: never pass gold fields
    assert set(public.keys()) <= {"id", "question", "history"}

    last_err: str | None = None
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            if isolation == "subprocess":
                result = _run_subprocess(project_root, public, timeout_sec)
            else:
                assert predict_fn is not None
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_call_predict, predict_fn, public)
                    result = fut.result(timeout=timeout_sec)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            result = dict(result)
            result.update(
                {
                    "id": case.id,
                    "latency_ms": round(latency_ms, 2),
                    "cold_start": cold_start and attempt == 0,
                    "attempts": attempt + 1,
                    "error": None,
                    "stages": result.get("stages") or {},
                }
            )
            result.setdefault("llm_calls", 0)
            result.setdefault("retrieved_ids", [])
            result.setdefault("citations", [])
            return result
        except concurrent.futures.TimeoutError:
            last_err = f"timeout after {timeout_sec}s"
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {timeout_sec}s"
        except Exception as exc:  # noqa: BLE001 — isolate case failures
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                break
            time.sleep(0.5 * (attempt + 1))

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "id": case.id,
        "answer": "",
        "behavior": None,
        "products": [],
        "retrieved_ids": [],
        "citations": [],
        "llm_calls": 0,
        "latency_ms": round(latency_ms, 2),
        "cold_start": cold_start,
        "attempts": retries + 1,
        "error": last_err or "unknown_error",
        "traceback": traceback.format_exc()[-2000:],
        "stages": {},
    }


def run_cases(
    cases: list[Case],
    project_root: Path,
    module_name: str,
    concurrency: int,
    timeout_sec: float,
    retries: int,
    isolation: str,
) -> list[dict[str, Any]]:
    predict_fn = None
    if isolation != "subprocess":
        # Ensure project root on path
        root_s = str(project_root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)
        predict_fn = load_predict_fn(module_name)

    answers: list[dict[str, Any] | None] = [None] * len(cases)
    cold = True

    def _task(idx_case: tuple[int, Case, bool]) -> tuple[int, dict[str, Any]]:
        idx, case, is_cold = idx_case
        return idx, run_one(
            case=case,
            predict_fn=predict_fn,
            project_root=project_root,
            timeout_sec=timeout_sec,
            retries=retries,
            isolation=isolation,
            cold_start=is_cold,
        )

    # First case sequential for cold-start measurement, then parallel.
    if not cases:
        return []

    first = run_one(
        case=cases[0],
        predict_fn=predict_fn,
        project_root=project_root,
        timeout_sec=timeout_sec,
        retries=retries,
        isolation=isolation,
        cold_start=True,
    )
    answers[0] = first

    rest = list(enumerate(cases[1:], start=1))
    if rest:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futs = [
                ex.submit(_task, (idx, case, False))
                for idx, case in rest
            ]
            for fut in concurrent.futures.as_completed(futs):
                idx, rec = fut.result()
                answers[idx] = rec

    return [a for a in answers if a is not None]
