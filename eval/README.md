# Eval contour

Autonomous loop: run `solution.py` on `questions_dataset.jsonl`, write a reproducible report under `eval/runs/<run_id>/`.

## Setup

```bash
cd /home/dan/projects/agent_consultant_mts
# PyYAML required; stdlib HTTP client used for the judge (no openai package).
pip install pyyaml   # or: uv pip install --python .venv/bin/python pyyaml
```

Secrets (full judge only), from env — never written into reports:

- `OPENAI_API_KEY` / `PLATFORM_API_KEY`
- `OPENAI_BASE_URL` / `PLATFORM_BASE_URL` (optional)
- `OPENAI_MODEL` / `PLATFORM_MODEL`

## Commands

Fast loop (deterministic metrics, no LLM judge):

```bash
python -m eval.run --no-judge --limit 20 --tag smoke
```

Full dev split (~70%, holdout excluded by default):

```bash
python -m eval.run --tag full-dev
```

Holdout (explicit only):

```bash
python -m eval.run --split holdout --tag holdout-1
```

Ablation / compare:

```bash
python -m eval.run --no-judge --tag chunking-v2 --baseline-run YYYYMMDD-HHMM__chunking-v1
```

Filter by dataset tag (e.g. `solar`, `dialog`, `hard`):

```bash
python -m eval.run --no-judge --case-tag solar --tag solar-only
```

Disable judge cache:

```bash
python -m eval.run --no-cache --tag judge-fresh
```

## Solution I/O contract

`solution.py` must expose one of: `predict`, `answer`, `run_case`, `solve`, `generate`.

Preferred:

```python
def predict(question: str, history: list[dict] | None = None) -> dict | str:
    ...
```

Eval sends **only** `question` + `history` (never `gold_*` / nuggets).  
Dict may include: `answer`, `behavior`, `products`, `retrieved_ids`, `retrieved_scores`, `citations`, `llm_calls`, `stages`.

## Artifacts

See `eval/runs/<run_id>/` — `summary.md`, `metrics.json`, `cases.csv`, `answers.jsonl`, `failures.md`, `diff.md`, plus `eval/runs/index.csv`.
