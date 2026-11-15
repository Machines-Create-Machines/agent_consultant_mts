"""Thresholds, fusion weights, paths, timeouts (Consultant v4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
RUNTIME_DB = BUILD_DIR / "runtime.sqlite"


@dataclass
class Thresholds:
    T_floor: float
    T_chunk: float
    T_high: float
    T_mid: float
    delta: float


# Calibrated-ish defaults for reranker / RRF scores (see scripts/calibrate_retrieval.py).
THRESHOLDS_RERANK = Thresholds(
    T_floor=0.18, T_chunk=0.32, T_high=0.55, T_mid=0.38, delta=0.05
)
THRESHOLDS_NO_RERANK = Thresholds(
    T_floor=0.012, T_chunk=0.010, T_high=0.020, T_mid=0.014, delta=0.002
)

# Evidence gate (reranker score when available; RRF otherwise).
GATE_ANSWER = 0.45
GATE_PARTIAL = 0.28
# Single-branch RRF@1 ≈ 0.0164; require stronger multi-hit for full answer.
GATE_ANSWER_RRF = 0.020
GATE_PARTIAL_RRF = 0.012

FUSION_WEIGHTS = {
    "dense": 0.55,
    "bm25": 0.45,
    "questions": 0.0,
    "entity": 0.0,
}

RRF_K = 60
MMR_LAMBDA = 0.7  # unused in v2 online path
FUSE_TOP_N = 25
RERANK_TOP_N = 12
RETRIEVE_K = {
    "pricing": 8,
    "product_consult": 8,
    "task_diagnosis": 8,
    "comparison": 10,
    "default": 8,
}
BM25_TOP = 25
DENSE_TOP = 25
RERANK_CANDIDATES = 20

MAX_EVIDENCE_CHARS = 16000
MAX_DOC_CHARS = 14000
MAX_KB_CHARS = 220000
MAX_ANSWER_WORDS = 150
MAX_COPY_NGRAM = 12
CLARIFY_STREAK_MAX = 2
LEAD_FIELD_ATTEMPTS = 2
PROMPT_VERSION = "v5.0-fullkb"
FULL_KB = True
# Keyword planner only — writer sees the whole KB, so P5 is not worth a second call.
PLANNER_LLM = False

LLM_TIMEOUT_SEC = 120.0
LLM_CONNECT_TIMEOUT_SEC = 10.0
LLM_RETRIES = 1
LLM_CIRCUIT_COOLDOWN_SEC = 60.0
LLM_HTTP_CONCURRENCY = 2
MAX_ATTEMPTS = {"P5": 1, "P6": 1, "P7": 1}

P5_TEMP = 0.0
P5_MAX_TOKENS = 280
P6_TEMP = 0.4
P6_MAX_TOKENS = 900
P7_TEMP = 0.1
P7_MAX_TOKENS = 700
P8_TEMP = 0.2
P8_MAX_TOKENS = 900

THEMATIC_SLOTS = ("object_type", "pain", "scale", "metering_state", "use_case")
DIALOG_SLOTS = ("segment", "region", "name", "phone", "company")

PRODUCT_NAMES = (
    "Выход на оптовый рынок",
    "Облачная платформа учёта",
    "Солнечные электростанции",
    "Поиск коммерческих потерь",
    "Распознавание показаний по фотографии",
)
