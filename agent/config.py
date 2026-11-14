"""Thresholds, fusion weights, paths, timeouts."""

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


THRESHOLDS_RERANK = Thresholds(
    T_floor=0.15, T_chunk=0.35, T_high=0.60, T_mid=0.40, delta=0.05
)
THRESHOLDS_NO_RERANK = Thresholds(
    T_floor=0.20, T_chunk=0.40, T_high=0.65, T_mid=0.45, delta=0.07
)

FUSION_WEIGHTS = {
    "dense": 0.45,
    "bm25": 0.30,
    "questions": 0.15,
    "entity": 0.10,
}

MMR_LAMBDA = 0.7
FUSE_TOP_N = 25
RERANK_TOP_N = 6
RETRIEVE_K = {
    "pricing": 6,
    "product_consult": 8,
    "task_diagnosis": 8,
    "comparison": 10,
    "default": 6,
}

MAX_EVIDENCE_CHARS = 12000  # ~3k tokens
MAX_ANSWER_WORDS = 150
CLARIFY_STREAK_MAX = 2
LEAD_FIELD_ATTEMPTS = 2

LLM_TIMEOUT_SEC = 4.0  # fail-fast; dead PLATFORM API must not hang CSV smoke
LLM_CONNECT_TIMEOUT_SEC = 2.0
LLM_RETRIES = 0  # no retries; first connectivity/timeout failure trips process circuit
# Process-level: after trip, skip all further LLM calls until process exit (no reopen).
LLM_CIRCUIT_COOLDOWN_SEC = 0.0
GLOBAL_BUDGET_SEC = 8.0

P5_TEMP = 0.1
P5_MAX_TOKENS = 300
P6_TEMP = 0.2
P6_MAX_TOKENS = 600
P7_TEMP = 0.0
P7_MAX_TOKENS = 600
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
