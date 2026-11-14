"""Dataclasses for the online agent cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    heading: str
    heading_path: list[str]
    text_raw: str
    text_ctx: str
    chunk_type: str = "text"
    parent_chunk_id: str | None = None
    n_tokens: int = 0
    products: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    hyp_questions: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    caption: str | None = None


@dataclass
class Card:
    product: str
    summary: str
    source_file: str
    chunk_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    not_for: list[str] = field(default_factory=list)
    pricing_model: str | None = None
    caveats: list[str] = field(default_factory=list)


@dataclass
class Aspect:
    name: str
    attr: str = "механизм"
    keywords: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class SlotValue:
    value: Any
    source: str = "stated"  # stated | inferred | conflict | missing


@dataclass
class UnderstandResult:
    intent: str = "other"
    out_of_scope: bool = False
    standalone_query: str = ""
    query_variants: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    aspects: list[Aspect] = field(default_factory=list)
    slot_delta: dict[str, SlotValue] = field(default_factory=dict)
    topic_switch: bool = False
    unresolved_reference: bool = False
    lead_signal: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk_id: str
    score: float
    text: str = ""
    source_file: str = ""
    heading: str = ""
    n_tokens: int = 0
    products: list[str] = field(default_factory=list)
    branches: dict[str, float] = field(default_factory=dict)


@dataclass
class Decision:
    action: str  # answer | clarify | partial | not_in_kb | refuse | greeting | lead | off_topic
    coverage: str = "none"  # high | partial | none
    clarity: str = "clear"  # clear | partial | unclear
    clarify_slot: str | None = None
    missing_aspects: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class GenerateResult:
    answer: str
    cited_ids: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    next_action: str = "none"
    missing_slot: str | None = None
    answer_with_cites: str = ""


@dataclass
class PipelineResult:
    answer: str
    behavior: str
    products: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    llm_calls: int = 0
    stages: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "behavior": self.behavior,
            "products": self.products,
            "retrieved_ids": self.retrieved_ids,
            "retrieved_scores": self.retrieved_scores,
            "citations": self.citations,
            "llm_calls": self.llm_calls,
            "stages": self.stages,
        }
