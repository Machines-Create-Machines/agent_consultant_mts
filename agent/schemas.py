"""Dataclasses for the online agent cycle (Consultant v4)."""

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
    what_it_is: str = ""
    fits: str = ""
    overview_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Document:
    source_file: str
    title: str
    product: str | None
    clean_text: str
    n_chars: int
    headings: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Aspect:
    """Kept for eval/backward compat."""

    name: str
    attr: str = "механизм"
    keywords: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class SlotValue:
    value: Any
    source: str = "stated"


@dataclass
class PlannerResult:
    route: str = "consult"  # consult | off_topic | out_of_scope | lead | greeting
    standalone_query: str = ""
    question_kind: str = "fact"  # fact | recommendation
    explicit_product: str | None = None
    explicit_products: list[str] = field(default_factory=list)
    consult_intent: str = "energy_fact"
    # catalog | product_fact | energy_fact | recommendation | comparison
    required_facets: list[str] = field(default_factory=list)
    missing_critical_fact: str | None = None
    query_scope: str = "specific"  # specific | catalog_overview
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def intent(self) -> str:
        if self.route == "lead":
            return "lead_intent"
        if self.route == "out_of_scope":
            return "out_of_scope_tech"
        if self.route in ("off_topic", "greeting"):
            return self.route
        if self.consult_intent == "catalog":
            return "product_consult"
        if self.consult_intent in ("recommendation", "comparison"):
            return "task_diagnosis"
        if self.consult_intent == "product_fact":
            return "product_consult"
        return "product_consult"

    @property
    def out_of_scope(self) -> bool:
        return self.route == "out_of_scope"

    @property
    def lead_signal(self) -> bool:
        return self.route == "lead"

    @property
    def query_variants(self) -> list[str]:
        return []

    @property
    def entities(self) -> list[str]:
        if self.explicit_products:
            return list(self.explicit_products)
        return [self.explicit_product] if self.explicit_product else []

    @property
    def aspects(self) -> list[Aspect]:
        attr = "наличие" if self.question_kind == "fact" else "применимость"
        return [Aspect(name="основной вопрос", attr=attr, keywords=[self.standalone_query], weight=1)]

    @property
    def slot_delta(self) -> dict[str, SlotValue]:
        return {}

    @property
    def topic_switch(self) -> bool:
        return False

    @property
    def unresolved_reference(self) -> bool:
        return False


UnderstandResult = PlannerResult


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
class SourceBlock:
    chunk_id: str
    heading: str
    text: str
    source_file: str = ""
    products: list[str] = field(default_factory=list)


@dataclass
class ContextPack:
    intent: str
    blocks: list[SourceBlock] = field(default_factory=list)
    cards_text: str = ""
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    used_rerank: bool = False


@dataclass
class Decision:
    action: str
    coverage: str = "none"
    clarity: str = "clear"
    clarify_slot: str | None = None
    missing_aspects: list[str] = field(default_factory=list)
    reason: str = ""
    top_score: float = 0.0


@dataclass
class GenerateResult:
    answer: str
    cited_ids: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    next_action: str = "none"
    missing_slot: str | None = None
    answer_with_cites: str = ""
    mode: str = "answer"
    follow_up: str | None = None


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
