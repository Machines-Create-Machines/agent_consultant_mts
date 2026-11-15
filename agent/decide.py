"""Coverage × clarity decision matrix."""

from __future__ import annotations

from agent.clarity import ClarityResult, worth_asking
from agent.coverage import CoverageResult
from agent.schemas import Decision, SlotValue, UnderstandResult


def decide(
    coverage: CoverageResult,
    clarity: ClarityResult,
    understand: UnderstandResult,
    slots: dict[str, SlotValue],
    history: list[dict] | None,
) -> Decision:
    cov = coverage.level
    clr = clarity.level

    if cov == "none":
        return Decision(
            action="not_in_kb",
            coverage=cov,
            clarity=clr,
            missing_aspects=coverage.missing_aspects,
            reason="coverage_none",
        )

    if cov == "high" and clr == "clear":
        return Decision(action="answer", coverage=cov, clarity=clr, reason="high_clear")

    if cov == "high" and clr != "clear":
        slot = worth_asking(understand.intent, slots, history, cov)
        return Decision(
            action="answer",
            coverage=cov,
            clarity=clr,
            clarify_slot=slot,
            reason="high_unclear",
        )

    if cov == "partial" and clr == "clear":
        return Decision(
            action="partial",
            coverage=cov,
            clarity=clr,
            missing_aspects=coverage.missing_aspects,
            reason="partial_clear",
        )

    # partial + unclear
    slot = worth_asking(understand.intent, slots, history, cov)
    if slot:
        return Decision(
            action="clarify",
            coverage=cov,
            clarity=clr,
            clarify_slot=slot,
            reason="partial_unclear_ask",
        )
    return Decision(
        action="partial",
        coverage=cov,
        clarity=clr,
        missing_aspects=coverage.missing_aspects,
        reason="partial_unclear_noask",
    )
