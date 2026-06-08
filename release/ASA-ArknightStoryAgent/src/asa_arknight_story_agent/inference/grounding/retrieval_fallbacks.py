from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.reveal.answer import build_reveal_answer
from asa_arknight_story_agent.inference.special.answers import (
    build_event_reference_answer,
    build_suiling_crisis_answer,
    suiling_crisis_answer_needs_correction,
)


def answer_from_retrieval_fallbacks(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    max_round_reached: bool,
) -> ConclusionResult | None:
    if conclusion.next_action not in {"retrieve_more", "abstain"}:
        return None
    if not (max_round_reached or conclusion.next_action == "abstain"):
        return None

    fallback_answer = (
        build_reveal_answer(question=question, hypothesis=hypothesis, evidence=evidence)
        or build_suiling_crisis_answer(question=question, hypothesis=hypothesis, evidence=evidence)
        or build_event_reference_answer(question=question, hypothesis=hypothesis, evidence=evidence)
    )
    if not fallback_answer:
        return None
    return ConclusionResult(
        next_action="answer_directly",
        answer=fallback_answer,
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
    )


def corrected_special_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
) -> ConclusionResult | None:
    suiling_crisis_answer = build_suiling_crisis_answer(
        question=question,
        hypothesis=hypothesis,
        evidence=evidence,
    )
    if not suiling_crisis_answer or not suiling_crisis_answer_needs_correction(conclusion.answer):
        return None
    return ConclusionResult(
        next_action="answer_directly",
        answer=suiling_crisis_answer,
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
    )
