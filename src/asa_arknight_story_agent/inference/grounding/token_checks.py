from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.grounding.fallback import build_grounded_fallback_answer
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


GROUNDING_HIT_RATE_THRESHOLD = 0.25
GROUNDING_MIN_MISSED_LONG_TOKENS = 4


def validate_token_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    long_tokens: list[str],
    evidence_pool: str,
    max_round_reached: bool,
) -> ConclusionResult | None:
    missing_tokens = [token for token in long_tokens if token not in evidence_pool]
    hit_count = len(long_tokens) - len(missing_tokens)
    hit_rate = hit_count / len(long_tokens) if long_tokens else 1.0

    if (
        hit_rate >= GROUNDING_HIT_RATE_THRESHOLD
        or len(missing_tokens) < GROUNDING_MIN_MISSED_LONG_TOKENS
    ):
        return None

    if max_round_reached:
        grounded_answer = build_grounded_fallback_answer(
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
            missing_tokens=missing_tokens,
        )
        return ConclusionResult(
            next_action="abstain",
            answer=grounded_answer,
            missing_slots=conclusion.missing_slots or ["grounding 校验未通过的关键词"],
            clarification_question="",
            follow_up_hypothesis=None,
        )
    follow_up_hypothesis = HypothesisDocument(
        question=question,
        intent=hypothesis.intent,
        query_type=hypothesis.query_type,
        entities=hypothesis.entities,
        keywords=dedupe_keep_order(hypothesis.keywords + missing_tokens[:6])[:20],
        expected_answer_type=hypothesis.expected_answer_type,
        dialogue_context=hypothesis.dialogue_context,
    )
    return ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=conclusion.missing_slots or list(missing_tokens[:6]),
        clarification_question="",
        follow_up_hypothesis=follow_up_hypothesis,
    )
