from __future__ import annotations

from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import build_heuristic_follow_up_hypothesis
from asa_arknight_story_agent.inference.grounding.identity import unsupported_required_entity_anchor
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def validate_required_anchor(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence_pool: str,
    max_round_reached: bool,
) -> ConclusionResult | None:
    unsupported_anchor = unsupported_required_entity_anchor(question, hypothesis, evidence_pool)
    if not unsupported_anchor:
        return None

    missing_slots = [f"需要包含“{unsupported_anchor}”或其别名的直接证据"]
    if max_round_reached:
        return ConclusionResult(
            next_action="abstain",
            answer=f"现有检索证据不足以确认“{unsupported_anchor}”所指的具体内容。",
            missing_slots=missing_slots,
            clarification_question="",
            follow_up_hypothesis=None,
        )
    return ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=missing_slots,
        clarification_question="",
        follow_up_hypothesis=build_heuristic_follow_up_hypothesis(question, hypothesis, missing_slots),
    )
