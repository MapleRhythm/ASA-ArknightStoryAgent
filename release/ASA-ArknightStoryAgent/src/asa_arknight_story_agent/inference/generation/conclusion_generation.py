from __future__ import annotations

from asa_arknight_story_agent.inference.payload.json_utils import extract_json_object, repair_json_like_output
from asa_arknight_story_agent.inference.payload.normalization import (
    normalize_conclusion_payload,
    parse_conclusion_json_like_output,
)
from asa_arknight_story_agent.inference.pipeline.constants import RETRIEVAL_ACTIONS_ORDER
from asa_arknight_story_agent.inference.pipeline.types import (
    ConclusionResult,
    HypothesisDocument,
    ModelOutputError,
)


def normalize_minimal_conclusion_output(raw_output: str, prompt_mode: str) -> str:
    if prompt_mode == "minimal" and not raw_output.lstrip().startswith(("{", "<think>")):
        return "{" + raw_output
    return raw_output


def parse_conclusion_output(
    raw_output: str,
    *,
    question: str,
    current_hypothesis: HypothesisDocument,
    max_round_reached: bool,
) -> ConclusionResult:
    repaired_output = repair_json_like_output(raw_output)
    payload = extract_json_object(repaired_output)
    if payload:
        return normalize_conclusion_payload(
            payload,
            question=question,
            dialogue_context=current_hypothesis.dialogue_context,
            current_intent=current_hypothesis.intent,
            current_hypothesis=current_hypothesis,
            max_round_reached=max_round_reached,
        )

    conclusion = parse_conclusion_json_like_output(
        repaired_output,
        question=question,
        dialogue_context=current_hypothesis.dialogue_context,
        current_intent=current_hypothesis.intent,
        current_hypothesis=current_hypothesis,
        max_round_reached=max_round_reached,
    )
    if conclusion is None:
        raise ModelOutputError(f"invalid conclusion json: {repaired_output}")
    return conclusion


def default_failed_conclusion(*, max_round_reached: bool) -> ConclusionResult:
    if max_round_reached:
        return ConclusionResult(
            next_action="abstain",
            answer="现有检索证据不足以确认，且已达到检索轮次上限。",
            missing_slots=["conclusion_generation 未产生可解析结论"],
            clarification_question="",
            follow_up_hypothesis=None,
        )
    return ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=["conclusion_generation 未产生可解析结论，需要继续补充直接证据"],
        clarification_question="",
        follow_up_hypothesis=None,
    )


def select_self_consistent_conclusion(conclusions: list[ConclusionResult]) -> ConclusionResult:
    action_counts: dict[str, int] = {}
    for conclusion in conclusions:
        action_counts[conclusion.next_action] = action_counts.get(conclusion.next_action, 0) + 1
    winning_action = max(
        action_counts,
        key=lambda action: (action_counts[action], -RETRIEVAL_ACTIONS_ORDER.index(action)),
    )
    return next(conclusion for conclusion in conclusions if conclusion.next_action == winning_action)
