from __future__ import annotations

from asa_arknight_story_agent.inference.payload.json_like_fields import (
    extract_json_like_bare_field,
    extract_json_like_missing_slots,
    extract_json_like_string_field,
)
from asa_arknight_story_agent.inference.payload.utils import normalize_string_list
from asa_arknight_story_agent.inference.pipeline.constants import RETRIEVAL_ACTIONS
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import build_heuristic_follow_up_hypothesis
from asa_arknight_story_agent.inference.planning.query_understanding import build_hypothesis
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order
from asa_arknight_story_agent.inference.payload.truncated_answer_recovery import recover_truncated_grounded_answer
from asa_arknight_story_agent.inference.payload.tuple_conclusion_parsing import parse_tuple_like_conclusion_output


def parse_conclusion_json_like_output(
    text: str,
    *,
    question: str,
    dialogue_context: str,
    current_intent: str,
    current_hypothesis: HypothesisDocument | None = None,
    max_round_reached: bool = False,
) -> ConclusionResult | None:
    tuple_conclusion = parse_tuple_like_conclusion_output(
        text,
        question=question,
        dialogue_context=dialogue_context,
        current_intent=current_intent,
        current_hypothesis=current_hypothesis,
        max_round_reached=max_round_reached,
    )
    if tuple_conclusion is not None:
        return tuple_conclusion

    truncated_conclusion = recover_truncated_grounded_answer(
        text,
        question=question,
        max_round_reached=max_round_reached,
    )
    if truncated_conclusion is not None:
        return truncated_conclusion

    next_action = extract_json_like_bare_field(text, "next_action")
    next_action = {
        "retrieve": "retrieve_more",
        "answer": "answer_directly",
        "direct_answer": "answer_directly",
    }.get(next_action, next_action)
    answer = extract_json_like_string_field(text, "answer").strip()
    if not answer:
        answer = extract_json_like_string_field(text, "final_answer").strip()
    if not next_action and answer:
        next_action = "answer_directly"
    if next_action not in RETRIEVAL_ACTIONS:
        return None

    missing_slots = extract_json_like_missing_slots(text)
    clarification_question = extract_json_like_string_field(text, "clarification_question").strip()
    if next_action in {"answer_directly", "abstain"}:
        if not answer:
            return None
        return ConclusionResult(
            next_action=next_action,
            answer=answer,
            missing_slots=missing_slots,
            clarification_question=clarification_question,
            follow_up_hypothesis=None,
        )
    if next_action == "clarify_user":
        if not clarification_question:
            return None
        return ConclusionResult(
            next_action=next_action,
            answer="",
            missing_slots=missing_slots,
            clarification_question=clarification_question,
            follow_up_hypothesis=None,
        )
    if not missing_slots:
        missing_slots = ["需要补充更直接的证据"]
    if max_round_reached:
        return ConclusionResult(
            next_action="abstain",
            answer="现有检索证据不足以确认，且已达到检索轮次上限。",
            missing_slots=missing_slots,
            clarification_question="",
            follow_up_hypothesis=None,
        )
    follow_up_hypothesis = (
        build_heuristic_follow_up_hypothesis(question, current_hypothesis, missing_slots)
        if current_hypothesis is not None
        else build_hypothesis(question + " " + " ".join(missing_slots[:4]), dialogue_context)
    )
    return ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=dedupe_keep_order([*missing_slots, "JSON-like 结论已使用启发式续检索"]),
        clarification_question="",
        follow_up_hypothesis=follow_up_hypothesis,
    )
