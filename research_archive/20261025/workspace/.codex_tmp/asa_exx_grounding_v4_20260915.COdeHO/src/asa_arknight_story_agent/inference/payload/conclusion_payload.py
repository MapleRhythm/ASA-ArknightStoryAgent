from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.payload.conclusion_followup_payload import resolve_retrieve_more_follow_up
from asa_arknight_story_agent.inference.payload.conclusion_payload_preprocessing import preprocess_conclusion_payload
from asa_arknight_story_agent.inference.payload.conclusion_payload_schema import (
    normalize_conclusion_fields,
    validate_conclusion_schema,
)
from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload
from asa_arknight_story_agent.inference.pipeline.constants import (
    FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS,
    INITIAL_HYPOTHESIS_SCHEMA_FIELDS,
)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def normalize_conclusion_payload(
    payload: dict[str, Any],
    *,
    question: str,
    dialogue_context: str,
    current_intent: str,
    current_hypothesis: HypothesisDocument | None = None,
    max_round_reached: bool = False,
) -> ConclusionResult:
    payload = preprocess_conclusion_payload(payload)

    if set(payload).issubset(set(INITIAL_HYPOTHESIS_SCHEMA_FIELDS)) or set(payload).issubset(
        set(FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS) | {"intent"}
    ):
        follow_up_hypothesis = normalize_hypothesis_payload(
            payload,
            question=question,
            dialogue_context=dialogue_context,
            current_intent=current_intent,
        )
        next_action = "abstain" if max_round_reached else "retrieve_more"
        answer = "现有检索证据不足以确认，且已达到检索轮次上限。" if max_round_reached else ""
        return ConclusionResult(
            next_action=next_action,
            answer=answer,
            missing_slots=["需要补充更直接的桥接证据"],
            clarification_question="",
            follow_up_hypothesis=None if max_round_reached else follow_up_hypothesis,
        )

    validate_conclusion_schema(payload, question=question)
    fields = normalize_conclusion_fields(payload)
    next_action = fields.next_action
    answer = fields.answer
    missing_slots = fields.missing_slots
    follow_up_hypothesis: HypothesisDocument | None = None
    if next_action == "retrieve_more":
        next_action, answer, missing_slots, follow_up_hypothesis = resolve_retrieve_more_follow_up(
            question=question,
            dialogue_context=dialogue_context,
            current_intent=current_intent,
            current_hypothesis=current_hypothesis,
            follow_up_hypothesis_payload=fields.follow_up_hypothesis_payload,
            missing_slots=missing_slots,
            max_round_reached=max_round_reached,
        )
    else:
        follow_up_hypothesis = None
    return ConclusionResult(
        next_action=next_action,
        answer=answer,
        missing_slots=missing_slots,
        clarification_question=fields.clarification_question,
        follow_up_hypothesis=follow_up_hypothesis,
        supported_facts=fields.supported_facts,
        inferred_facts=fields.inferred_facts,
    )
