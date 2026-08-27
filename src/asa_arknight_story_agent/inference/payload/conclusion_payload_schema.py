from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from asa_arknight_story_agent.inference.payload.utils import (
    answer_from_structured_facts,
    compact_inferred_facts_payload,
    compact_supported_facts_payload,
    normalize_string_list,
)
from asa_arknight_story_agent.inference.pipeline.constants import CONCLUSION_SCHEMA_FIELDS, RETRIEVAL_ACTIONS
from asa_arknight_story_agent.inference.pipeline.types import ModelOutputError


@dataclass
class NormalizedConclusionFields:
    next_action: str
    answer: str
    missing_slots: list[str]
    clarification_question: str
    supported_facts: list[dict[str, Any]]
    inferred_facts: list[Any]
    follow_up_hypothesis_payload: Any


def validate_conclusion_schema(payload: dict[str, Any], *, question: str) -> None:
    extra_keys = set(payload) - set(CONCLUSION_SCHEMA_FIELDS)
    if extra_keys:
        raise ModelOutputError(f"unexpected conclusion fields: {sorted(extra_keys)}")
    optional_missing_fields = {
        "question",
        "clarification_question",
        "follow_up_hypothesis",
        "reflect_tokens",
        "final_answer",
        "supported_facts",
        "inferred_facts",
        "reason",
    }
    missing_fields = [
        field for field in CONCLUSION_SCHEMA_FIELDS if field not in payload and field not in optional_missing_fields
    ]
    if missing_fields:
        raise ModelOutputError(f"missing conclusion fields: {missing_fields}")
    payload_question = str(payload.get("question") or question).strip()
    if not payload_question:
        raise ModelOutputError("conclusion must contain question")


def normalize_conclusion_fields(payload: dict[str, Any]) -> NormalizedConclusionFields:
    next_action = str(payload.get("next_action", "")).strip()
    next_action = {
        "retrieve": "retrieve_more",
        "answer": "answer_directly",
        "direct_answer": "answer_directly",
    }.get(next_action, next_action)
    if next_action not in RETRIEVAL_ACTIONS:
        raise ModelOutputError(f"invalid conclusion action: {next_action or '<empty>'}")
    answer = str(payload.get("answer", "") or "").strip()
    supported_facts = compact_supported_facts_payload(payload.get("supported_facts"))
    inferred_facts = compact_inferred_facts_payload(payload.get("inferred_facts"))
    if not answer and next_action == "answer_directly":
        answer = answer_from_structured_facts(supported_facts, inferred_facts)
    missing_slots = normalize_string_list(payload.get("missing_slots"), limit=8)
    clarification_question = str(payload.get("clarification_question") or "").strip()
    if next_action in {"answer_directly", "abstain"} and not answer:
        raise ModelOutputError(f"{next_action} requires non-empty answer")
    if next_action == "clarify_user" and not clarification_question:
        raise ModelOutputError("clarify_user requires clarification_question")
    return NormalizedConclusionFields(
        next_action=next_action,
        answer=answer,
        missing_slots=missing_slots,
        clarification_question=clarification_question,
        supported_facts=supported_facts,
        inferred_facts=inferred_facts,
        follow_up_hypothesis_payload=payload.get("follow_up_hypothesis"),
    )
