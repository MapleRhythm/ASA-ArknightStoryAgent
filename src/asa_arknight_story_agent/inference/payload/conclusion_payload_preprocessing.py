from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.pipeline.constants import CONCLUSION_IGNORED_EXTRA_FIELDS


def preprocess_conclusion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    if "next_action" not in payload:
        decision_action = infer_action_from_legacy_fields(payload)
        if decision_action:
            payload["next_action"] = decision_action
    if "clarification_question" not in payload and payload.get("follow_up_question"):
        payload["clarification_question"] = payload.get("follow_up_question")
    if "missing_slots" not in payload:
        additional_evidence = payload.get("additional_evidence_needed")
        payload["missing_slots"] = additional_evidence if isinstance(additional_evidence, list) else []
    if not str(payload.get("answer") or "").strip() and str(payload.get("final_answer") or "").strip():
        payload["answer"] = payload.get("final_answer")
    payload.setdefault("answer", "")
    return {key: value for key, value in payload.items() if key not in CONCLUSION_IGNORED_EXTRA_FIELDS}


def infer_action_from_legacy_fields(payload: dict[str, Any]) -> str:
    decision = str(payload.get("decision") or "").strip().lower()
    decision_action = {
        "retrieve": "retrieve_more",
        "retrieve_more": "retrieve_more",
        "need_more_evidence": "retrieve_more",
        "more_evidence": "retrieve_more",
        "answer": "answer_directly",
        "answer_directly": "answer_directly",
        "direct_answer": "answer_directly",
        "clarify": "clarify_user",
        "clarify_user": "clarify_user",
        "abstain": "abstain",
    }.get(decision)
    if decision_action:
        return decision_action
    if str(payload.get("answer") or "").strip():
        return "answer_directly"
    if payload.get("additional_evidence_needed"):
        return "retrieve_more"
    return ""
