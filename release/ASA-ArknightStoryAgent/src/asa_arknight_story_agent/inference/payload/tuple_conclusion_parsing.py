from __future__ import annotations

import ast

from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import build_heuristic_follow_up_hypothesis
from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload
from asa_arknight_story_agent.inference.payload.utils import normalize_string_list
from asa_arknight_story_agent.inference.pipeline.constants import RETRIEVAL_ACTIONS
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument, ModelOutputError
from asa_arknight_story_agent.inference.planning.query_understanding import build_hypothesis
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def parse_tuple_like_conclusion_output(
    text: str,
    *,
    question: str,
    dialogue_context: str,
    current_intent: str,
    current_hypothesis: HypothesisDocument | None = None,
    max_round_reached: bool = False,
) -> ConclusionResult | None:
    raw = str(text or "").strip()
    if raw.startswith("{") and raw.endswith("}") and "(" in raw[:3]:
        raw = raw[1:-1].strip()
    if not (raw.startswith("(") and raw.endswith(")")):
        return None
    pythonish = (
        raw.replace(": null", ": None")
        .replace(": true", ": True")
        .replace(": false", ": False")
        .replace(", null", ", None")
        .replace(", true", ", True")
        .replace(", false", ", False")
    )
    try:
        payload = ast.literal_eval(pythonish)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(payload, tuple) or len(payload) < 2:
        return None
    values = list(payload)
    action_aliases = {"retrieve", "answer", "direct_answer"}
    action_index = next(
        (
            index
            for index, value in enumerate(values[:2])
            if str(value).strip() in RETRIEVAL_ACTIONS | action_aliases
        ),
        None,
    )
    if action_index is None:
        return None
    next_action = str(values[action_index]).strip()
    next_action = {
        "retrieve": "retrieve_more",
        "answer": "answer_directly",
        "direct_answer": "answer_directly",
    }.get(next_action, next_action)
    tail = values[action_index + 1 :]
    answer = str(tail[0] if len(tail) >= 1 and tail[0] is not None else "").strip()
    clarification_question = str(tail[1] if len(tail) >= 2 and tail[1] is not None else "").strip()
    missing_slots = normalize_string_list(tail[2] if len(tail) >= 3 else [], limit=8)
    follow_up_payload = next((item for item in tail if isinstance(item, dict)), None)

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
    follow_up_hypothesis = None
    if isinstance(follow_up_payload, dict):
        try:
            follow_up_hypothesis = normalize_hypothesis_payload(
                follow_up_payload,
                question=question,
                dialogue_context=dialogue_context,
                current_intent=current_intent,
            )
        except ModelOutputError:
            follow_up_hypothesis = None
    if follow_up_hypothesis is None:
        follow_up_hypothesis = (
            build_heuristic_follow_up_hypothesis(question, current_hypothesis, missing_slots)
            if current_hypothesis is not None
            else build_hypothesis(question + " " + " ".join(missing_slots[:4]), dialogue_context)
        )
    return ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=dedupe_keep_order([*missing_slots, "tuple-like 结论已转换为续检索"]),
        clarification_question="",
        follow_up_hypothesis=follow_up_hypothesis,
    )
