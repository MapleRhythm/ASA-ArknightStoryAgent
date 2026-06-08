from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import build_heuristic_follow_up_hypothesis
from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument, ModelOutputError
from asa_arknight_story_agent.inference.planning.query_understanding import build_hypothesis
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def resolve_retrieve_more_follow_up(
    *,
    question: str,
    dialogue_context: str,
    current_intent: str,
    current_hypothesis: HypothesisDocument | None,
    follow_up_hypothesis_payload: Any,
    missing_slots: list[str],
    max_round_reached: bool,
) -> tuple[str, str, list[str], HypothesisDocument | None]:
    answer = ""
    if not missing_slots:
        missing_slots = ["需要补充更直接的证据"]
    follow_up_hypothesis: HypothesisDocument | None = None
    if isinstance(follow_up_hypothesis_payload, dict):
        try:
            follow_up_hypothesis = normalize_hypothesis_payload(
                follow_up_hypothesis_payload,
                question=question,
                dialogue_context=dialogue_context,
                current_intent=current_intent,
            )
        except ModelOutputError:
            if not max_round_reached:
                follow_up_hypothesis = build_fallback_follow_up_hypothesis(
                    question=question,
                    dialogue_context=dialogue_context,
                    current_hypothesis=current_hypothesis,
                    missing_slots=missing_slots,
                )
                missing_slots = dedupe_keep_order(
                    [*missing_slots, "follow_up_hypothesis 不可用，已使用启发式续检索"]
                )
    elif not max_round_reached:
        follow_up_hypothesis = build_fallback_follow_up_hypothesis(
            question=question,
            dialogue_context=dialogue_context,
            current_hypothesis=current_hypothesis,
            missing_slots=missing_slots,
        )
        missing_slots = dedupe_keep_order([*missing_slots, "模型未返回 follow_up_hypothesis，已使用启发式续检索"])
    if max_round_reached:
        return "abstain", "现有检索证据不足以确认，且已达到检索轮次上限。", missing_slots, None
    return "retrieve_more", answer, missing_slots, follow_up_hypothesis


def build_fallback_follow_up_hypothesis(
    *,
    question: str,
    dialogue_context: str,
    current_hypothesis: HypothesisDocument | None,
    missing_slots: list[str],
) -> HypothesisDocument:
    if current_hypothesis is not None:
        return build_heuristic_follow_up_hypothesis(question, current_hypothesis, missing_slots)
    return build_hypothesis(question + " " + " ".join(missing_slots[:4]), dialogue_context)
