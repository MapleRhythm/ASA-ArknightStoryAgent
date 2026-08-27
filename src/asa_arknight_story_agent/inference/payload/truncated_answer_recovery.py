from __future__ import annotations

from asa_arknight_story_agent.inference.payload.json_like_fields import (
    extract_json_like_bare_field,
    extract_json_like_repeated_string_field,
    extract_json_like_string_field,
    extract_truncated_supported_facts,
)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult


def recover_truncated_grounded_answer(
    text: str,
    *,
    question: str,
    max_round_reached: bool = False,
) -> ConclusionResult | None:
    next_action = extract_json_like_bare_field(text, "next_action")
    next_action = {
        "answer": "answer_directly",
        "direct_answer": "answer_directly",
        "retrieve": "retrieve_more",
    }.get(next_action, next_action)
    if next_action != "answer_directly":
        return None

    final_answer = extract_json_like_string_field(text, "final_answer").strip()
    if final_answer:
        answer = final_answer
    else:
        facts = extract_json_like_repeated_string_field(text, "fact", limit=8)
        if not facts:
            return None
        # Use only completed fact strings from the truncated JSON. The regular
        # grounding guard still verifies the recovered answer against evidence.
        answer = "；".join(facts)
        if len(answer) > 800:
            answer = answer[:799].rstrip("；，。 ") + "。"

    if not answer:
        return None
    return ConclusionResult(
        next_action="answer_directly",
        answer=answer,
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
        supported_facts=extract_truncated_supported_facts(text, limit=8),
        grounding_warnings=["recovered_from_truncated_json"],
    )
