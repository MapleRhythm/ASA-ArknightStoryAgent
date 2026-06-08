from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.event_reference.detection import is_event_reference_question
from asa_arknight_story_agent.inference.event_reference.strips import select_event_reference_strips
from asa_arknight_story_agent.inference.event_reference.terms import event_reference_anchor_terms
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument

__all__ = [
    "is_event_reference_question",
    "event_reference_anchor_terms",
    "select_event_reference_strips",
    "build_event_reference_answer",
]


def build_event_reference_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> str | None:
    anchor, strips = select_event_reference_strips(question=question, hypothesis=hypothesis, evidence=evidence)
    if not anchor or not strips:
        return None

    answer = f"{anchor}一事，现有证据可确认的是：{strips[0]}"
    if len(strips) > 1:
        answer += " 相关证据还显示：" + "；".join(strips[1:])
    return answer
