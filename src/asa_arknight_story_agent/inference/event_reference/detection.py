from __future__ import annotations

import re

from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


def is_event_reference_question(question: str, hypothesis: HypothesisDocument) -> bool:
    compact = re.sub(r"\s+", "", "\n".join([question or "", hypothesis.question or "", hypothesis.expected_answer_type or ""]))
    if any(marker in compact for marker in ("一事", "这件事", "此事", "具体是指", "指的是什么", "指什么", "发生了什么")):
        return True
    return False
