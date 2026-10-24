from __future__ import annotations

from asa_arknight_story_agent.inference.common.lexicon import REVEAL_QUERY_TERMS
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


def is_reveal_question(question: str, hypothesis: HypothesisDocument) -> bool:
    text = "\n".join([question or "", hypothesis.question or "", " ".join(hypothesis.keywords)])
    return hypothesis.query_type in {"reveal", "mystery"} or any(term in text for term in REVEAL_QUERY_TERMS)
