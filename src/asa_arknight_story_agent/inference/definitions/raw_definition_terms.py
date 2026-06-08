from __future__ import annotations

from asa_arknight_story_agent.inference.anchors.terms import clean_anchor_term, extract_action_targets
from asa_arknight_story_agent.inference.common.lexicon import COMMON_NON_ENTITY_WORDS
from asa_arknight_story_agent.inference.pipeline.constants import NOISY_RETRIEVAL_TOKENS
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def raw_exact_anchor_terms(question: str, hypothesis: HypothesisDocument) -> list[str]:
    terms = dedupe_keep_order(
        [
            *hypothesis.entities,
            *extract_action_targets(question + "\n" + hypothesis.question),
            *extract_content_tokens(question),
        ]
    )
    anchors: list[str] = []
    for term in terms:
        cleaned = clean_anchor_term(term)
        if (
            not cleaned
            or cleaned in COMMON_NON_ENTITY_WORDS
            or cleaned in NOISY_RETRIEVAL_TOKENS
            or len(cleaned) == 1 and not cleaned.isascii()
        ):
            continue
        if cleaned.isascii() and len(cleaned) < 2:
            continue
        anchors.append(cleaned)
    return dedupe_keep_order(anchors)[:8]
