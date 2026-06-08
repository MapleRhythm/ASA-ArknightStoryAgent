from __future__ import annotations

import re

from asa_arknight_story_agent.inference.anchors.terms import clean_anchor_term
from asa_arknight_story_agent.inference.grounding.identity import primary_entity_anchor_required
from asa_arknight_story_agent.inference.common.lexicon import COMMON_NON_ENTITY_WORDS
from asa_arknight_story_agent.inference.pipeline.constants import NOISY_RETRIEVAL_TOKENS
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


EVENT_REFERENCE_NOISY_TERMS = {"事件", "关系", "具体", "指什么", "是什么", "发生"}


def event_reference_anchor_terms(question: str, hypothesis: HypothesisDocument) -> list[str]:
    raw_terms: list[str] = []
    primary = primary_entity_anchor_required(question, hypothesis)
    if primary:
        raw_terms.append(primary)
    else:
        raw_terms.extend(term for term in hypothesis.entities if term and term in question)
        if not raw_terms:
            raw_terms.extend(extract_content_tokens(question))

    anchors: list[str] = []
    for raw_term in raw_terms:
        term = clean_anchor_term(str(raw_term or ""))
        if term.endswith("一事"):
            term = term[:-2]
        if (
            len(term) < 2
            or term in COMMON_NON_ENTITY_WORDS
            or term in NOISY_RETRIEVAL_TOKENS
            or term in EVENT_REFERENCE_NOISY_TERMS
        ):
            continue
        anchors.append(term)
    return dedupe_keep_order([re.sub(r"\s+", "", anchor) for anchor in anchors if anchor])[:8]
