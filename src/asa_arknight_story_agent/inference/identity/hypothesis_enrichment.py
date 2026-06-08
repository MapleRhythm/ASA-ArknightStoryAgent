from __future__ import annotations

from asa_arknight_story_agent.inference.common.lexicon import (
    COMMON_NON_ENTITY_WORDS,
    RELATION_TERMS,
    TITLE_TERMS,
)
from asa_arknight_story_agent.inference.common.patterns import QUESTION_TOKEN_RE
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def enrich_hypothesis(
    hypothesis: HypothesisDocument,
    bridge_terms: list[str],
    follow_up_queries: list[str],
) -> HypothesisDocument:
    extra_entities = [
        term
        for term in bridge_terms
        if term not in RELATION_TERMS and term not in TITLE_TERMS and len(term) <= 6
    ]
    extra_keywords = bridge_terms + [
        token
        for query in follow_up_queries
        for token in QUESTION_TOKEN_RE.findall(query)
        if token not in COMMON_NON_ENTITY_WORDS
    ]
    return HypothesisDocument(
        question=hypothesis.question,
        intent=hypothesis.intent,
        query_type=hypothesis.query_type,
        entities=dedupe_keep_order(hypothesis.entities + extra_entities)[:12],
        keywords=dedupe_keep_order(hypothesis.keywords + extra_keywords)[:20],
        expected_answer_type=hypothesis.expected_answer_type,
        dialogue_context=hypothesis.dialogue_context,
    )
