from __future__ import annotations

from asa_arknight_story_agent.inference.planning.dialogue_context import sanitize_dialogue_context
from asa_arknight_story_agent.inference.planning.entity_extraction import (
    expand_entities_with_aliases,
    extract_entities,
)
from asa_arknight_story_agent.inference.planning.hypothesis_keywords import build_hypothesis_keywords
from asa_arknight_story_agent.inference.planning.intent_detection import detect_intent, infer_query_type
from asa_arknight_story_agent.inference.common.lexicon import LEGACY_INTENT_MAP
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.retrieval.query_rendering import build_retrieval_query

__all__ = [
    "detect_intent",
    "infer_query_type",
    "extract_entities",
    "expand_entities_with_aliases",
    "build_hypothesis",
    "build_retrieval_query",
]


def build_hypothesis(question: str, dialogue_context: str = "") -> HypothesisDocument:
    intent, answer_type = detect_intent(question)
    entities = extract_entities(question, dialogue_context)
    sanitized_context = sanitize_dialogue_context(dialogue_context)
    keywords = build_hypothesis_keywords(
        question=question,
        dialogue_context=dialogue_context,
        entities=entities,
        answer_type=answer_type,
    )
    return HypothesisDocument(
        question=question,
        intent=LEGACY_INTENT_MAP.get(intent, intent),
        query_type=infer_query_type(question, intent, answer_type),
        entities=entities,
        keywords=keywords,
        expected_answer_type=answer_type,
        dialogue_context=sanitized_context,
    )
