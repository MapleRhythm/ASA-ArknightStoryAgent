from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.common.lexicon import LEGACY_INTENT_MAP
from asa_arknight_story_agent.inference.payload.utils import normalize_string_list
from asa_arknight_story_agent.inference.pipeline.constants import (
    FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS,
    HYPOTHESIS_INTENTS,
    INITIAL_HYPOTHESIS_SCHEMA_FIELDS,
    QUERY_TYPES,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument, ModelOutputError
from asa_arknight_story_agent.inference.planning.query_understanding import (
    detect_intent,
    expand_entities_with_aliases,
    expand_related_retrieval_terms,
    extract_content_tokens,
    extract_entities,
    infer_query_type,
)
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def normalize_hypothesis_payload(
    payload: dict[str, Any],
    *,
    question: str,
    dialogue_context: str,
    current_intent: str | None = None,
) -> HypothesisDocument:
    is_follow_up = current_intent is not None
    allowed_fields = FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS if is_follow_up else INITIAL_HYPOTHESIS_SCHEMA_FIELDS
    allowed_keys = set(allowed_fields)
    if is_follow_up:
        # Be tolerant here: some model outputs still echo `intent` even though
        # follow-up prompts ask it to inherit the previous round's intent.
        allowed_keys.add("intent")
    extra_keys = set(payload) - allowed_keys
    if extra_keys:
        raise ModelOutputError(f"unexpected hypothesis fields: {sorted(extra_keys)}")
    optional_missing_fields = {"dialogue_context", "query_type", "expected_answer_type", "reflect_tokens"}
    if not is_follow_up:
        optional_missing_fields.update({"question", "intent"})
    missing_fields = [
        field
        for field in allowed_fields
        if field not in payload and field not in optional_missing_fields
    ]
    if missing_fields:
        raise ModelOutputError(f"missing hypothesis fields: {missing_fields}")

    inferred_intent, inferred_answer_type = detect_intent(question)
    intent = current_intent or str(payload.get("intent", "")).strip() or inferred_intent
    intent = LEGACY_INTENT_MAP.get(intent, intent)
    if intent not in HYPOTHESIS_INTENTS:
        raise ModelOutputError(f"invalid hypothesis intent: {intent or '<empty>'}")

    entities = normalize_string_list(payload.get("entities"), limit=12)
    if not entities:
        raise ModelOutputError("hypothesis must contain non-empty entities")

    keywords = normalize_string_list(payload.get("keywords"), limit=20)
    if not keywords:
        raise ModelOutputError("hypothesis must contain non-empty keywords")
    heuristic_entities = extract_entities(question, dialogue_context)
    heuristic_keywords = extract_content_tokens(question)
    entities = dedupe_keep_order(entities + heuristic_entities)[:12]
    keywords = dedupe_keep_order(
        keywords
        + heuristic_keywords
        + expand_related_retrieval_terms(entities + keywords + heuristic_keywords)
    )[:24]

    expected_answer_type = str(payload.get("expected_answer_type", "")).strip() or inferred_answer_type
    if not expected_answer_type:
        raise ModelOutputError("hypothesis must contain expected_answer_type")
    query_type = str(payload.get("query_type", "")).strip()
    if query_type not in QUERY_TYPES:
        query_type = infer_query_type(question, intent, expected_answer_type)

    alias_keywords = expand_entities_with_aliases(entities, keywords)
    if alias_keywords:
        keywords = dedupe_keep_order(keywords + alias_keywords)[:24]

    return HypothesisDocument(
        question=question,
        intent=intent,
        query_type=query_type,
        entities=entities,
        keywords=keywords,
        expected_answer_type=expected_answer_type,
        dialogue_context=dialogue_context.strip(),
    )
