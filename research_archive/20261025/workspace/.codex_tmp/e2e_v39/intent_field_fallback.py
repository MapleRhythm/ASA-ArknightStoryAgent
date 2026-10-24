"""Experimental initial-planner field fallback with an explicit audit sidecar."""
from copy import deepcopy
from dataclasses import replace


def normalize_with_intent_fallback(
    payload, *, normalize, error_type, detect_intent, hypothesis_intents,
    query_types, question, dialogue_context="", current_intent=None,
):
    """Never mutate raw payload or broaden allowed intent enums."""
    arguments = {"question": question, "dialogue_context": dialogue_context}
    if current_intent is not None:
        arguments["current_intent"] = current_intent
    try:
        return normalize(payload, **arguments), None
    except error_type as original:
        raw_intent = payload.get("intent")
        if (
            current_intent is not None
            or not isinstance(raw_intent, str)
            or raw_intent not in query_types
            or raw_intent in hypothesis_intents
            or not str(original).startswith("invalid hypothesis intent:")
        ):
            raise
        replacement, _ = detect_intent(question)
        if replacement not in hypothesis_intents:
            raise
        copied = deepcopy(payload)
        copied["intent"] = replacement
        # The original strict normalizer validates EVERY field again. A second
        # error remains an error; do not accept arbitrary malformed payloads.
        result = normalize(copied, **arguments)
        return result, {
            "kind": "initial_planner_query_enum_in_intent_field",
            "original_error": str(original),
            "raw_intent": raw_intent,
            "replacement_intent": replacement,
            "replacement_source": "existing_detect_intent_question_heuristic",
            "raw_payload_modified": False,
            "canonical_validation_rerun": True,
        }


def preserve_fallback_routing(control, recovered, *, entity_limit=12, keyword_limit=24):
    """Retain every old cue and all fallback routing fields, or do not recover."""
    entities = list(dict.fromkeys([*recovered.entities, *control.entities]))
    keywords = list(dict.fromkeys([*recovered.keywords, *control.keywords]))
    if len(entities) > entity_limit or len(keywords) > keyword_limit:
        return control, {"cue_merge_applied": False, "reason": "would_drop_existing_cues_at_capacity"}
    return replace(control, entities=entities, keywords=keywords), {
        "cue_merge_applied": True, "all_control_cues_preserved": True,
        "routing_fields_from_canonical_full_fallback": True,
    }
