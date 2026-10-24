"""Experimental final-entity projection; no production module mutation."""
from dataclasses import replace


def project_model_entities(control, cleaned_model_entities):
    """Keep all other normalized fields, including expanded keywords, unchanged."""
    if not cleaned_model_entities:
        raise ValueError("valid non-empty cleaned model entities required")
    if len(cleaned_model_entities) > 12:
        raise ValueError("model entity limit exceeded")
    if any(not isinstance(term, str) or not term for term in cleaned_model_entities):
        raise ValueError("invalid cleaned model entity")
    if len(set(cleaned_model_entities)) != len(cleaned_model_entities):
        raise ValueError("cleaned model entities must be deduplicated")
    if list(control.entities[:len(cleaned_model_entities)]) != list(cleaned_model_entities):
        raise ValueError("control no longer preserves the model-entity prefix")
    return replace(control, entities=list(cleaned_model_entities))
