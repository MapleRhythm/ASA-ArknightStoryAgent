from copy import deepcopy
from dataclasses import asdict, dataclass
import pytest

from intent_field_fallback import normalize_with_intent_fallback, preserve_fallback_routing


class Invalid(ValueError):
    pass


INTENTS = {"plot_fact", "plot_reasoning", "character_relation"}
QUERY_TYPES = {"fact", "relation", "mystery", "reveal"}


def normalize(payload, *, question, dialogue_context, current_intent=None):
    if set(payload) - {"intent", "entities", "keywords"}:
        raise Invalid("unexpected hypothesis fields")
    intent = current_intent or payload["intent"]
    if intent not in INTENTS:
        raise Invalid(f"invalid hypothesis intent: {intent}")
    if not payload.get("entities"):
        raise Invalid("hypothesis must contain non-empty entities")
    if not payload.get("keywords"):
        raise Invalid("hypothesis must contain non-empty keywords")
    return {**deepcopy(payload), "intent": intent, "question": question, "dialogue_context": dialogue_context}


def call(payload, **kwargs):
    return normalize_with_intent_fallback(
        payload, normalize=normalize, error_type=Invalid,
        detect_intent=lambda question: ("plot_fact", "事实问答"),
        hypothesis_intents=INTENTS, query_types=QUERY_TYPES, question="甲乙是谁？", **kwargs,
    )


def payload(intent="plot_fact"):
    return {"intent": intent, "entities": ["甲乙"], "keywords": ["身份"]}


def test_valid_payload_unchanged_without_diagnostic():
    data = payload()
    result, diagnostic = call(data)
    assert result == normalize(data, question="甲乙是谁？", dialogue_context="")
    assert diagnostic is None


@pytest.mark.parametrize("wrong", sorted(QUERY_TYPES))
def test_query_enum_replaced_by_existing_fallback_intent_and_error_retained(wrong):
    data = payload(wrong)
    original = deepcopy(data)
    result, diagnostic = call(data)
    assert data == original
    assert result["entities"] == ["甲乙"] and result["keywords"] == ["身份"]
    assert result["intent"] == "plot_fact"
    assert diagnostic["raw_intent"] == wrong
    assert diagnostic["original_error"] == f"invalid hypothesis intent: {wrong}"
    assert diagnostic["raw_payload_modified"] is False


@pytest.mark.parametrize("wrong", ["unknown", "回答答案", "", "MYSTERY"])
def test_arbitrary_invalid_intents_not_accepted(wrong):
    with pytest.raises(Invalid):
        call(payload(wrong))


@pytest.mark.parametrize("field", ["entities", "keywords"])
def test_invalid_other_field_still_rejected_after_intent_recovery(field):
    data = payload("mystery")
    data[field] = []
    with pytest.raises(Invalid, match=f"non-empty {field}"):
        call(data)


def test_extra_schema_key_not_recovered():
    data = {**payload("mystery"), "final_answer": "not allowed"}
    with pytest.raises(Invalid, match="unexpected hypothesis fields"):
        call(data)


def test_follow_up_inherits_original_current_intent_without_repair():
    data = payload("mystery")
    result, diagnostic = call(data, current_intent="plot_reasoning")
    assert result["intent"] == "plot_reasoning"
    assert diagnostic is None


def test_follow_up_invalid_current_intent_not_repaired():
    with pytest.raises(Invalid):
        call(payload("mystery"), current_intent="bad_current")


@dataclass
class Hypothesis:
    entities: list
    keywords: list
    query_type: str = "mystery"
    expected_answer_type: str = "事实问答"
    intent: str = "plot_fact"
    question: str = "甲乙是谁？"
    dialogue_context: str = ""


def test_conservative_recovery_preserves_all_routing_and_existing_cues():
    control = Hypothesis(["碎片", "旧实体"], ["旧线索"])
    recovered = Hypothesis(["甲乙", "碎片"], ["新线索"], query_type="fact", expected_answer_type="name")
    before = asdict(control)
    result, diagnostic = preserve_fallback_routing(control, recovered)
    assert result.entities == ["甲乙", "碎片", "旧实体"]
    assert result.keywords == ["新线索", "旧线索"]
    for key in before.keys() - {"entities", "keywords"}:
        assert asdict(result)[key] == before[key]
    assert asdict(control) == before
    assert diagnostic["all_control_cues_preserved"]


@pytest.mark.parametrize("field", ["entities", "keywords"])
def test_capacity_overflow_keeps_original_fallback_without_truncation(field):
    control = Hypothesis(["旧实体"], ["旧线索"])
    recovered = Hypothesis(["甲乙"], ["身份"])
    setattr(recovered, field, [f"词{index}" for index in range(30)])
    result, diagnostic = preserve_fallback_routing(control, recovered)
    assert result is control
    assert diagnostic == {"cue_merge_applied": False, "reason": "would_drop_existing_cues_at_capacity"}
