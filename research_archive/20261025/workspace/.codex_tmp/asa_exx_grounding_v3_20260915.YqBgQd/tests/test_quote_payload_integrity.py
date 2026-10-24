import copy

import pytest

from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
from asa_arknight_story_agent.inference.payload.utils import compact_supported_facts_payload
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult


def conclusion(facts):
    return ConclusionResult(
        next_action="answer_directly", answer="角色甲在读书。", missing_slots=[],
        clarification_question="", follow_up_hypothesis=None, supported_facts=facts,
    )


@pytest.mark.parametrize("length", [20, 80, 81, 128, 401])
def test_normalization_preserves_complete_emitted_quote(length):
    text = "原" * length
    source = [{"fact": "角色甲在读书。", "evidence_refs": [{"evidence_id": "E1", "quote": text}]}]
    frozen = copy.deepcopy(source)
    result = compact_supported_facts_payload(source)
    assert source == frozen
    assert result[0]["evidence_refs"][0]["quote"] == text


def test_overlong_false_suffix_cannot_be_laundered_by_truncation():
    visible_quote = "角色甲在窗边读书。" * 10
    prefix = visible_quote[:80]
    emitted = prefix + "角色乙已经投降。"
    facts = compact_supported_facts_payload([
        {"fact": "角色甲在读书。", "evidence_refs": [{"evidence_id": "E1", "quote": emitted}]}
    ])
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion(facts), evidence=[], question="角色甲在做什么？",
        evidence_prompt_text="[E1] " + visible_quote,
    )
    assert "supported_fact_1_ref_1_quote_over_80" in issues
    assert "supported_fact_1_ref_1_quote_not_found" in issues


def test_valid_eighty_character_quote_still_passes():
    quote = "原" * 80
    facts = compact_supported_facts_payload([
        {"fact": "原", "evidence_refs": [{"evidence_id": "E1", "quote": quote}]}
    ])
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion(facts), evidence=[], question="原文是什么？",
        evidence_prompt_text="[E1] " + quote,
    )
    assert issues == []


def test_evidence_id_payload_is_unchanged():
    assert compact_supported_facts_payload([{"fact": "事实", "evidence_ids": ["E1", "E2"]}]) == [
        {"fact": "事实", "evidence_refs": [{"evidence_id": "E1"}, {"evidence_id": "E2"}]}
    ]
