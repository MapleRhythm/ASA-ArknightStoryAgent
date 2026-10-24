import copy

import pytest

from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload
from asa_arknight_story_agent.inference.payload.utils import compact_inferred_facts_payload
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


QUESTION = "桌旁发生了什么？"
VISIBLE = "[E1] 桌面上的茶杯仍然温热。\n[E2] 桌旁的椅子还有刚刚移动过的痕迹。"
HYPOTHESIS = HypothesisDocument(
    question=QUESTION, intent="plot_fact", query_type="fact", entities=["茶杯"],
    keywords=["桌旁"], expected_answer_type="行为", dialogue_context="",
)


def validate(inferred, *, visible=VISIBLE, final=False):
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{
            "fact": "桌面上的茶杯仍然温热。",
            "evidence_refs": [{"evidence_id": "E1", "quote": "桌面上的茶杯仍然温热。"}],
        }],
        "inferred_facts": inferred, "final_answer": "桌面上的茶杯仍然温热。可能刚有人离开桌旁。",
    }
    result = normalize_conclusion_payload(
        payload, question=QUESTION, dialogue_context="", current_intent=HYPOTHESIS.intent,
        current_hypothesis=HYPOTHESIS, max_round_reached=final,
    )
    return validate_conclusion_grounding(
        question=QUESTION, hypothesis=HYPOTHESIS,
        evidence=[{"id": "hidden", "text": VISIBLE}], conclusion=result,
        max_round_reached=final, mode="quote", evidence_prompt_text=visible,
    )


def cited_inference():
    return [{
        "fact": "可能刚有人离开桌旁。",
        "evidence_refs": [{"evidence_id": "E2", "quote": "桌旁的椅子还有刚刚移动过的痕迹。"}],
    }]


def test_parser_preserves_inference_references_without_mutating_input():
    raw = cited_inference()
    before = copy.deepcopy(raw)
    assert compact_inferred_facts_payload(raw) == raw
    assert raw == before


@pytest.mark.parametrize("final,action", [(False, "retrieve_more"), (True, "abstain")])
def test_uncited_inference_is_not_promoted_to_a_confirmed_answer(final, action):
    result = validate([{"fact": "角色乙已经投降。"}], final=final)
    assert result.next_action == action
    assert "角色乙已经投降" not in result.answer
    assert "inferred_fact_1_missing_evidence_refs" in result.grounding_warnings


def test_fake_reference_is_preserved_and_rejected_not_silently_dropped():
    raw = cited_inference()
    raw[0]["evidence_refs"][0]["evidence_id"] = "E99"
    result = validate(raw)
    assert result.next_action == "retrieve_more"
    assert "inferred_fact_1_ref_1_evidence_id_not_found:E99" in result.grounding_warnings


def test_hidden_source_cannot_support_an_inferred_reference():
    result = validate(cited_inference(), visible="[E1] 桌面上的茶杯仍然温热。")
    assert result.next_action == "retrieve_more"
    assert "inferred_fact_1_ref_1_evidence_id_not_found:E2" in result.grounding_warnings


def test_reference_checked_inference_still_has_explicit_non_fact_label():
    result = validate(cited_inference())
    assert result.next_action == "answer_directly"
    assert "基于上述证据的推断（非直接事实）：可能刚有人离开桌旁。" in result.answer
    assert "可能刚有人离开桌旁" not in result.answer.split("\n", 1)[0]


def test_inferred_quote_is_not_truncated_and_must_match():
    raw = cited_inference()
    raw[0]["evidence_refs"][0]["quote"] = "原" * 81
    result = validate(raw)
    assert result.next_action == "retrieve_more"
    assert "inferred_fact_1_ref_1_quote_over_80" in result.grounding_warnings
    assert "inferred_fact_1_ref_1_quote_not_found" in result.grounding_warnings


def test_legacy_bare_inference_text_has_no_citation_bypass():
    assert validate(["角色乙已经投降。"]).next_action == "retrieve_more"


def test_supported_fact_cannot_borrow_inference_quote_for_sensitive_relation():
    question = "甲与乙是什么关系？"
    hyp = HypothesisDocument(
        question=question, intent="plot_fact", query_type="relation",
        entities=["甲", "乙"], keywords=["关系"], expected_answer_type="关系", dialogue_context="",
    )
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{
            "fact": "甲是乙的母亲。",
            "evidence_refs": [{"evidence_id": "E1", "quote": "甲与乙在门口碰面。"}],
        }],
        "inferred_facts": [{
            "fact": "可能她们是亲人。",
            "evidence_refs": [{"evidence_id": "E2", "quote": "丙是丁的母亲。"}],
        }],
        "final_answer": "甲是乙的母亲。可能她们是亲人。",
    }
    parsed = normalize_conclusion_payload(
        payload, question=question, dialogue_context="", current_intent=hyp.intent,
        current_hypothesis=hyp, max_round_reached=False,
    )
    result = validate_conclusion_grounding(
        question=question, hypothesis=hyp, evidence=[], conclusion=parsed, max_round_reached=False,
        mode="quote", evidence_prompt_text="[E1] 甲与乙在门口碰面。\n[E2] 丙是丁的母亲。",
    )
    assert result.next_action == "retrieve_more"
    assert "supported_fact_1_has_terms_outside_quotes:母亲" in result.grounding_warnings
