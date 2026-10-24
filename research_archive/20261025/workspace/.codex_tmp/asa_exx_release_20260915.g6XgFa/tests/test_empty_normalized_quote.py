"""A string erased by quote normalization is not evidence for an answer."""
import copy

import pytest

from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument

QUESTION = "角色甲在做什么？"
VISIBLE = "[E1] 角色甲在窗边读书。\n[E2] 天色渐晚。"
HYPOTHESIS = HypothesisDocument(
    question=QUESTION, intent="plot_fact", query_type="fact",
    entities=["角色甲"], keywords=["行为"], expected_answer_type="行为",
)
EMPTY_AFTER_NORMALIZATION = ["...", "…", "……", "[E1]", "[E2]...", "[E1]\n…", "... [E12] …"]


def make_conclusion(quote, group="supported", final=False):
    suspicious = {
        "fact": "角色甲在唱歌。",
        "evidence_refs": [{"evidence_id": "E1", "quote": quote}],
    }
    good = {
        "fact": "角色甲在窗边读书。",
        "evidence_refs": [{"evidence_id": "E1", "quote": "角色甲在窗边读书。"}],
    }
    payload = {
        "next_action": "answer_directly", "final_answer": "角色甲在唱歌。",
        "supported_facts": [suspicious] if group == "supported" else [good],
        "inferred_facts": [suspicious] if group == "inferred" else [],
    }
    return normalize_conclusion_payload(
        payload, question=QUESTION, dialogue_context="", current_intent=HYPOTHESIS.intent,
        current_hypothesis=HYPOTHESIS, max_round_reached=final,
    )


@pytest.mark.parametrize("quote", EMPTY_AFTER_NORMALIZATION)
@pytest.mark.parametrize("group", ["supported", "inferred"])
@pytest.mark.parametrize("final", [False, True])
def test_normalization_empty_quote_cannot_support_fact_or_inference(quote, group, final):
    conclusion = make_conclusion(quote, group, final)
    before = copy.deepcopy(conclusion)
    result = validate_conclusion_grounding(
        question=QUESTION, hypothesis=HYPOTHESIS, evidence=[], conclusion=conclusion,
        max_round_reached=final, mode="quote", evidence_prompt_text=VISIBLE,
    )
    assert result.next_action == ("abstain" if final else "retrieve_more")
    assert "角色甲在唱歌" not in result.answer
    assert f"{group}_fact_1_ref_1_quote_empty_after_normalization" in result.grounding_warnings
    assert conclusion == before


@pytest.mark.parametrize("quote", ["角色甲在窗边读书。", "[E1] 角色甲在窗边读书。", "…角色甲在窗边读书。…"])
def test_existing_nonempty_normalization_behavior_is_preserved(quote):
    conclusion = make_conclusion(quote)
    conclusion.answer = conclusion.supported_facts[0]["fact"] = "角色甲在窗边读书。"
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion, question=QUESTION, evidence=[], evidence_prompt_text=VISIBLE,
    )
    assert issues == []


def test_nonempty_single_character_reference_is_not_arbitrarily_length_rejected():
    conclusion = make_conclusion("7")
    conclusion.answer = conclusion.supported_facts[0]["fact"] = "7"
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion, question=QUESTION, evidence=[], evidence_prompt_text="[E1] 7",
    )
    assert issues == []


def test_empty_quote_error_does_not_hide_missing_evidence_id():
    conclusion = make_conclusion("[E1]")
    conclusion.supported_facts[0]["evidence_refs"][0]["evidence_id"] = "E999"
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion, question=QUESTION, evidence=[], evidence_prompt_text=VISIBLE,
    )
    assert "supported_fact_1_ref_1_evidence_id_not_found:E999" in issues
    assert "supported_fact_1_ref_1_quote_empty_after_normalization" in issues


def test_failed_reference_is_not_salvaged_by_good_sibling_reference():
    conclusion = make_conclusion("...")
    conclusion.supported_facts[0]["evidence_refs"].append({
        "evidence_id": "E1", "quote": "角色甲在窗边读书。",
    })
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion, question=QUESTION, evidence=[], evidence_prompt_text=VISIBLE,
    )
    assert "supported_fact_1_ref_1_quote_empty_after_normalization" in issues
