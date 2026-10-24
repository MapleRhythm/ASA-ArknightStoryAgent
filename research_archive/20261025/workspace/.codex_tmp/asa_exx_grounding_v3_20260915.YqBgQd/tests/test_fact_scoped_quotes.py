"""A fact must not borrow sensitive relation terms from a sibling fact."""
import copy

import pytest

from asa_arknight_story_agent.inference.grounding.grounded_fact_answers import answer_from_grounded_facts
from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


QUESTION = "甲与乙是什么关系？"
VISIBLE = "[E1] 甲与乙在门口碰面。\n[E2] 丙是丁的母亲。\n[E3] 甲是乙的母亲。"
HYPOTHESIS = HypothesisDocument(
    question=QUESTION, intent="plot_fact", query_type="relation",
    entities=["甲", "乙"], keywords=["关系"], expected_answer_type="关系",
)


def fact(text, evidence_id, quote):
    return {"fact": text, "evidence_refs": [{"evidence_id": evidence_id, "quote": quote}]}


def payload(group, reverse=False, valid=False):
    bad = fact("甲是乙的母亲。", "E1", "甲与乙在门口碰面。")
    if valid:
        bad["evidence_refs"].append({"evidence_id": "E3", "quote": "甲是乙的母亲。"})
    sibling = fact("丙是丁的母亲。", "E2", "丙是丁的母亲。")
    facts = [sibling, bad] if reverse else [bad, sibling]
    return {
        "next_action": "answer_directly",
        "supported_facts": facts if group == "supported" else [
            fact("甲与乙在门口碰面。", "E1", "甲与乙在门口碰面。"),
        ],
        "inferred_facts": facts if group == "inferred" else [],
        "final_answer": "甲是乙的母亲。",
    }


def parse(value, final=False):
    return normalize_conclusion_payload(
        value, question=QUESTION, dialogue_context="", current_intent=HYPOTHESIS.intent,
        current_hypothesis=HYPOTHESIS, max_round_reached=final,
    )


def validate(value, final=False):
    return validate_conclusion_grounding(
        question=QUESTION, hypothesis=HYPOTHESIS, evidence=[], conclusion=parse(value, final),
        max_round_reached=final, mode="quote", evidence_prompt_text=VISIBLE,
    )


@pytest.mark.parametrize("group", ["supported", "inferred"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("final", [False, True])
def test_fact_cannot_borrow_sibling_quote_or_uncited_visible_block(group, reverse, final):
    result = validate(payload(group, reverse), final)
    assert result.next_action == ("abstain" if final else "retrieve_more")
    assert "甲是乙的母亲" not in result.answer
    index = 2 if reverse else 1
    assert f"{group}_fact_{index}_has_terms_outside_quotes:母亲" in result.grounding_warnings


@pytest.mark.parametrize("group", ["supported", "inferred"])
def test_rendering_checks_each_facts_own_quotes(group):
    result = answer_from_grounded_facts(parse(payload(group)))
    assert "甲是乙的母亲" not in result
    assert "丙是丁的母亲" in result


@pytest.mark.parametrize("group", ["supported", "inferred"])
def test_fact_with_its_own_supporting_second_reference_is_kept(group):
    value = payload(group, valid=True)
    original = copy.deepcopy(value)
    result = validate(value)
    assert result.next_action == "answer_directly"
    assert "甲是乙的母亲" in result.answer
    assert value == original


def test_multiple_supported_facts_with_own_quotes_remain_valid():
    value = {
        "next_action": "answer_directly",
        "supported_facts": [
            fact("甲与乙在门口碰面。", "E1", "甲与乙在门口碰面。"),
            fact("丙是丁的母亲。", "E2", "丙是丁的母亲。"),
        ],
        "inferred_facts": [], "final_answer": "甲与乙在门口碰面。丙是丁的母亲。",
    }
    result = validate(value)
    assert result.next_action == "answer_directly"
    assert result.answer == value["final_answer"]
