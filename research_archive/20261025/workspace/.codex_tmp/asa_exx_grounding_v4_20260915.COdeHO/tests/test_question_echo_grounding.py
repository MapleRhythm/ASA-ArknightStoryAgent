"""Explicit unanswered questions cannot become supported story facts."""
import copy
import pytest

from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument

QUESTION = "角色甲在做什么？"
FACT = "角色甲在窗边读书。"
VISIBLE = "[E1] " + FACT
HYPOTHESIS = HypothesisDocument(
    question=QUESTION, intent="plot_fact", query_type="fact",
    entities=["角色甲"], keywords=["行为"], expected_answer_type="行为",
)


def fact(text):
    return {"fact": text, "evidence_refs": [{"evidence_id": "E1", "quote": FACT}]}


def conclusion(text, group="supported"):
    return ConclusionResult(
        next_action="answer_directly", answer=FACT,
        missing_slots=[], clarification_question="", follow_up_hypothesis=None,
        supported_facts=[fact(text)] if group == "supported" else [fact(FACT)],
        inferred_facts=[fact(text)] if group == "inferred" else [],
    )


@pytest.mark.parametrize("text", [QUESTION, "角色甲 在做什么?", "角色甲在做什么。"])
@pytest.mark.parametrize("group", ["supported", "inferred"])
@pytest.mark.parametrize("final", [False, True])
def test_fact_or_inference_echo_is_rejected(text, group, final):
    original = conclusion(text, group)
    before = copy.deepcopy(original)
    result = validate_conclusion_grounding(
        question=QUESTION, hypothesis=HYPOTHESIS, evidence=[], conclusion=original,
        max_round_reached=final, mode="quote", evidence_prompt_text=VISIBLE,
    )
    assert result.next_action == ("abstain" if final else "retrieve_more")
    assert f"{group}_fact_1_repeats_question" in result.grounding_warnings
    assert original == before
    if result.follow_up_hypothesis:
        terms = result.follow_up_hypothesis.entities + result.follow_up_hypothesis.keywords
        assert not any("repeats_question" in term or "_fact_" in term for term in terms)


@pytest.mark.parametrize("final", [False, True])
def test_echo_final_answer_rebuilt_from_valid_facts(final):
    original = conclusion(FACT)
    original.answer = QUESTION
    result = validate_conclusion_grounding(
        question=QUESTION, hypothesis=HYPOTHESIS, evidence=[], conclusion=original,
        max_round_reached=final, mode="quote", evidence_prompt_text=VISIBLE,
    )
    assert result.next_action == "answer_directly"
    assert result.answer == FACT
    assert "final_answer_repeats_question" in result.grounding_warnings


@pytest.mark.parametrize("text", [
    FACT, "角色甲并非在唱歌。", "角色甲问了“谁在读书？”",
])
def test_real_statement_or_embedded_question_not_blanket_rejected(text):
    original = conclusion(text)
    original.answer = text
    original.supported_facts[0]["evidence_refs"][0]["quote"] = text
    issues, warnings = validate_grounded_quotes(
        conclusion=original, evidence=[], question=QUESTION, evidence_prompt_text="[E1] " + text,
    )
    assert not any("repeats_question" in issue for issue in issues + warnings)


@pytest.mark.parametrize("question", ["角色甲在窗边读书。", "角色甲", "《你是谁？》", ""])
def test_noninterrogative_requests_not_assumed_invalid_when_repeated(question):
    original = conclusion(question or FACT)
    original.answer = question or FACT
    issues, warnings = validate_grounded_quotes(
        conclusion=original, evidence=[], question=question, evidence_prompt_text=VISIBLE,
    )
    assert not any("repeats_question" in issue for issue in issues + warnings)


def test_good_sibling_does_not_certify_question_as_another_fact():
    original = conclusion(QUESTION)
    original.supported_facts.append(fact(FACT))
    issues, _ = validate_grounded_quotes(
        conclusion=original, evidence=[], question=QUESTION, evidence_prompt_text=VISIBLE,
    )
    assert "supported_fact_1_repeats_question" in issues


def test_echo_does_not_hide_other_reference_failures():
    original = conclusion(QUESTION)
    original.supported_facts[0]["evidence_refs"] = [{"evidence_id": "E999", "quote": "..."}]
    issues, _ = validate_grounded_quotes(
        conclusion=original, evidence=[], question=QUESTION, evidence_prompt_text=VISIBLE,
    )
    assert "supported_fact_1_repeats_question" in issues
    assert "supported_fact_1_ref_1_evidence_id_not_found:E999" in issues
    assert "supported_fact_1_ref_1_quote_empty_after_normalization" in issues
