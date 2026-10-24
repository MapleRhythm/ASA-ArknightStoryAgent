"""Failing the final lexical check must not expose unvalidated raw snippets."""
from copy import deepcopy

import pytest

from asa_arknight_story_agent.inference.grounding import token_checks, token_flow
from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument

QUESTION = "角色甲做了什么？"
HYPOTHESIS = HypothesisDocument(
    question=QUESTION, intent="plot_fact", query_type="fact",
    entities=["角色甲"], keywords=["行动"], expected_answer_type="行为",
)
MISSING = ["不存在的词甲", "不存在的词乙", "不存在的词丙", "不存在的词丁"]
SAFE_REFUSAL = "现有检索证据不足以给出可校验的回答。"


def conclusion():
    return ConclusionResult(
        next_action="answer_directly", answer="角色甲在门口读书。",
        missing_slots=[], clarification_question="", follow_up_hypothesis=None,
        supported_facts=[{
            "fact": "角色甲在门口读书。",
            "evidence_refs": [{"evidence_id": "E1", "quote": "角色甲在门口读书。"}],
        }],
        generation_diagnostics={"request_id": "preserve_me"},
        grounding_warnings=["prior_warning"],
    )


def validate(evidence, *, final=True, long_tokens=None, pool="已有证据"):
    return token_checks.validate_token_grounding(
        question=QUESTION, hypothesis=HYPOTHESIS, evidence=evidence, conclusion=conclusion(),
        long_tokens=MISSING if long_tokens is None else long_tokens,
        evidence_pool=pool, max_round_reached=final,
    )


@pytest.mark.parametrize("evidence", [
    [],
    [{"document": {"clean_text": "角色甲其实在旅行。UNVALIDATED_SOURCE_FRAGMENT"}}],
    [{"document": {"clean_text": "角色甲：" + "未核验内容" * 500 + "UNVALIDATED_END"}}],
])
def test_final_rejection_only_shows_refusal_without_raw_source_fragments(evidence):
    frozen = deepcopy(evidence)
    result = validate(evidence)
    assert result.next_action == "abstain"
    assert result.answer == SAFE_REFUSAL
    assert result.supported_facts == [] and result.inferred_facts == []
    assert "当前证据只能确认" not in result.answer
    assert "UNVALIDATED" not in result.answer and "…" not in result.answer
    assert evidence == frozen


def test_rejection_preserves_prior_diagnostics_and_records_reason():
    result = validate([])
    assert result.grounding_warnings == ["prior_warning"]
    assert result.generation_diagnostics["request_id"] == "preserve_me"
    assert result.generation_diagnostics["grounding_status"] == "rejected_to_abstain"
    assert result.generation_diagnostics["token_grounding"] == {
        "hit_rate": 0.0, "required_hit_rate": 0.25, "missing_tokens": MISSING,
    }


def test_legacy_fragment_builder_is_not_called(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("Raw-snippet fallback was invoked")
    monkeypatch.setattr(token_checks, "build_grounded_fallback_answer", forbidden, raising=False)
    assert validate([]).answer == SAFE_REFUSAL


def test_before_final_round_retrieval_decision_remains_unchanged():
    result = validate([], final=False)
    assert result.next_action == "retrieve_more" and result.answer == ""
    assert result.follow_up_hypothesis.keywords == ["行动", *MISSING]
    assert result.missing_slots == MISSING


@pytest.mark.parametrize("tokens,pool", [
    (["存在的词甲", "存在的词乙", "不存在的词甲", "不存在的词乙"], "存在的词甲 存在的词乙"),
    (["不存在的词甲", "不存在的词乙", "不存在的词丙"], "已有证据"),
    (["已找到词甲", *MISSING], "已找到词甲 不存在的词甲"),
])
def test_existing_acceptance_thresholds_are_not_changed(tokens, pool):
    assert validate([], long_tokens=tokens, pool=pool) is None


def test_no_long_tokens_still_produces_no_rejection():
    assert validate([], long_tokens=[]) is None


@pytest.mark.parametrize("mode", ["quote", "grounded", "strict"])
def test_full_structured_path_cannot_resurrect_unvalidated_hidden_snippets(monkeypatch, mode):
    monkeypatch.setattr(token_flow, "grounding_extract_answer_tokens", lambda *args: MISSING)
    monkeypatch.setattr(token_flow, "validate_required_anchor", lambda **kwargs: None)
    monkeypatch.setattr(token_flow, "has_direct_causal_grounding", lambda **kwargs: False)
    evidence = [{"document": {
        "clean_text": "角色甲在门口读书。角色甲杀害了角色乙。UNVALIDATED_NOT_IN_PROMPT",
    }}]
    frozen = deepcopy(evidence)
    result = validate_conclusion_grounding(
        question=QUESTION, hypothesis=HYPOTHESIS, evidence=evidence, conclusion=conclusion(),
        max_round_reached=True, mode=mode, evidence_prompt_text="[E1] 角色甲在门口读书。",
    )
    assert result.next_action == "abstain" and result.answer == SAFE_REFUSAL
    assert result.generation_diagnostics["grounding_status"] == "rejected_to_abstain"
    assert evidence == frozen
