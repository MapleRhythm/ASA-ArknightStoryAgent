from __future__ import annotations

import pytest

from asa_arknight_story_agent.inference.grounding.evidence_id_validation import (
    validate_evidence_id_grounding,
)
from asa_arknight_story_agent.inference.grounding.quote_validation import (
    validate_grounded_quotes,
)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.grounding import validation


TEXT = "阿米娅检查了报告。"
EVIDENCE = [{"document": {"id": "story#1", "clean_text": TEXT}}]


def conclusion(eid: str = "E1") -> ConclusionResult:
    return ConclusionResult(
        next_action="answer_directly",
        answer=TEXT,
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
        supported_facts=[
            {"fact": TEXT, "evidence_refs": [{"evidence_id": eid, "quote": TEXT}]}
        ],
    )


@pytest.mark.parametrize("validator", [validate_evidence_id_grounding, validate_grounded_quotes])
@pytest.mark.parametrize("visible", ["", " \n\t", "[E1]\n", TEXT])
def test_explicit_empty_or_unparseable_evidence_never_uses_hidden_documents(validator, visible):
    issues, _ = validator(
        conclusion=conclusion(),
        evidence=EVIDENCE,
        question="阿米娅做了什么？",
        evidence_prompt_text=visible,
    )
    assert any("evidence_id_not_found:E1" in issue for issue in issues)


@pytest.mark.parametrize("validator", [validate_evidence_id_grounding, validate_grounded_quotes])
def test_none_preserves_legacy_offline_validation(validator):
    issues, _ = validator(
        conclusion=conclusion(),
        evidence=EVIDENCE,
        question="阿米娅做了什么？",
        evidence_prompt_text=None,
    )
    assert not issues


@pytest.mark.parametrize("validator", [validate_evidence_id_grounding, validate_grounded_quotes])
def test_valid_visible_evidence_does_not_require_hidden_objects(validator):
    issues, _ = validator(
        conclusion=conclusion(),
        evidence=[],
        question="阿米娅做了什么？",
        evidence_prompt_text=f"[E1]\n{TEXT}",
    )
    assert not issues


def test_quote_validator_rejects_unknown_id_even_when_no_evidence_exists():
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion("E99"), evidence=[], question="阿米娅做了什么？",
    )
    assert any("evidence_id_not_found:E99" in issue for issue in issues)


def test_quote_cannot_match_a_hidden_document_suffix():
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion(),
        evidence=EVIDENCE,
        question="阿米娅做了什么？",
        evidence_prompt_text="[E1]\n杜宾安排了训练。",
    )
    assert any("quote_not_found" in issue for issue in issues)


def hypothesis():
    return HypothesisDocument(
        question="阿米娅做了什么？", intent="plot_fact", query_type="fact",
        entities=["阿米娅"], keywords=["报告"], expected_answer_type="事实",
    )


@pytest.mark.parametrize("mode", ["quote", "grounded", "strict", "evidence_id"])
@pytest.mark.parametrize("action", ["answer_directly", "retrieve_more", "abstain"])
def test_structured_modes_cannot_return_uncited_special_fallbacks(monkeypatch, mode, action):
    def forbidden_fallback(**kwargs):
        pytest.fail("uncited special fallback must not bypass structured grounding")

    monkeypatch.setattr(validation, "answer_from_retrieval_fallbacks", forbidden_fallback)
    monkeypatch.setattr(validation, "corrected_special_answer", forbidden_fallback)
    candidate = conclusion()
    candidate.next_action = action
    result = validation.validate_conclusion_grounding(
        question="阿米娅做了什么？", hypothesis=hypothesis(), evidence=EVIDENCE,
        conclusion=candidate, max_round_reached=True, mode=mode, evidence_prompt_text="",
    )
    assert result.next_action != "answer_directly"


def test_legacy_weak_mode_still_allows_deterministic_fallback(monkeypatch):
    fallback = conclusion()
    monkeypatch.setattr(validation, "answer_from_retrieval_fallbacks", lambda **kw: fallback)
    candidate = conclusion()
    candidate.next_action = "abstain"
    result = validation.validate_conclusion_grounding(
        question="阿米娅做了什么？", hypothesis=hypothesis(), evidence=EVIDENCE,
        conclusion=candidate, max_round_reached=True, mode="weak", evidence_prompt_text="",
    )
    assert result is fallback


@pytest.mark.parametrize("final_round", [False, True])
def test_quote_rejection_never_exposes_unvalidated_snippets(final_round):
    result = validation.validate_conclusion_grounding(
        question="阿米娅做了什么？", hypothesis=hypothesis(), evidence=EVIDENCE,
        conclusion=conclusion(), max_round_reached=final_round, mode="quote",
        evidence_prompt_text="",
    )
    expected_action = "abstain" if final_round else "retrieve_more"
    assert result.next_action == expected_action
    assert TEXT not in result.answer
    assert result.generation_diagnostics["grounding_status"] == f"rejected_to_{expected_action}"


@pytest.mark.parametrize("mode", ["quote", "grounded", "strict", "evidence_id"])
@pytest.mark.parametrize("final_round", [False, True])
def test_structured_runtime_still_accepts_valid_visible_claims(mode, final_round):
    result = validation.validate_conclusion_grounding(
        question="阿米娅做了什么？", hypothesis=hypothesis(), evidence=EVIDENCE,
        conclusion=conclusion(), max_round_reached=final_round, mode=mode,
        evidence_prompt_text=f"[E1]\n{TEXT}",
    )
    assert result.next_action == "answer_directly"
    assert TEXT in result.answer
    assert not result.missing_slots
