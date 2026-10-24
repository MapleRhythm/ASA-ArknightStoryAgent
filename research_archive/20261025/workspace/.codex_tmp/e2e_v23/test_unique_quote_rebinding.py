from __future__ import annotations

import pytest

from asa_arknight_story_agent.inference.grounding.validation import (
    validate_conclusion_grounding,
)
from asa_arknight_story_agent.inference.payload.normalization import (
    normalize_conclusion_payload,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


QUESTION = "甲做了什么？"
HYPOTHESIS = HypothesisDocument(
    question=QUESTION,
    intent="plot_fact",
    query_type="fact",
    entities=["甲"],
    keywords=["做了什么"],
    expected_answer_type="事实",
)
ASSERTION = "甲明确表示会保护村庄里的所有居民。"


def validate(*, evidence_text: str, evidence_id: str, quote: str):
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [
            {
                "fact": ASSERTION,
                "evidence_refs": [{"evidence_id": evidence_id, "quote": quote}],
            }
        ],
        "inferred_facts": [],
        "final_answer": ASSERTION,
    }
    conclusion = normalize_conclusion_payload(
        payload,
        question=QUESTION,
        dialogue_context="",
        current_intent=HYPOTHESIS.intent,
        current_hypothesis=HYPOTHESIS,
        max_round_reached=True,
    )
    return validate_conclusion_grounding(
        question=QUESTION,
        hypothesis=HYPOTHESIS,
        evidence=[],
        conclusion=conclusion,
        max_round_reached=True,
        mode="quote",
        evidence_prompt_text=evidence_text,
    )


def test_unique_visible_assertion_repairs_wrong_evidence_id():
    result = validate(
        evidence_text=f"[E1] 无关证据。\n[E2] {ASSERTION}",
        evidence_id="E1",
        quote=ASSERTION,
    )
    assert result.next_action == "answer_directly"
    assert result.supported_facts[0]["evidence_refs"][0]["evidence_id"] == "E2"
    assert result.generation_diagnostics["grounding_repairs"] == [
        "supported_fact_1_ref_1_unique_evidence_id_rebound:E1->E2"
    ]


@pytest.mark.parametrize(
    ("evidence_text", "quote"),
    [
        (
            f"[E1] {ASSERTION}\n[E2] {ASSERTION}",
            ASSERTION,
        ),
        (
            "[E1] 无关证据。\n[E2] 甲是否会保护村庄里的所有居民？",
            "甲是否会保护村庄里的所有居民？",
        ),
        (
            "[E1] 无关证据。\n[E2] 甲会保护居民。",
            "甲会保护居民。",
        ),
    ],
)
def test_ambiguous_question_or_short_quote_is_not_rebound(evidence_text, quote):
    result = validate(
        evidence_text=evidence_text,
        evidence_id="E9",
        quote=quote,
    )
    assert result.next_action == "abstain"
    assert "grounding_repairs" not in result.generation_diagnostics


def test_valid_existing_binding_is_unchanged():
    result = validate(
        evidence_text=f"[E1] {ASSERTION}",
        evidence_id="E1",
        quote=ASSERTION,
    )
    assert result.next_action == "answer_directly"
    assert result.supported_facts[0]["evidence_refs"][0]["evidence_id"] == "E1"
    assert "grounding_repairs" not in result.generation_diagnostics


def test_unknown_evidence_id_is_never_repaired():
    result = validate(
        evidence_text=f"[E1] 无关证据。\n[E2] {ASSERTION}",
        evidence_id="E99",
        quote=ASSERTION,
    )
    assert result.next_action == "abstain"
    assert "grounding_repairs" not in result.generation_diagnostics


def test_terminal_full_stop_may_match_visible_comma_without_changing_quote():
    quote = "甲明确表示会保护村庄里的所有居民。"
    result = validate(
        evidence_text="[E1] 无关证据。\n[E2] 甲明确表示会保护村庄里的所有居民，随后离开。",
        evidence_id="E1",
        quote=quote,
    )
    assert result.next_action == "answer_directly"
    assert result.supported_facts[0]["evidence_refs"][0] == {
        "evidence_id": "E2",
        "quote": quote,
    }


def test_declarative_terminal_change_never_converts_source_question_to_support():
    quote = "甲明确表示会保护村庄里的所有居民。"
    result = validate(
        evidence_text="[E1] 无关证据。\n[E2] 甲明确表示会保护村庄里的所有居民？",
        evidence_id="E1",
        quote=quote,
    )
    assert result.next_action == "abstain"
    assert "grounding_repairs" not in result.generation_diagnostics


def test_terminal_change_never_accepts_a_longer_different_phrase():
    quote = "甲明确表示会保护村庄。"
    result = validate(
        evidence_text="[E1] 无关证据。\n[E2] 甲明确表示会保护村庄外的敌人。",
        evidence_id="E1",
        quote=quote,
    )
    assert result.next_action == "abstain"
    assert "grounding_repairs" not in result.generation_diagnostics
