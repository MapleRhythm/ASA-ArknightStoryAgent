"""Regression coverage for the withdrawn unique-quote rebinding experiment.

Finding a quote in a different block does not establish the fact's subject.
The canonical validator must reject the bad binding, not silently repair it.
"""
from __future__ import annotations

import pytest

from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


QUESTION = "甲做了什么？"
HYPOTHESIS = HypothesisDocument(
    question=QUESTION, intent="plot_fact", query_type="fact", entities=["甲"],
    keywords=["做了什么"], expected_answer_type="事实",
)
ASSERTION = "甲明确表示会保护村庄里的所有居民。"


def validate(evidence_text, evidence_id, quote):
    conclusion = normalize_conclusion_payload(
        {
            "next_action": "answer_directly",
            "supported_facts": [{
                "fact": ASSERTION,
                "evidence_refs": [{"evidence_id": evidence_id, "quote": quote}],
            }],
            "inferred_facts": [],
            "final_answer": ASSERTION,
        },
        question=QUESTION, dialogue_context="", current_intent=HYPOTHESIS.intent,
        current_hypothesis=HYPOTHESIS, max_round_reached=True,
    )
    return validate_conclusion_grounding(
        question=QUESTION, hypothesis=HYPOTHESIS, evidence=[],
        conclusion=conclusion, max_round_reached=True, mode="quote",
        evidence_prompt_text=evidence_text,
    )


@pytest.mark.parametrize(
    ("evidence_text", "evidence_id", "quote"),
    [
        (f"[E1] 无关证据。\n[E2] {ASSERTION}", "E1", ASSERTION),
        (f"[E1] 无关证据。\n[E2] {ASSERTION}\n[E3] {ASSERTION}", "E1", ASSERTION),
        (f"[E1] 无关证据。\n[E2] {ASSERTION}", "E99", ASSERTION),
        ("[E1] 无关证据。\n[E2] 甲会保护居民。", "E1", "甲会保护居民。"),
        ("[E1] 无关证据。\n[E2] 甲是否会保护村庄里的所有居民？",
         "E1", "甲是否会保护村庄里的所有居民？"),
    ],
)
def test_invalid_binding_is_not_automatically_repaired(evidence_text, evidence_id, quote):
    result = validate(evidence_text, evidence_id, quote)
    assert result.next_action == "abstain"
    assert "grounding_repairs" not in result.generation_diagnostics


@pytest.mark.parametrize(
    "source",
    [
        "甲明确表示会保护村庄里的所有居民，随后离开。",
        "甲明确表示会保护村庄里的所有居民？",
        "甲明确表示会保护村庄里的所有居民以外的敌人。",
    ],
)
def test_changed_terminal_punctuation_is_not_silently_accepted(source):
    result = validate(f"[E1] {source}", "E1", ASSERTION)
    assert result.next_action == "abstain"


def test_valid_existing_binding_is_unchanged():
    result = validate(f"[E1] {ASSERTION}", "E1", ASSERTION)
    assert result.next_action == "answer_directly"
    assert result.supported_facts[0]["evidence_refs"][0]["evidence_id"] == "E1"
    assert "grounding_repairs" not in result.generation_diagnostics
