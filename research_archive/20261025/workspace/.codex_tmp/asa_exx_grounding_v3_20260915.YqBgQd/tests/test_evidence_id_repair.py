from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from asa_arknight_story_agent.inference.generation.evidence_id_repair import repair_evidence_id_conclusion
from asa_arknight_story_agent.inference.generation.answer_conclusion_generation import should_try_final_direct_answer
from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.grounding.partial_support import preserve_partial_support
from asa_arknight_story_agent.inference.grounding.evidence_id_validation import surface_normalize
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument

QUESTION = "阿米娅如何处理申请，杜宾之后做了什么？"
HYPOTHESIS = HypothesisDocument(QUESTION, "plot_fact", "fact", ["阿米娅", "杜宾"], ["申请"], "事实")
TEXT = "阿米娅同意申请。杜宾安排训练。"
BRIEF = "[E1]\n" + TEXT
EVIDENCE = [{"document": {"id": "story#1", "clean_text": TEXT}}]


def fact(text, evidence_id="E1"):
    return {"fact": text, "evidence_refs": [{"evidence_id": evidence_id}]}


def candidate(facts):
    return ConclusionResult(
        "answer_directly", "；".join(item["fact"] for item in facts),
        [], "", None, supported_facts=facts,
    )


def checked(facts, final=False, brief=BRIEF):
    return validate_conclusion_grounding(
        question=QUESTION, hypothesis=HYPOTHESIS, evidence=EVIDENCE,
        conclusion=candidate(facts), max_round_reached=final,
        mode="evidence_id", evidence_prompt_text=brief,
    )


def payload(facts):
    return json.dumps({
        "next_action": "answer_directly", "supported_facts": [
            {"fact": item["fact"], "evidence_ids": [r["evidence_id"] for r in item["evidence_refs"]]}
            for item in facts
        ],
    }, ensure_ascii=False)


def run_repair(replacement, original=None, trace=None):
    calls = []
    original = original or checked([fact("阿米娅同意申请。"), fact("杜宾开展星际探险。")])
    trace = trace if trace is not None else [{"round": 1}]

    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if isinstance(replacement, Exception):
            raise replacement
        return replacement

    pipeline = SimpleNamespace(
        answer_grounding_mode="evidence_id", max_retrieval_rounds=2,
        prompt_evidence_require_full_documents=True,
        generator=SimpleNamespace(generate=generate, max_tokens=1536, repeat_penalty=1.05),
    )
    output = repair_evidence_id_conclusion(
        pipeline=pipeline, question=QUESTION, hypothesis=HYPOTHESIS,
        evidence=EVIDENCE, evidence_prompt_text=BRIEF, conclusion=original,
        current_round=1, retrieval_trace=trace,
    )
    return output, calls, trace


def test_same_evidence_repair_keeps_valid_fact_and_fixes_invalid_one():
    original = checked([fact("阿米娅同意申请。"), fact("杜宾开展星际探险。")])
    untouched = deepcopy(original)
    result, calls, trace = run_repair(
        payload([fact("阿米娅同意申请。"), fact("杜宾安排训练。")]), original,
    )
    assert result.next_action == "answer_directly"
    assert "星际探险" not in result.answer
    assert len(calls) == 1
    assert BRIEF in calls[0][0]
    assert calls[0][0].endswith("<|im_start|>assistant\n")
    assert calls[0][1]["max_tokens"] == 1536
    assert trace[0]["grounding_repair_attempts"][0]["status"] == "accepted_after_same_evidence_repair"
    assert asdict(original) == asdict(untouched)


@pytest.mark.parametrize("replacement", [
    "{bad JSON", "[]", RuntimeError("network unavailable"),
    '{"next_action":"abstain","reason":"不足"}',
    payload([fact("阿米娅同意申请。")]),
    payload([fact("杜宾安排训练。"), fact("杜宾安排训练。")]),
    payload([fact("阿米娅同意申请。"), fact("杜宾是阿米娅的母亲。")]),
    payload([fact("阿米娅同意申请。"), fact("杜宾安排训练。", "E99")]),
])
def test_failed_repair_never_loses_original_or_accepts_unsupported_answer(replacement):
    original = checked([fact("阿米娅同意申请。"), fact("杜宾开展星际探险。")])
    result, calls, trace = run_repair(replacement, original)
    assert result is original
    assert len(calls) == 1
    assert trace[0]["grounding_repair_attempts"][0]["status"] != "accepted_after_same_evidence_repair"
    assert original.generation_diagnostics["grounding_partial_support"][0]["fact"]["fact"] == "阿米娅同意申请。"


def test_structural_failure_is_not_misclassified_as_surface_mismatch():
    result, calls, _ = run_repair("{}", checked([fact("阿米娅同意申请。", "E99")]))
    assert not calls
    assert result.generation_diagnostics["grounding_failure_kind"] == "structural_or_mixed"


def test_repair_budget_is_one_per_question_even_in_next_round():
    trace = [{"round": 1, "grounding_repair_attempts": [{"status": "failed"}]}, {"round": 2}]
    _, calls, _ = run_repair("{}", trace=trace)
    assert not calls


def test_final_partial_keeps_only_validated_fact_and_remains_abstain():
    conclusion = checked([fact("阿米娅同意申请。"), fact("杜宾开展星际探险。")], final=True)
    result = preserve_partial_support(
        conclusion=conclusion, question=QUESTION, evidence=EVIDENCE,
        evidence_prompt_text=BRIEF, retrieval_trace=[{"round": 2}],
    )
    assert result.next_action == "abstain"
    assert "尚不足以完整回答" in result.answer
    assert "阿米娅同意申请" in result.answer
    assert "星际探险" not in result.answer
    assert result.generation_diagnostics["answer_completeness"] == "partial"


def test_nonfinal_partial_does_not_skip_retrieval():
    conclusion = checked([fact("阿米娅同意申请。"), fact("杜宾开展星际探险。")])
    result = preserve_partial_support(
        conclusion=conclusion, question=QUESTION, evidence=EVIDENCE,
        evidence_prompt_text=BRIEF, retrieval_trace=[{"round": 1}],
    )
    assert result.next_action == "retrieve_more"
    assert result.answer == ""
    assert result.generation_diagnostics["partial_answer_facts"]


@pytest.mark.parametrize("still_visible", [True, False])
def test_cross_round_partial_rebinds_by_body_not_old_number(still_visible):
    first = checked([fact("阿米娅同意申请。"), fact("杜宾开展星际探险。")])
    final = ConclusionResult("abstain", "不足", [], "", None)
    brief = "[E1]\n这是完全无关的内容。"
    if still_visible:
        brief += "\n[E2]\n" + TEXT
    result = preserve_partial_support(
        conclusion=final, question=QUESTION, evidence=EVIDENCE,
        evidence_prompt_text=brief,
        retrieval_trace=[{"round": 1, "conclusion": asdict(first)}, {"round": 2}],
    )
    if still_visible:
        assert result.supported_facts[0]["evidence_refs"] == [{"evidence_id": "E2"}]
        assert result.generation_diagnostics["partial_answer_facts"][0]["source_round"] == 1
    else:
        assert result.answer == "不足"
        assert result.supported_facts == []


def test_new_failed_candidate_does_not_resurrect_previous_facts():
    first = checked([fact("阿米娅同意申请。"), fact("杜宾开展星际探险。")])
    final = checked([fact("杜宾开展星际探险。")], final=True)
    result = preserve_partial_support(
        conclusion=final, question=QUESTION, evidence=EVIDENCE, evidence_prompt_text=BRIEF,
        retrieval_trace=[{"round": 1, "conclusion": asdict(first)}, {"round": 2}],
    )
    assert not result.supported_facts


def test_no_duplicate_final_direct_fallback_for_exx():
    assert not should_try_final_direct_answer(
        pipeline=SimpleNamespace(max_retrieval_rounds=2, answer_grounding_mode="evidence_id"),
        current_hypothesis=HYPOTHESIS, prompt_evidence=EVIDENCE, current_round=2,
    )


def test_surface_normalization_preserves_negation_numbers_and_role_order():
    assert surface_normalize("阿米娅的申请。") == "阿米娅申请"
    assert surface_normalize("不能超过十五小时") != surface_normalize("能超过十六小时")
    assert surface_normalize("甲救了乙") != surface_normalize("乙救了甲")
