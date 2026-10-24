from types import SimpleNamespace

import pytest

from asa_arknight_story_agent.inference.cpu_pipeline import CPUInferencePipeline
from asa_arknight_story_agent.inference.evidence.full_document_contract import validate_full_document_prompt
from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import build_minimal_conclusion_prompt
from asa_arknight_story_agent.inference.generation.hypothesis_prompt_rendering import (
    build_hypothesis_prompt, build_follow_up_hypothesis_prompt,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument, ConclusionResult


def fixture():
    p = CPUInferencePipeline(
        retriever=None, generator=SimpleNamespace(max_tokens=1536),
        prompt_evidence_top_k=20, prompt_evidence_require_full_documents=True,
        prompt_evidence_max_chars_per_doc=0, prompt_conclusion_evidence_max_total_chars=0,
        answer_grounding_mode="evidence_id", conclusion_prompt_mode="minimal",
    )
    h = HypothesisDocument(question="谁完成检查？", intent="plot_fact", query_type="fact",
                           entities=["甲"], keywords=["检查"], expected_answer_type="fact")
    evidence = [{"document": {"id": f"doc{i}", "clean_text": f"第{i}条。甲完成检查。" * 90},
                 "evidence_chain_text": "不应替换正文的链条。"} for i in range(20)]
    return p, h, evidence


def test_zero_remains_unlimited_and_20_full_chunks_reach_prompt():
    p, h, evidence = fixture()
    assert p.prompt_evidence_max_chars_per_doc == p.prompt_conclusion_evidence_max_total_chars == 0
    prepared = p.prepare_prompt_evidence(h.question, h, evidence)
    assert [x["document"]["id"] for x in prepared] == [x["document"]["id"] for x in evidence]
    assert all("evidence_chain_text" not in x for x in prepared)
    prompt, body = build_minimal_conclusion_prompt(
        question=h.question, current_hypothesis=h, evidence=evidence,
        current_round=1, max_retrieval_rounds=2, prompt_evidence_top_k=20, prompt_evidence=prepared,
        grounding_mode="evidence_id", evidence_max_chars_per_doc=0, evidence_max_total_chars=0,
    )
    validate_full_document_prompt(p, prompt, prepared)
    assert "不应替换" not in body
    assert prompt.endswith("<|im_start|>assistant\n")


def test_missing_evidence_fails_closed():
    p, _, evidence = fixture()
    with pytest.raises(ValueError, match="Full evidence missing"):
        validate_full_document_prompt(p, "[E1]\n不完整", evidence)


def test_full_mode_rejects_refinement():
    with pytest.raises(ValueError, match="unmodified top-K"):
        CPUInferencePipeline(retriever=None, generator=None,
                             prompt_evidence_require_full_documents=True, enable_crag_refinement=True)


def test_initial_prefix_can_match_training_without_changing_legacy():
    assert build_hypothesis_prompt("谁？").endswith("{")
    assert build_hypothesis_prompt("谁？", assistant_prefix="").endswith("<|im_start|>assistant\n")


def test_follow_up_uses_full_evidence_and_empty_prefix():
    p, h, evidence = fixture()
    prepared = p.prepare_prompt_evidence(h.question, h, evidence)
    c = ConclusionResult(next_action="retrieve_more", answer="", missing_slots=[],
                         clarification_question="", follow_up_hypothesis=h)
    prompt = build_follow_up_hypothesis_prompt(
        h.question, h, evidence, [], [], c, 1, 2, 20, prompt_evidence=prepared,
        require_full_prompt_evidence=True, assistant_prefix="",
    )
    validate_full_document_prompt(p, prompt, prepared)
    assert prompt.endswith("<|im_start|>assistant\n")
