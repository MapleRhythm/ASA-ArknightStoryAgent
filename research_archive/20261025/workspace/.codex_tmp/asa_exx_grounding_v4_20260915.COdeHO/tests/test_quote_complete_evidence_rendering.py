from __future__ import annotations

import copy

import pytest

from asa_arknight_story_agent.inference.evidence.rendering import (
    evidence_id_text_map_from_prompt, render_evidence_blocks, render_short_evidence_brief,
)
from asa_arknight_story_agent.inference.generation.conclusion_prompt_rendering import build_conclusion_prompt
from asa_arknight_story_agent.inference.generation.direct_answer_prompt import build_answer_prompt
from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def hypothesis():
    return HypothesisDocument(
        question="阿米娅是否同意？", intent="plot_fact", query_type="fact",
        entities=["阿米娅"], keywords=["同意"], expected_answer_type="事实",
    )


def item(doc_id, text, **kwargs):
    return {"document": {"id": doc_id, "clean_text": text}, **kwargs}


def build(path, evidence, *, total=12000, mode="quote", trace=None):
    common = dict(
        prompt_evidence_top_k=10, prompt_evidence=evidence,
        evidence_max_chars_per_doc=30, evidence_max_total_chars=total,
        grounding_mode=mode, evidence_packing_trace=trace,
    )
    if path == "direct":
        return build_answer_prompt("阿米娅是否同意？", hypothesis(), evidence, **common)
    return build_conclusion_prompt(
        "阿米娅是否同意？", hypothesis(), evidence, [], 1, 2,
        prompt_mode=path, **common,
    )


@pytest.mark.parametrize("path", ["minimal", "full", "direct"])
@pytest.mark.parametrize("mode", ["quote", "strict", "grounded"])
def test_all_quote_answer_paths_preserve_tail_and_whole_chain(path, mode):
    tail = "阿米娅：我没有同意这个申请。"
    chain = "杜宾：这里是一段背景介绍。" * 110 + tail
    evidence = [item("target", tail, evidence_chain_text=chain)]
    before = copy.deepcopy(evidence)
    trace = []
    prompt, brief = build(path, evidence, mode=mode, trace=trace)
    assert chain in brief and brief in prompt
    assert tail in evidence_id_text_map_from_prompt(brief)["E1"]
    assert "…" not in brief
    assert evidence == before
    assert trace[0]["status"] == "included"
    assert trace[0]["source_id"] == "target"


@pytest.mark.parametrize("path", ["minimal", "full", "direct"])
def test_budget_skips_whole_block_then_preserves_later_block(path):
    large = item("large", "不应展示的长正文" * 300, evidence_chain_roles=["隐藏角色"],
                 evidence_chain_text="不应展示的长正文" * 300)
    large["document"]["story_name"] = "不应泄漏的章节提示"
    evidence = [item("first", "甲站在门口。"), large, item("last", "阿米娅没有同意申请。")]
    trace = []
    prompt, brief = build(path, evidence, total=500, trace=trace)
    blocks = evidence_id_text_map_from_prompt(brief)
    assert list(blocks) == ["E1", "E2"]
    assert "阿米娅没有同意申请。" in blocks["E2"]
    assert "不应展示" not in prompt
    assert "不应泄漏的章节提示" not in prompt
    assert "隐藏角色" not in prompt
    assert len(brief) <= 500
    assert [entry["status"] for entry in trace] == ["included", "budget_skip", "included"]
    assert trace[-1]["source_id"] == "last"
    assert trace[-1]["evidence_id"] == "E2"
    assert trace[-1]["budget_used_chars"] == len(brief)


@pytest.mark.parametrize("path", ["minimal", "full", "direct"])
def test_oversized_rank_one_never_overflows_or_hides_following_passage(path):
    prompt, brief = build(
        path, [item("large", "巨大背景" * 1000), item("short", "阿米娅没有同意申请。")],
        total=300,
    )
    assert list(evidence_id_text_map_from_prompt(brief)) == ["E1"]
    assert "巨大背景" not in prompt
    assert "阿米娅没有同意申请。" in brief
    assert len(brief) <= 300


@pytest.mark.parametrize("path", ["minimal", "full", "direct"])
def test_all_blocks_too_large_returns_empty_evidence(path):
    trace = []
    _, brief = build(path, [item("large", "巨大背景" * 1000)], total=30, trace=trace)
    assert brief == ""
    assert trace[0]["status"] == "budget_skip"


def test_new_quote_budget_includes_separator_characters():
    evidence = [item("first", "甲"), item("second", "乙")]
    brief = render_short_evidence_brief(
        evidence, preserve_complete_evidence=True, overflow_policy="skip", max_total_chars=12,
    )
    assert brief == "[E1] 甲"  # Two 6-char blocks plus newline require 13.
    assert render_short_evidence_brief(
        evidence, preserve_complete_evidence=True, overflow_policy="skip", max_total_chars=13,
    ) == "[E1] 甲\n[E2] 乙"


def test_citation_validation_uses_new_visible_ids_after_a_skipped_document():
    evidence = [
        item("first", "杜宾检查了报告。"), item("skipped", "不能借用的证据。" * 300),
        item("last", "阿米娅没有同意申请。"),
    ]
    _, brief = build("minimal", evidence, total=100)
    conclusion = ConclusionResult(
        next_action="answer_directly", answer="阿米娅没有同意申请。",
        missing_slots=[], clarification_question="", follow_up_hypothesis=None,
        supported_facts=[{
            "fact": "阿米娅没有同意申请。",
            "evidence_refs": [{"evidence_id": "E2", "quote": "阿米娅没有同意申请。"}],
        }],
    )
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion, evidence=evidence, question="阿米娅是否同意？",
        evidence_prompt_text=brief,
    )
    assert not issues
    conclusion.supported_facts[0]["evidence_refs"][0]["quote"] = "不能借用的证据。"
    issues, _ = validate_grounded_quotes(
        conclusion=conclusion, evidence=evidence, question="阿米娅是否同意？",
        evidence_prompt_text=brief,
    )
    assert any("quote_not_found" in issue for issue in issues)


def test_legacy_brief_callers_can_still_request_short_summaries():
    evidence = [item("doc", "背景介绍。" * 100)]
    brief = render_short_evidence_brief(evidence, max_chars_per_doc=30, max_total_chars=500)
    assert "…" in brief


def test_unknown_overflow_policy_is_rejected():
    with pytest.raises(ValueError, match="overflow policy"):
        render_evidence_blocks([], overflow_policy="truncate")
    with pytest.raises(ValueError, match="overflow policy"):
        render_short_evidence_brief([], overflow_policy="truncate")
