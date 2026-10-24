"""Renderer must add layout without rewriting statements or quote policy."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from asa_arknight_story_agent.inference.grounding.grounded_fact_answers import answer_from_grounded_facts


def cited(text):
    return {"fact": text, "evidence_refs": [{"evidence_id": "E1", "quote": text}]}


def conclusion(facts=(), inferred=()):
    return SimpleNamespace(
        answer="保留原始摘要", supported_facts=[cited(s) for s in facts],
        inferred_facts=[cited(s) for s in inferred],
    )


@pytest.mark.parametrize("tail", ["。", "！", "？", "……", "；", ".", "!", "?", "」", "”", "", "。。"])
def test_only_layout_is_added(tail):
    texts = ["甲已离开" + tail, "乙仍留在原地" + tail]
    row = conclusion(texts)
    before = deepcopy(vars(row))
    assert answer_from_grounded_facts(row) == "根据当前证据可确认：" + "\n".join(texts)
    assert vars(row) == before


@pytest.mark.parametrize("texts,expected", [
    ([], "保留原始摘要"),
    (["甲已离开。"], "甲已离开。"),
    (["甲已离开。", "甲已离开。"], "甲已离开。"),
])
def test_zero_single_and_dedup_fallback(texts, expected):
    assert answer_from_grounded_facts(conclusion(texts)) == expected


def test_direct_and_inferred_remain_separate():
    row = conclusion(["甲已离开。", "乙仍留在原地。"], ["可能有人返回。", "可能有人旁观。"])
    assert answer_from_grounded_facts(row) == (
        "根据当前证据可确认：甲已离开。\n乙仍留在原地。\n"
        "基于上述证据的推断（非直接事实）：可能有人返回。\n可能有人旁观。"
    )


def test_inference_only_is_not_confirmed():
    row = conclusion(inferred=["可能有人返回。", "可能有人旁观。"])
    assert answer_from_grounded_facts(row) == (
        "基于上述证据的推断（非直接事实）：可能有人返回。\n可能有人旁观。"
    )


def test_sensitive_fact_still_cannot_borrow_a_sibling_quote():
    row = conclusion(["甲是乙的母亲。", "丙是丁的母亲。"])
    row.supported_facts[0]["evidence_refs"][0]["quote"] = "甲与乙在门口碰面。"
    assert answer_from_grounded_facts(row) == "丙是丁的母亲。"


def test_fact_and_reference_contents_not_repaired():
    row = conclusion(["甲说“好了。”", "乙已离开。。"])
    row.supported_facts[0]["evidence_refs"][0]["evidence_id"] = "E99"
    before = deepcopy(vars(row))
    answer_from_grounded_facts(row)
    assert vars(row) == before


def test_internal_linebreaks_are_preserved():
    row = conclusion(["甲已离开。\n但乙没有。", "丙仍在门口。"])
    assert answer_from_grounded_facts(row) == (
        "根据当前证据可确认：甲已离开。\n但乙没有。\n丙仍在门口。"
    )
