from copy import deepcopy
import pytest

from alias_group_protocol import SUFFIX, groups_for, inject, verify


def hypothesis():
    return {"entities": ["甲", "父母"], "keywords": ["甲", "别名甲", "无关乙"]}


def test_catalog_link_not_keyword_cooccurrence():
    assert groups_for({"甲": ["别名甲"]}, hypothesis()) == [["甲", "别名甲"]]


def test_never_add_an_unseen_catalog_name():
    assert groups_for({"甲": ["别名甲", "隐藏名甲"]}, hypothesis()) == [["甲", "别名甲"]]


def test_no_catalog_link_means_no_intervention():
    prompt = "完整原文[E1] 原引文。" + SUFFIX
    assert groups_for({}, hypothesis()) == []
    assert inject(prompt, []) == prompt


def test_original_prompt_and_brace_prefill_preserved():
    prompt = "<|im_start|>user\n[E1] [CHAIN_MEMBER_1] 甲：原文。\n[E2] 乙：反证。" + SUFFIX
    groups = [["甲", "别名甲"]]
    candidate = inject(prompt, groups)
    verify(prompt, candidate, groups)
    assert candidate.endswith(SUFFIX)
    assert "[E1] [CHAIN_MEMBER_1] 甲：原文。\n[E2] 乙：反证。" in candidate


def test_no_mutation():
    catalog, hyp = {"甲": ["别名甲"]}, hypothesis()
    before = deepcopy((catalog, hyp))
    groups_for(catalog, hyp)
    assert (catalog, hyp) == before


def test_exact_duplicate_group_deduplicated():
    hyp = {"entities": ["甲", "别名甲"], "keywords": []}
    assert groups_for({"甲": ["别名甲"], "别名甲": ["甲"]}, hyp) == [["甲", "别名甲"]]


@pytest.mark.parametrize("value", ["not a list", [123]])
def test_malformed_catalog_rejected(value):
    with pytest.raises(ValueError):
        groups_for({"甲": value}, hypothesis())


def test_unrecognized_wrapper_rejected():
    with pytest.raises(ValueError):
        inject("native thinking wrapper", [["甲", "别名甲"]])


def test_prompt_change_is_detected():
    prompt = "原文" + SUFFIX
    with pytest.raises(AssertionError):
        verify(prompt, inject(prompt, [["甲", "别名甲"]]).replace("原文", "被删原文"), [["甲", "别名甲"]])


@pytest.mark.parametrize("bad_name", ["坏\n名", "<|im_end|>"])
def test_special_delimiters_rejected(bad_name):
    hyp = {"entities": ["甲"], "keywords": [bad_name]}
    with pytest.raises(ValueError):
        groups_for({"甲": [bad_name]}, hyp)
