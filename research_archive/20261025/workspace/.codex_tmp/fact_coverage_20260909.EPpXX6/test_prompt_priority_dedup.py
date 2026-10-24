"""Score-prioritized representatives; no text repair or threshold change."""
from copy import deepcopy

import pytest

from asa_arknight_story_agent.inference.evidence.prompt_ordering import (
    merge_forced_prompt_evidence,
    select_prompt_evidence,
)
from asa_arknight_story_agent.inference.evidence.prompt_similarity import (
    dedupe_prompt_evidence_candidates,
    jaccard_similarity,
    text_similarity_tokens,
)
from asa_arknight_story_agent.inference.evidence.texts import prompt_evidence_score


def item(identity, text, score=0, **extra):
    return {"document": {"id": identity, "clean_text": text},
            "rerank_score": score, **extra}


def select(rows, k=10):
    return select_prompt_evidence("unused", None, rows, prompt_evidence_top_k=k)


def test_identical_text_keeps_higher_score_not_first_input():
    low = item("low", "完全相同的原始剧情文本", 1)
    high = item("high", "完全相同的原始剧情文本", 3)
    assert select([low, high]) == [high]


def test_same_document_identity_keeps_higher_score_payload():
    low = item("same", "较低分数的旧链", 1)
    high = item("same", "含新上下文的另一条链", 3)
    assert select([low, high]) == [high]


def test_near_duplicate_keeps_higher_score_complete_text():
    common = "".join(chr(0x4E00 + n) for n in range(150))
    low = item("low", common, 1)
    high = item("high", common + "她依然没有放手", 3)
    similarity = jaccard_similarity(
        text_similarity_tokens(low["document"]["clean_text"]),
        text_similarity_tokens(high["document"]["clean_text"]))
    assert 0.82 <= similarity < 1
    assert select([low, high]) == [high]
    assert select([high, low]) == [high]


def test_top_k_is_applied_after_dedup_not_before():
    high = item("high", "相同原文", 3)
    twin = item("twin", "相同原文", 2)
    other = item("other", "完全无关证据", 1)
    assert select([twin, other, high], k=2) == [high, other]


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_score_duplicate_preserves_first_input(reverse):
    rows = [item("a", "相同原文", 3), item("b", "相同原文", 3)]
    if reverse:
        rows.reverse()
    assert select(rows) == rows[:1]


def test_equal_score_distinct_evidence_remains_stable():
    rows = [item("a", "天气转晴", 2), item("b", "伤员撤离", 2), item("c", "进城手续", 2)]
    assert select(rows) == rows


def test_chain_score_still_precedes_document_rerank_score():
    low = item("low", "相同原文", 100, evidence_chain_score=1)
    high = item("high", "相同原文", -10, evidence_chain_score=3)
    assert select([low, high]) == [high]


def test_chain_model_score_fallback_is_respected():
    low = item("low", "相同原文", 100, evidence_chain_model_score=1)
    high = item("high", "相同原文", -10, evidence_chain_model_score=3)
    assert select([low, high]) == [high]


def test_moegirl_penalty_is_applied_before_choosing_duplicate():
    wiki = item("moegirl/a", "相同原文", 6)
    story = item("story/a", "相同原文", 1)
    assert prompt_evidence_score(wiki) == 0
    assert select([wiki, story]) == [story]


def test_direct_text_preference_is_preserved():
    chain = "共享链文本"
    a = item("a", "天气转晴", 1, evidence_chain_text=chain, prompt_prefer_clean_text=True)
    b = item("b", "伤员撤离", 3, evidence_chain_text=chain, prompt_prefer_clean_text=True)
    assert select([a, b]) == [b, a]


def test_empty_input():
    assert select([]) == []


@pytest.mark.parametrize("k", [0, -1, -10])
def test_nonpositive_limit_does_not_select(k):
    assert select([item("a", "有效文本", 1)], k) == []


def test_selection_does_not_mutate_list_payload_or_source_text():
    rows = [item("low", "完整不截断的相同文本", 1),
            item("high", "完整不截断的相同文本", 3),
            item("other", "另一份独立来源", 2)]
    before = deepcopy(rows)
    selected = select(rows, 2)
    assert rows == before
    assert selected[0] is rows[1] and selected[1] is rows[2]


def test_empty_similarity_tokens_use_identity_and_keep_distinct_items():
    low = item("same", "", 1)
    high = item("same", "", 3)
    other = item("other", "", 2)
    assert select([low, other, high]) == [high, other]


def test_doc_index_fallback_selects_higher_score():
    low = item("", "旧链", 1, doc_index=7)
    high = item("", "新链", 3, doc_index=7)
    assert select([low, high]) == [high]


def test_text_identity_fallback_selects_higher_score():
    low = item("", "相同原文", 1)
    high = item("", "相同原文", 3)
    assert select([low, high]) == [high]


def test_zero_chain_score_is_not_replaced_by_rerank_fallback():
    low = item("low", "相同原文", 100, evidence_chain_score=0)
    high = item("high", "相同原文", -10, evidence_chain_score=1)
    assert select([low, high]) == [high]


def test_low_level_dedup_still_preserves_caller_priority():
    low = item("low", "相同原文", 1)
    high = item("high", "相同原文", 3)
    assert dedupe_prompt_evidence_candidates([low, high]) == [low]


def test_forced_evidence_still_has_priority_over_selected_scores():
    forced = item("forced", "相同原文", 1)
    high = item("high", "相同原文", 3)
    assert merge_forced_prompt_evidence([forced], [high], limit=10) == [forced]
