from copy import deepcopy

import pytest

from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.retrieval import minirag_prompt_hints as hints


def hypothesis(**extra):
    fields = dict(
        question="夏栎如何救援？",
        intent="plot_reasoning",
        query_type="reasoning",
        entities=["夏栎"],
        keywords=["救援"],
        expected_answer_type="剧情事件描述",
    )
    fields.update(extra)
    return HypothesisDocument(**fields)


def evidence(chain):
    return [{"document": {"id": "story/1", "clean_text": "完整剧情原文", "story_name": "剧情"},
             "evidence_chain_text": chain, "evidence_chain_roles": ["action"]}]


@pytest.mark.parametrize("tag", [
    "[CHAIN_LEN=8] ", "[CAUSAL_ORDER=model_candidate] ",
    "[EVIDENCE_TYPES=(context|motive|outcome)] ",
    "[CHAIN_MEMBER_1] ", "[CHAIN_MEMBER_23]\n",
])
def test_internal_wrappers_do_not_enter_anchor_extraction(monkeypatch, tag):
    seen = []
    def capture(text, hyp):
        seen.append(text)
        return ["真实剧情实体"]
    monkeypatch.setattr(hints, "extract_question_anchor_terms", capture)
    rows = evidence(tag + "夏栎：完整对白。")
    hints.render_minirag_hints_for_prompt(rows, hypothesis())
    assert seen == ["夏栎：完整对白。"]


def test_combined_tags_do_not_displace_real_anchors():
    raw = "[CHAIN_LEN=8] [CAUSAL_ORDER=model_candidate] [EVIDENCE_TYPES=(context|motive|outcome)]\n[CHAIN_MEMBER_1] 夏栎：完整对白。"
    result = hints.render_minirag_hints_for_prompt(evidence(raw), hypothesis())
    for marker in ("CHAIN_LEN", "CAUSAL_ORDER", "EVIDENCE_TYPES", "model_candidate", "CHAIN_MEMBER_1"):
        assert marker not in result


def test_clean_chain_text_is_unchanged(monkeypatch):
    seen = []
    monkeypatch.setattr(hints, "extract_question_anchor_terms", lambda text, hyp: seen.append(text) or [])
    hints.render_minirag_hints_for_prompt(evidence("夏栎：完整对白。\n苏茜：谢谢。"), hypothesis())
    assert seen == ["夏栎：完整对白。\n苏茜：谢谢。"]


def test_unwrapped_literal_words_are_not_blacklisted(monkeypatch):
    seen = []
    monkeypatch.setattr(hints, "extract_question_anchor_terms", lambda text, hyp: seen.append(text) or [])
    text = "角色说 CHAIN_LEN 与 model_candidate 是原文中出现的词。"
    hints.render_minirag_hints_for_prompt(evidence(text), hypothesis())
    assert seen == [text]


def test_unknown_bracketed_story_content_remains(monkeypatch):
    seen = []
    monkeypatch.setattr(hints, "extract_question_anchor_terms", lambda text, hyp: seen.append(text) or [])
    text = "[剧情] [UNKNOWN=actor] 完整对白。"
    hints.render_minirag_hints_for_prompt(evidence(text), hypothesis())
    assert seen == [text]


def test_source_hypothesis_and_roles_are_not_mutated():
    rows, hyp = evidence("[CHAIN_LEN=3] [CHAIN_MEMBER_1] 完整对白。"), hypothesis()
    old_rows, old_hyp = deepcopy(rows), deepcopy(hyp)
    result = hints.render_minirag_hints_for_prompt(rows, hyp)
    assert rows == old_rows and hyp == old_hyp
    assert "neighbors=救援,action" in result


def test_no_chain_preserves_metadata_and_question_hints():
    rows = evidence("")
    result = hints.render_minirag_hints_for_prompt(rows, hypothesis())
    assert result == "entities=夏栎,剧情 | relations=夏栎-相关-剧情 | neighbors=救援,action"


def test_no_evidence_preserves_hypothesis_entities():
    assert hints.render_minirag_hints_for_prompt([], hypothesis()) == "entities=夏栎 | relations= | neighbors=救援"


def test_explicit_user_hypothesis_not_silently_filtered():
    hyp = hypothesis(question="解释 CHAIN_LEN", entities=["CHAIN_LEN"], keywords=[])
    assert "entities=CHAIN_LEN" in hints.render_minirag_hints_for_prompt([], hyp)
