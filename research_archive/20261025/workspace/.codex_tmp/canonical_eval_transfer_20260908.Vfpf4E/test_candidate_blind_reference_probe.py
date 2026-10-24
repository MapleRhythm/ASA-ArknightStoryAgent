import json
import pytest

from candidate_blind_reference_probe import messages, validate, VERSION


def test_candidate_and_reviews_never_enter_reference_request():
    row = {"question": "问题", "evidence_brief": "[E1]原文", "grounded": {"answer": "不能发送候选答案"},
           "review": "不能发送评分", "gold": "不能发送gold"}
    text = json.dumps(messages(row), ensure_ascii=False)
    assert "不能发送" not in text


def test_reference_quotes_must_bind_to_visible_text():
    row = {"evidence_brief": "[E1]甲说乙来了"}
    value = {"version": VERSION, "question_requirements": ["谁来了"],
             "answerability": "sufficient", "supported_answer": "乙来了",
             "evidence_refs": [{"evidence_id": "E1", "quote": "乙来了"}], "limitations": []}
    assert validate(value, row, lambda _: {"E1": "甲说乙来了"})
    value["evidence_refs"][0]["quote"] = "丙来了"
    with pytest.raises(ValueError, match="verbatim"):
        validate(value, row, lambda _: {"E1": "甲说乙来了"})
