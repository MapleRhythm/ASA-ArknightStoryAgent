import copy
import json

import pytest

from judge_visible_answers import messages, validate


def row():
    return {
        "id": "x", "question": "谁批评了她？", "evidence_brief": "[E1] 甲：乙骂了她，我不同意。",
        "grounded": {"next_action": "answer_directly", "answer": "甲批评了她。"},
        "hidden_gold": "禁止发送此字段", "review_note": "禁止发送复核意见",
    }


def verdict():
    return {
        "protocol": "asa_visible_answer_semantics_v1", "verdict": "unsupported",
        "context_sufficiency": "sufficient", "action_appropriateness": "appropriate",
        "unsupported_spans": ["甲批评了她"], "missing_requirements": [],
        "reason": "甲在转述乙，不能把批评归给甲。",
    }


def test_request_does_not_send_gold_or_review():
    original = row()
    before = copy.deepcopy(original)
    text = json.dumps(messages(original), ensure_ascii=False)
    assert "禁止发送" not in text
    assert original == before


def test_accepts_bound_unsupported_span():
    assert validate(verdict(), row())["verdict"] == "unsupported"


def test_rejects_invented_unsupported_span():
    value = verdict()
    value["unsupported_spans"] = ["她批评了甲"]
    with pytest.raises(ValueError, match="verbatim"):
        validate(value, row())


def test_rejects_complete_with_missing_requirements():
    value = verdict()
    value.update(verdict="supported_complete", unsupported_spans=[], missing_requirements=["批评者"])
    with pytest.raises(ValueError, match="inconsistent"):
        validate(value, row())


def test_no_answer_must_match_action():
    value = verdict()
    value.update(verdict="no_answer", unsupported_spans=[])
    with pytest.raises(ValueError, match="inconsistency"):
        validate(value, row())
