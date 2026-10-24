from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_answer_slots_v1 as probe

CASE = {
    "id": "test", "question": "甲为何与乙同行？", "shown_answer": "双方有共同目标。",
    "evidence": "秘密参考资料", "old_judge": "旧判断", "action": "answer_directly",
}


def value(status="filled"):
    return {
        "version": probe.VERSION, "coverage": {"filled": "complete", "unfilled": "none", "uncertain": "uncertain"}[status],
        "slots": [{"requirement": "同行原因", "status": status,
                   "answer_spans": ["双方有共同目标。"] if status == "filled" else [], "reason": "测试"}],
        "reason": "测试",
    }


def test_payload_excludes_evidence_actions_and_old_labels():
    payload = json.loads(probe.messages(CASE)[1]["content"])
    assert payload == {"question": CASE["question"], "shown_answer": CASE["shown_answer"]}


@pytest.mark.parametrize("status", ["filled", "unfilled", "uncertain"])
def test_single_slot_valid_schema(status):
    result = value(status)
    assert probe.validate(result, CASE) == result


def test_partial_requires_a_missing_slot():
    result = value()
    result["coverage"] = "partial"
    result["slots"].append({"requirement": "另一个明确子问", "status": "unfilled", "answer_spans": [], "reason": "缺失"})
    assert probe.validate(result, CASE) == result


def test_filled_requires_actual_verbatim_answer_span():
    result = value()
    result["slots"][0]["answer_spans"] = ["甲为何与乙同行"]
    with pytest.raises(ValueError, match="verbatim"):
        probe.validate(result, CASE)


def test_filled_must_not_be_empty():
    result = value()
    result["slots"][0]["answer_spans"] = []
    with pytest.raises(ValueError, match="inconsistent"):
        probe.validate(result, CASE)


def test_missing_slot_cannot_be_silently_complete():
    result = value("unfilled")
    result["coverage"] = "complete"
    with pytest.raises(ValueError, match="inconsistent"):
        probe.validate(result, CASE)


def test_repeated_slot_is_not_double_counted():
    result = value()
    result["slots"].append(deepcopy(result["slots"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        probe.validate(result, CASE)


def test_empty_question_decomposition_rejected():
    result = value()
    result["slots"] = []
    with pytest.raises(ValueError, match="asked slots"):
        probe.validate(result, CASE)
