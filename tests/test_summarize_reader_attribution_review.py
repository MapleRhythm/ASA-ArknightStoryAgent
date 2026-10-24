import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_reader_attribution_review.py"
SPEC = importlib.util.spec_from_file_location("attribution_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs():
    prompt = "question: 做了什么？\nevidence:\n[E1]\n检查报告。\noutput_schema: grounded_action_exx_v1"
    raw = '{"next_action":"answer_directly","supported_facts":[{"fact":"检查报告。","evidence_ids":["E1"]}]}'
    row = {
        "id": "q::flat", "system": "system",
        "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": raw}],
        "raw_output": raw, "diagnostic": {"source_id": "q", "condition": "flat"},
    }
    review = {
        "condition_order": ["flat"], "reviewer": "model-review",
        "independent_human_review": False, "criteria": {}, "limitations": [],
        "reviews": [{"reader": "base", "source_id": "q", "outcomes": ["complete"], "reasons": ["supported"]}],
    }
    return [row], {"base": [copy.deepcopy(row)]}, review


def test_full_denominator_and_protocol_gate():
    matrix, predictions, review = inputs()
    report = MODULE.summarize(matrix, predictions, review)
    assert report["results"]["base"]["flat"]["primary_success"] == 1
    assert report["independent_human_review"] is False
    predictions["base"][0]["hit_max_new_tokens"] = True
    report = MODULE.summarize(matrix, predictions, review)
    assert report["results"]["base"]["flat"]["rows"] == 1
    assert report["results"]["base"]["flat"]["primary_success"] == 0


def test_rejects_missing_prediction_duplicate_review_and_prompt_drift():
    matrix, predictions, review = inputs()
    with pytest.raises(ValueError, match="cover matrix"):
        MODULE.summarize(matrix, {"base": []}, review)
    duplicated = copy.deepcopy(review)
    duplicated["reviews"] *= 2
    with pytest.raises(ValueError, match="duplicate or invalid"):
        MODULE.summarize(matrix, predictions, duplicated)
    predictions["base"][0]["conversations"][0]["value"] += "changed"
    with pytest.raises(ValueError, match="differs from matrix"):
        MODULE.summarize(matrix, predictions, review)


def test_cannot_label_an_answer_as_unanswered_or_omit_review():
    matrix, predictions, review = inputs()
    review["reviews"][0]["outcomes"] = ["unanswered"]
    with pytest.raises(ValueError, match="conflicts"):
        MODULE.summarize(matrix, predictions, review)
    review["reviews"] = []
    with pytest.raises(ValueError, match="missing"):
        MODULE.summarize(matrix, predictions, review)
