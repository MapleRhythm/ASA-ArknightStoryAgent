from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_answer_slots_v1 as old
import probe_answer_slots_v2 as new

CASE = {"question": "甲去了什么地方？", "shown_answer": "甲去了某个地方。", "old_review": "秘密", "evidence": "秘密原文"}


def value():
    return {
        "version": new.VERSION, "coverage": "none",
        "slots": [{"requirement": "具体去处", "status": "unfilled", "answer_spans": [], "reason": "没有具体地点"}],
        "reason": "缺少被问的信息",
    }


def test_only_the_ambiguous_clause_changes():
    assert old.RUBRIC == new.RUBRIC.replace(new.NEW_CLAUSE, new.OLD_CLAUSE)
    assert new.VERSION == old.VERSION and new.validate is old.validate


def test_request_does_not_leak_review_or_evidence():
    assert json.loads(new.messages(CASE)[1]["content"]) == {k: CASE[k] for k in ("question", "shown_answer")}
    assert new.messages(CASE)[1] == old.messages(CASE)[1]


def test_missing_slot_keeps_an_empty_array():
    assert new.validate(value(), CASE) == value()


@pytest.mark.parametrize("missing", ["answer_spans", "requirement", "status", "reason"])
def test_no_silent_field_repair(missing):
    result = value()
    del result["slots"][0][missing]
    with pytest.raises(ValueError, match="schema"):
        new.validate(result, CASE)
    assert missing not in result["slots"][0]


def test_extra_field_is_still_an_error():
    result = value()
    result["slots"][0]["answer_spans_note"] = None
    with pytest.raises(ValueError, match="schema"):
        new.validate(result, CASE)


def test_nonverbatim_quote_is_still_an_error():
    result = value()
    result["coverage"] = "complete"
    result["slots"][0].update(status="filled", answer_spans=["甲去了另一个地方。"])
    with pytest.raises(ValueError, match="verbatim"):
        new.validate(result, CASE)


def test_unfilled_with_span_is_still_an_error():
    result = value()
    result["slots"][0]["answer_spans"] = [CASE["shown_answer"]]
    with pytest.raises(ValueError, match="inconsistent"):
        new.validate(result, CASE)


def test_cannot_promote_none_to_complete():
    result = value()
    result["coverage"] = "complete"
    with pytest.raises(ValueError, match="inconsistent"):
        new.validate(result, CASE)


def test_validator_does_not_mutate_input():
    result = value()
    snapshot = deepcopy(result)
    new.validate(result, CASE)
    assert result == snapshot
