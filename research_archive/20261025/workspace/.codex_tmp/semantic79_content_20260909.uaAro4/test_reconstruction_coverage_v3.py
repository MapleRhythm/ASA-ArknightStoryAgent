from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_reconstruction_coverage39_v3 import prepare, rubric


def row(rid="one", question="甲去了哪里？", parsed="甲去了学校。", shown="甲去了学校。"):
    return {"id": rid, "question": question, "status": "ok", "parsed": {"answer": parsed},
            "grounded": {"answer": shown, "next_action": "answer_directly"}, "evidence_brief": "不得发送"}


def test_unchanged_answer_is_one_provider_input():
    rows, cases, pairs = prepare({"results": [row()]})
    assert len(rows) == len(cases) == len(pairs) == 1
    assert not pairs[0]["changed"]
    assert pairs[0]["parsed_group_id"] == pairs[0]["shown_group_id"]
    assert len(cases[0]["members"]) == 2


def test_changed_answer_produces_two_blind_inputs():
    _, cases, pairs = prepare({"results": [row(shown="甲去了某个地方。")]})
    assert len(cases) == 2 and pairs[0]["changed"]
    for case in cases:
        assert set(json.loads(rubric.messages(case)[1]["content"])) == {"question", "shown_answer"}


def test_question_is_part_of_exact_reuse_key():
    _, cases, pairs = prepare({"results": [row(), row("two", question="甲去了哪里读书？")]})
    assert len(cases) == 2
    assert pairs[0]["shown_group_id"] != pairs[1]["shown_group_id"]


def test_different_states_can_share_identical_inputs_without_disappearing():
    rows, cases, pairs = prepare({"results": [row(), row("two")]})
    assert len(rows) == len(pairs) == 2 and len(cases) == 1
    assert len(cases[0]["members"]) == 4


def test_nonanswers_are_excluded_only_from_subanalysis_and_input_is_preserved():
    direct, more, error = row(), row("more"), row("error")
    more["grounded"].update(next_action="retrieve_more", answer="")
    error["status"] = "error"
    value = {"results": [direct, more, error]}
    before = deepcopy(value)
    rows, _, _ = prepare(value)
    assert rows == [direct] and value == before
