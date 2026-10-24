from copy import deepcopy
import pytest
from audit_quote_pointer8b_v1 import combine_triplet, prepare, split_requests


def arms():
    shared = {"source_id": "source", "question": "question", "evidence_brief": "[E1] source",
              "current_round": 1, "max_retrieval_rounds": 2,
              "status": "parse_or_validation_error"}
    return [[dict(shared, id=arm, renderer=arm)] for arm in ("a", "b", "c")]


def test_all_states_retained():
    original, native, pointer = arms()
    native[0].update(status="ok", grounded={"next_action": "retrieve_more"})
    cases, states = prepare(combine_triplet(original, native, pointer, size=1))
    assert cases == []
    assert [r["state"] for r in states] == ["pipeline_error", "no_answer_action", "pipeline_error"]


@pytest.mark.parametrize("field", [
    "source_id", "question", "evidence_brief", "current_round", "max_retrieval_rounds",
])
def test_cross_arm_drift_rejected(field):
    values = arms()
    values[2][0][field] = "changed"
    with pytest.raises(AssertionError, match=field):
        combine_triplet(*values, size=1)


def test_missing_state_rejected():
    a, b, c = arms()
    with pytest.raises(AssertionError, match="missing evaluation state"):
        combine_triplet(a, b, [], size=1)


def test_duplicate_id_rejected():
    a, b, c = arms()
    c[0]["id"] = a[0]["id"]
    with pytest.raises(AssertionError, match="duplicate state id"):
        combine_triplet(a, b, c, size=1)


def test_identical_failed_audit_reused():
    case = {"id": "same", "text": "unchanged"}
    failure = {"id": "same", "status": "error", "error": "do not retry"}
    assert split_requests([case], [case], [failure], lambda c: c["text"]) == ([failure], [])


def test_changed_message_not_reused():
    with pytest.raises(AssertionError, match="different provider input"):
        split_requests([{"id": "same", "text": "new"}], [{"id": "same", "text": "old"}],
                       [{"id": "same", "status": "ok"}], lambda c: c["text"])


def test_controls_not_mutated():
    values = arms()
    original = deepcopy(values)
    combine_triplet(*values, size=1)
    assert values == original


def test_direct_display_used_not_raw_answer():
    a, b, c = arms()
    c[0].update(status="ok", raw_output="not displayed", expanded_payload={"final_answer": "not displayed"},
                grounded={"next_action": "answer_directly", "answer": "displayed answer"})
    cases, states = prepare(combine_triplet(a, b, c, size=1))
    assert len(cases) == 2 and len(states) == 3
    assert all(case["shown_answer"] == "displayed answer" for case in cases)


def test_identical_display_deduplicates():
    values = arms()
    for arm in values:
        arm[0].update(status="ok", grounded={"next_action": "answer_directly", "answer": "same"})
    cases, states = prepare(combine_triplet(*values, size=1))
    assert len(cases) == 2 and len(states) == 3
    assert all(len(case["members"]) == 3 for case in cases)
