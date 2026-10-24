import copy

import pytest

from build_same_evidence_retry_matrix import FEEDBACK, paired_rows, selected_ids


def render(system, user, *, assistant_prefix):
    return system + "\n" + user + "\n" + assistant_prefix


def source():
    return [{
        "id": "x", "system": "system", "user": "question and visible evidence",
        "assistant_prefix": "{", "runtime_prompt": "original",
        "runtime_prompt_sha256": "original-hash", "evidence_brief": "[E1] 原文",
        "hypothesis": {"entities": ["实体"]}, "current_round": 2, "max_retrieval_rounds": 3,
        "prompt_evidence": [{"id": "document", "text": "原文"}],
    }]


def test_only_rejected_answer_proposals_are_selected():
    rows = [
        {"id": "a", "status": "ok", "parsed": {"next_action": "answer_directly"}, "grounded": {"next_action": "retrieve_more"}},
        {"id": "b", "status": "ok", "parsed": {"next_action": "answer_directly"}, "grounded": {"next_action": "answer_directly"}},
        {"id": "c", "status": "ok", "parsed": {"next_action": "abstain"}, "grounded": {"next_action": "abstain"}},
        {"id": "d", "status": "error"},
    ]
    assert selected_ids(rows) == {"a"}


def test_pair_has_identical_evidence_round_and_seed_and_does_not_mutate_source():
    rows = source()
    before = copy.deepcopy(rows)
    result = paired_rows(rows, {"x"}, render)
    assert rows == before and len(result) == 2
    assert result[0]["sampling_seed"] == result[1]["sampling_seed"]
    for row in result:
        for key in ("evidence_brief", "hypothesis", "current_round", "max_retrieval_rounds", "prompt_evidence"):
            assert row[key] == before[0][key]


def test_control_is_byte_identical_feedback_changes_user_suffix_only():
    rows = paired_rows(source(), {"x"}, render)
    by_kind = {r["renderer"]: r for r in rows}
    assert by_kind["retry_control_v1"]["runtime_prompt"] == "original"
    feedback = by_kind["retry_quote_feedback_v1"]
    assert feedback["user"] == source()[0]["user"] + "\n" + FEEDBACK
    assert feedback["system"] == source()[0]["system"]
    assert feedback["assistant_prefix"] == "{"


def test_rejects_missing_or_duplicate_matrix_rows():
    with pytest.raises(ValueError, match="absent"):
        paired_rows(source(), {"other"}, render)
    with pytest.raises(ValueError, match="duplicate"):
        paired_rows(source() * 2, {"x"}, render)
