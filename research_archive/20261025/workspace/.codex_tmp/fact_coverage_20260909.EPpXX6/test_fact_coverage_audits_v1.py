from copy import deepcopy

from audit_fact_coverage_pairs_v1 import audit_messages, prepare, summarize


def sample(rid, answer="收到明确指令后执行。", action="answer_directly"):
    return {
        "id": rid, "renderer": "hidden_variant", "question": "为何执行？", "status": "ok",
        "grounded": {"next_action": action, "answer": answer},
        "evidence_brief": "[E1] 员工收到明确指令后执行。",
        "current_round": 1, "max_retrieval_rounds": 2,
    }


def test_same_inputs_reused_without_merging_different_evidence_semantics():
    a, b = sample("a"), sample("b")
    c = sample("c")
    c["evidence_brief"] = "[E1] 员工擅自执行。"
    cases, states = prepare({"results": [a, b, c]})
    assert len(cases) == 3  # one coverage input, two source inputs
    assert states[0]["coverage_group_id"] == states[2]["coverage_group_id"]
    assert states[0]["semantics_group_id"] != states[2]["semantics_group_id"]
    assert states[0]["semantics_group_id"] == states[1]["semantics_group_id"]


def test_no_answer_and_failures_retained_without_api_calls():
    failed = sample("b")
    failed["status"] = "parse_or_validation_error"
    cases, states = prepare({"results": [sample("a", "", "retrieve_more"), failed]})
    assert cases == []
    labeled, summary = summarize(states, [])
    assert len(labeled) == 2
    assert summary["hidden_variant"]["states"] == 2
    assert summary["hidden_variant"]["coverage"] == {"no_answer_action": 1, "pipeline_error": 1}


def test_variant_diagnostics_and_alternative_never_sent():
    row = sample("secret_parent_id")
    row["parsed"] = {"answer": "secret_previous_answer"}
    row["grounded"]["grounding_warnings"] = ["secret_warning"]
    cases, _ = prepare({"results": [row]})
    import json
    for case in cases:
        visible = json.dumps(audit_messages(case), ensure_ascii=False)
        assert not any(word in visible for word in (
            "hidden_variant", "secret_parent_id", "secret_previous_answer", "secret_warning",
        ))
        payload = json.loads(audit_messages(case)[1]["content"])
        if case["audit_kind"] == "coverage":
            assert set(payload) == {"question", "shown_answer"}
        else:
            assert set(payload) == {"question", "round", "max_rounds", "evidence", "candidate"}


def test_audit_error_is_neither_wrong_answer_nor_removed():
    cases, states = prepare({"results": [sample("a")]})
    results = [{"id": c["id"], "status": "error", "error": "malformed"} for c in cases]
    labeled, summary = summarize(states, results)
    assert labeled[0]["coverage_label"] == labeled[0]["semantics_label"] == "audit_error"
    assert summary["hidden_variant"]["states"] == 1


def test_preparation_does_not_change_input_or_leak_fact_only_content():
    row = sample("a")
    row["grounded"]["supported_facts"] = [{"fact": "secret_unshown_fact"}]
    before = deepcopy(row)
    cases, _ = prepare({"results": [row]})
    assert row == before
    for case in cases:
        assert "supported_facts" not in case["grounded"]
