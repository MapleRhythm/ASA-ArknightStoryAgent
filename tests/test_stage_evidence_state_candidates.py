import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/stage_evidence_state_candidates.py"
SPEC = importlib.util.spec_from_file_location("stage_evidence_state_candidates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source():
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲没有同意。", "evidence_ids": ["E1"]}],
    }
    return {
        "id": "train-source", "system": "original system",
        "meta": {"source_split": "train"},
        "conversations": [
            {"from": "human", "value": (
                "task: grounded_action_generation\nquestion: 甲是否同意？\n"
                'hypothesis: {"entities":["不得泄漏的旧规划答案"]}\nround: 1/2\n'
                "evidence:\n[E1]\n甲没有同意。\n[E2]\n乙在山顶散步。\n"
                "output_schema: grounded_action_exx_v1\nrules: old rules\n"
            )},
            {"from": "gpt", "value": MODULE.canonical(payload)},
        ],
        "raw_output": MODULE.canonical(payload),
    }


def test_stages_six_states_without_training_or_hidden_answer_in_missing_prompt():
    original = source()
    frozen = copy.deepcopy(original)
    rows, report = MODULE.stage([(original, "component1")])
    assert original == frozen
    assert len(rows) == 6 and report["components"] == 1
    assert report["training_ready"] is False
    assert {row["state_provenance"]["round"] for row in rows} == {"1/2", "2/2"}
    assert len({row["id"] for row in rows}) == 6
    for row in rows:
        assert row["meta"]["training_eligible"] is False
        assert row["meta"]["requires_reaudit"] is True
        assert "不得泄漏的旧规划答案" not in row["conversations"][0]["value"]
        assert MODULE.source_training_issue(row) == "evaluation_only_source"
        assert row["state_provenance"]["audited_input_sha256"] == MODULE.audited_input_sha256(row)
        if row["state_provenance"]["condition"] == "missing_candidate":
            assert "甲没有同意" not in row["conversations"][0]["value"]
            assert "甲没有同意" not in row["raw_output"]
            assert row["state_provenance"]["source_eid_to_visible_eid"] == {"E2": "E1"}
            expected = "retrieve_more" if row["state_provenance"]["round"] == "1/2" else "abstain"
            assert json.loads(row["raw_output"])["next_action"] == expected
        else:
            assert json.loads(row["raw_output"])["next_action"] == "answer_directly"


def test_overlap_closure_removes_duplicate_and_containing_passages():
    shared = "甲：" + "天边的星星正在照耀荒原上跋涉的旅人们。" * 4
    evidence = {
        "E1": shared, "E2": "场景前缀\n" + shared,
        "E3": "场景前缀\n" + shared + "\n场景后缀",
        "E4": "独立场景，乙在室内。",
    }
    graph = MODULE.overlap_graph(evidence)
    assert MODULE.removal_closure({"E1"}, graph) == {"E1", "E2", "E3"}
    assert graph["E4"] == set()


def test_short_negation_is_not_equated_by_overlap_heuristic():
    graph = MODULE.overlap_graph({"E1": "甲同意。", "E2": "甲没有同意。"})
    assert MODULE.removal_closure({"E1"}, graph) == {"E1"}


def test_empty_missing_context_is_skipped_not_replaced_with_fake_evidence():
    row = source()
    row["conversations"][0]["value"] = row["conversations"][0]["value"].replace(
        "[E2]\n乙在山顶散步。\n", ""
    )
    rows, report = MODULE.stage([(row, "component")])
    assert len(rows) == 2  # Full and distractor were exactly the same.
    assert report["counts"]["missing_skipped_no_nonempty_context"] == 1
    assert report["counts"]["duplicate_visible_state_skipped"] == 2
    assert all("missing" not in item["state_provenance"]["condition"] for item in rows)


def test_same_component_is_never_counted_as_new_family():
    other = source()
    other["id"] = "second"
    rows, report = MODULE.stage([(source(), "component"), (other, "component")])
    assert len(rows) == 6
    assert report["counts"]["skipped_same_component"] == 1
    assert report["component_is_not_independent_semantic_family"] is True


@pytest.mark.parametrize("meta", [
    {"source_split": "validation"}, {"training_eligible": False},
    {"evaluation_only": True}, {"requires_reaudit": True},
])
def test_nontraining_sources_cannot_be_staged(meta):
    row = source()
    row["meta"] = meta
    with pytest.raises(ValueError, match="source_not_train"):
        MODULE.stage([(row, "component")])


def test_remaps_multi_evidence_claims_without_changing_relation():
    row = source()
    row["conversations"][0]["value"] = row["conversations"][0]["value"].replace(
        "output_schema:", "[E3]\n乙的拒绝导致甲拒绝。\noutput_schema:"
    )
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "乙的拒绝导致甲拒绝。", "evidence_ids": ["E1", "E3"]}],
    }
    row["raw_output"] = MODULE.canonical(payload)
    row["conversations"][-1]["value"] = row["raw_output"]
    rows, _ = MODULE.stage([(row, "component")])
    full = next(item for item in rows if item["state_provenance"]["condition"] == "full")
    assert json.loads(full["raw_output"])["supported_facts"] == [
        {"fact": "乙的拒绝导致甲拒绝。", "evidence_ids": ["E1", "E2"]}
    ]


def test_frozen_file_provenance_is_required(tmp_path):
    path = tmp_path / "accepted.json"
    path.write_text("[]")
    info = {"path": str(path), "sha256": MODULE.sha256_file(path)}
    assert MODULE.checked_file(info) == path
    path.write_text("[{}]")
    with pytest.raises(ValueError, match="file_content_mismatch"):
        MODULE.checked_file(info)
    with pytest.raises(ValueError, match="missing_file_provenance"):
        MODULE.checked_file({})
