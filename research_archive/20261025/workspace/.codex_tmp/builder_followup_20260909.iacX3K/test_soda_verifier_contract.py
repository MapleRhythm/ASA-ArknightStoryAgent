"""Offline regression tests; no API, dataset rebuild or model load."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(os.environ.get(
    "ASA_VERIFIER_SCRIPT",
    str(Path(__file__).resolve().parents[1] / "scripts/build_soda_api_verifier_dataset.py"),
))
spec = importlib.util.spec_from_file_location("soda_verifier_contract_target", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def grounded(answer="甲去了城里。", quote="甲去了城里。", evidence_id="E1"):
    return {
        "next_action": "answer_directly",
        "supported_facts": [{
            "fact": answer, "evidence_refs": [{"evidence_id": evidence_id, "quote": quote}],
        }],
        "inferred_facts": [],
        "final_answer": answer,
    }


def row(tag=True, payload=None, question="甲去了哪里？", round_id=1, key="state1", evidence=None):
    prompt = (
        f"question: {question}\nround: {round_id}\nevidence_brief:\n"
        + (evidence if evidence is not None else "[E1] 甲去了城里。\n[E2] 乙留在村里。")
        + "\noutput_schema: grounded_action_v1\n"
    )
    return {
        "id": f"{key}-{'teacher' if tag else 'student'}",
        "task_type": "conclusion_generation",
        "kto_tag": tag,
        "meta": {"prompt_key": key},
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json.dumps(
                grounded() if payload is None else payload, ensure_ascii=False,
            )},
        ],
    }


def verdict(**changes):
    result = {
        "correct_action": "answer_directly", "evidence_sufficient": True,
        "use_for_training": True, "teacher_action_error": "none",
        "teacher_answer_uses_prior_knowledge": False, "student_action_error": "none",
        "supported_answer": "甲去了城里。", "missing_slots": [],
    }
    result.update(changes)
    return {"verifier": result}


def relabel(group, **changes):
    return builder.relabel_group("state1", group, verdict(**changes), Counter())


def chosen(records):
    return [json.loads(builder.response_text(r)) for r in records if r["kto_tag"] is True]


@pytest.mark.parametrize("quote", ["甲" * 81, " 前文\t  引文\n后文 "])
def test_quote_normalization_is_lossless(quote):
    refs = [{"evidence_id": "E1", "quote": quote}]
    assert builder.normalize_evidence_refs(refs) == refs


def test_normalization_does_not_drop_references_or_facts():
    refs = [{"evidence_id": f"E{i}", "quote": "甲" * 80} for i in range(3)]
    assert builder.normalize_evidence_refs(refs) == refs
    facts = [{"fact": f"事实{i}", "evidence_refs": refs} for i in range(7)]
    assert builder.normalize_supported_facts(facts) == facts


def test_inferred_citations_and_fact_ids_survive():
    payload = grounded()
    payload["supported_facts"][0]["fact_id"] = "F1"
    payload["inferred_facts"] = [{
        "fact": "甲可能离开了村子。", "premise_fact_ids": ["F1"],
        "evidence_refs": [{"evidence_id": "E1", "quote": "甲去了城里。"}],
    }]
    assert builder.normalize_grounded_action_payload(payload, question="问", prompt="") == payload


@pytest.mark.parametrize("action", ["answer_directly", "retrieve_more", "abstain", "clarify_user"])
def test_explicit_training_veto_is_not_overridden(action):
    assert relabel([row()], correct_action=action, use_for_training=False) == []


@pytest.mark.parametrize("flag", ["false", None, 1])
def test_non_boolean_training_approval_is_not_true(flag):
    assert relabel([row()], use_for_training=flag) == []


def test_known_bad_teacher_is_never_chosen():
    bad = grounded("甲去了月球。")
    good = grounded()
    result = relabel(
        [row(payload=bad), row(tag=False, payload=good)],
        teacher_action_error="unsupported_answer",
    )
    assert chosen(result) == [good]


def test_teacher_prior_knowledge_is_not_a_positive():
    assert not chosen(relabel([row()], teacher_answer_uses_prior_knowledge=True))


def test_unreviewed_student_is_not_a_positive():
    assert not chosen(relabel(
        [row(payload={"next_action": "retrieve_more"}), row(tag=False)],
        teacher_action_error="over_retrieve", student_action_error="unsupported_answer",
    ))


def test_no_positive_from_inconsistent_evidence_verdict():
    assert not chosen(relabel([row()], evidence_sufficient=False))


@pytest.mark.parametrize("change", [
    {"supported_facts": []},
    {"supported_facts": [{"fact": "甲去了城里。", "evidence_refs": []}]},
    {"supported_facts": [{"fact": "", "evidence_refs": [{"evidence_id": "E1", "quote": "甲去了城里。"}]}]},
    {"inferred_facts": ["甲可能住在城里。"]},
    {"supported_facts": [{"fact": "甲去了城里。", "evidence_refs": [{"evidence_id": [], "quote": "甲去了城里。"}]}]},
    {"supported_facts": [{"fact": "甲去了城里。", "evidence_refs": [{"evidence_id": "E1", "quote": 123}]}]},
    {"final_answer": ""},
])
def test_incomplete_direct_payload_never_becomes_positive(change):
    payload = grounded()
    payload.update(change)
    assert not chosen(relabel([row(payload=payload)]))


@pytest.mark.parametrize("payload", [
    grounded(quote="甲" * 81),
    grounded(quote="乙留在村里。", evidence_id="E1"),
    grounded(evidence_id="E99"),
    grounded(quote="证据之外的文本"),
])
def test_invalid_quotes_are_rejected_not_repaired(payload):
    assert not chosen(relabel([row(payload=payload)]))


def test_inferred_quote_is_checked_in_its_own_visible_block():
    payload = grounded()
    payload["inferred_facts"] = [{
        "fact": "乙可能没有离开。",
        "evidence_refs": [{"evidence_id": "E1", "quote": "乙留在村里。"}],
    }]
    assert not chosen(relabel([row(payload=payload)]))


@pytest.mark.parametrize("count", [3, 7])
def test_over_budget_content_is_not_silently_pruned(count):
    payload = grounded(quote="甲" * 80)
    payload["supported_facts"] *= count
    if count == 3:
        for fact in payload["supported_facts"]:
            fact["evidence_refs"] *= 2
    assert not chosen(relabel([row(payload=payload, evidence="[E1] " + "甲" * 80)]))


def test_supported_and_inferred_quotes_share_total_budget():
    payload = grounded(quote="甲" * 80)
    payload["supported_facts"] *= 3
    payload["inferred_facts"] = deepcopy(payload["supported_facts"])
    assert not chosen(relabel([row(payload=payload, evidence="[E1] " + "甲" * 80)]))


def test_valid_candidate_is_kept():
    assert chosen(relabel([row()])) == [grounded()]


def test_legacy_numbered_evidence_blocks_are_supported():
    assert chosen(relabel([row(evidence="1. E1: 甲去了城里。\n2. E2: 乙留在村里。")])) == [grounded()]


def test_duplicate_evidence_id_is_ambiguous():
    assert not chosen(relabel([row(evidence="[E1] 别的内容。\n[E1] 甲去了城里。")]))


def test_fenced_json_is_not_repaired_into_positive():
    record = row()
    record["conversations"][-1]["value"] = "```json\n" + builder.response_text(record) + "\n```"
    assert not chosen(relabel([record]))


def test_duplicate_json_keys_are_not_a_positive():
    record = row()
    record["conversations"][-1]["value"] = builder.response_text(record)[:-1] + ', "final_answer": "另一个答案"}'
    assert not chosen(relabel([record]))


def test_negative_is_the_original_reviewed_output():
    payload = grounded(quote="甲" * 100)
    bad = row(payload=payload)
    result = relabel([bad], teacher_action_error="unsupported_answer")
    negatives = [r for r in result if r["kto_tag"] is False]
    assert len(negatives) == 1
    assert negatives[0]["conversations"][-1]["value"] == bad["conversations"][-1]["value"]


def test_invalid_json_reviewed_negative_is_kept_verbatim():
    bad = row()
    bad["conversations"][-1]["value"] = ' {"next_action": '
    result = relabel([bad], teacher_action_error="invalid_output")
    negatives = [r for r in result if r["kto_tag"] is False]
    assert len(negatives) == 1
    assert negatives[0]["conversations"][-1]["value"] == bad["conversations"][-1]["value"]


def test_mismatched_student_input_cannot_supply_teacher_positive():
    result = relabel([
        row(payload={"next_action": "retrieve_more"}),
        row(tag=False, evidence="[E1] 只有学生看见的证据。"),
    ], teacher_action_error="over_retrieve")
    assert result == []


def test_question_states_do_not_cross_train_validation():
    records = [
        row(question="甲 去了哪里？" if n % 2 else "甲去了哪里？", round_id=n, key=f"state{n}")
        for n in range(12)
    ] + [row(question="乙留在哪里？", key="other")]
    train, val = builder.split_records(records, seed=7, val_ratio=0.3)
    questions = lambda rows: {
        "".join(builder.extract_question(builder.prompt_text(r)).split()) for r in rows
    }
    assert questions(train).isdisjoint(questions(val))
    assert len(train) + len(val) == len(records)


def test_same_input_output_cannot_have_opposite_kto_labels():
    with pytest.raises(ValueError, match="conflict"):
        builder.split_records([row(), row(tag=False)], seed=7, val_ratio=0.3)


def test_semantically_identical_json_cannot_have_opposite_labels():
    positive, negative = row(), row(tag=False)
    negative["conversations"][-1]["value"] = json.dumps(grounded(), sort_keys=True, indent=2)
    with pytest.raises(ValueError, match="conflict"):
        builder.split_records([positive, negative], seed=7, val_ratio=0.3)


def test_split_is_repeatable():
    records = [row(question=f"问题{n}？", key=f"state{n}") for n in range(10)]
    assert builder.split_records(records, seed=7, val_ratio=0.2) == builder.split_records(
        records, seed=7, val_ratio=0.2,
    )


def test_one_question_does_not_leave_training_empty():
    train, val = builder.split_records([row()], seed=7, val_ratio=0.9)
    assert len(train) == 1 and val == []


def test_fallback_groups_equal_inputs_without_question_field():
    first = row(payload={"next_action": "retrieve_more"})
    second = row(tag=False)
    for record in (first, second):
        record["conversations"][0]["value"] = "没有question字段的同一输入"
    other = row(question="另一个问题？", key="other")
    train, val = builder.split_records([first, second, other], seed=7, val_ratio=0.4)
    assert (first in train) == (second in train)
    assert len(train) + len(val) == 3


def test_build_gate_also_checks_explicitly_kept_unverified_positives(tmp_path):
    invalid = row(payload={"next_action": "answer_directly", "answer": "无结构引用的旧格式"})
    args = SimpleNamespace(
        input_dir=tmp_path, keep_unverified_conclusion=True,
        seed=7, val_ratio=0.2, dataset_name="synthetic",
    )
    summary = builder.build_dataset(args, tmp_path, {"state1": [invalid]}, {})
    assert summary["output_records"] == 0
    assert summary["training_ready"] is False


def test_build_reports_question_disjointness(tmp_path):
    groups = {
        f"state{n}": [row(question=f"问题{n // 2}？", key=f"state{n}", round_id=n)]
        for n in range(12)
    }
    args = SimpleNamespace(
        input_dir=tmp_path, keep_unverified_conclusion=False,
        seed=7, val_ratio=0.2, dataset_name="synthetic",
    )
    summary = builder.build_dataset(
        args, tmp_path, groups, {key: verdict() for key in groups},
    )
    assert summary["output_records"] == 12
    assert summary["train_val_question_overlap"] == 0
    assert summary["training_ready"] is False


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.1, float("nan")])
def test_invalid_validation_ratio_fails(ratio):
    with pytest.raises(ValueError, match="val_ratio"):
        builder.split_records([row()], seed=7, val_ratio=ratio)
