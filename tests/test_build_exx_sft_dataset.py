import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_exx_sft_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_exx_sft_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_renumber_evidence_and_payload_closes_prompt_label_gaps() -> None:
    evidence = "[E1]\n证据甲\n[E3]\n证据乙\n[E7]\n证据丙"
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "事实乙", "evidence_ids": ["E3"]},
            {"fact": "事实丙", "evidence_ids": ["E7", "E1"]},
        ],
    }

    actual_evidence, actual_payload = MODULE.renumber_evidence_and_payload(evidence, payload)

    assert actual_evidence == "[E1]\n证据甲\n[E2]\n证据乙\n[E3]\n证据丙"
    assert actual_payload["supported_facts"] == [
        {"fact": "事实乙", "evidence_ids": ["E2"]},
        {"fact": "事实丙", "evidence_ids": ["E3", "E1"]},
    ]
    assert payload["supported_facts"][0]["evidence_ids"] == ["E3"]


def test_canonicalize_rejects_multiturn_grounded_rows() -> None:
    record = {
        "id": "multi",
        "task_type": MODULE.ANSWER_TASK,
        "conversations": [
            {"from": "human", "value": "question: 问题\nhypothesis: {}\nround: 1/1\nevidence:\n[E1]\n证据\noutput_schema: old\n"},
            {"from": "gpt", "value": "{}"},
            {"from": "gpt", "value": "{}"},
        ],
    }

    try:
        MODULE.canonicalize_grounded_record(record)
    except ValueError as exc:
        assert "exactly one user" in str(exc)
    else:
        raise AssertionError("multiturn grounded row was silently truncated")
