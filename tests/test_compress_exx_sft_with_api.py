import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compress_exx_sft_with_api.py"
SPEC = importlib.util.spec_from_file_location("compress_exx_sft_with_api", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(payload: dict, *, facts: str = "[E1]\n证据") -> dict:
    return {
        "id": "row-1",
        "task_type": "grounded_action_generation",
        "system": MODULE.EXX_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": f"question: 问题\nevidence:\n{facts}"},
            {"from": "gpt", "value": json.dumps(payload, ensure_ascii=False)},
        ],
    }


def test_validate_accepts_minimal_answer_and_visible_ids() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "这是事实", "evidence_ids": ["E1"]}],
    }
    assert MODULE.validate_payload(payload, "question: 问题\nevidence:\n[E1]\n证据") == []


def test_validate_rejects_unknown_evidence_id() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "这是事实", "evidence_ids": ["E2"]}],
    }
    assert MODULE.validate_payload(payload, "question: 问题\nevidence:\n[E1]\n证据") == [
        "fact_1_evidence_ids"
    ]


def test_process_row_copies_short_target_without_api(monkeypatch) -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "这是事实", "evidence_ids": ["E1"]}],
    }
    record = row(payload)
    args = type("Args", (), {"max_facts": 4})()
    monkeypatch.setattr(MODULE, "call_api", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    result = MODULE.process_row(record, args)
    assert result["status"] == "copied"
    assert result["row"] == record

