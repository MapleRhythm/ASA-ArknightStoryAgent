import copy
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_set_audit_progress.py"
SPEC = importlib.util.spec_from_file_location("freeze_set_audit_progress", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture():
    row = {
        "id": "source", "system": "system",
        "conversations": [{"value": "question: 谁完成任务？\nround: 1/2\nevidence:\n[E1]\n甲完成任务。\noutput_schema: grounded_action_exx_v1\n"}],
        "raw_output": '{"next_action":"answer_directly","supported_facts":[{"fact":"甲完成任务。","evidence_ids":["E1"]}]}',
    }
    contract = {"endpoint": "https://judge.invalid", "model": "judge", "max_tokens": 4096}
    result = {
        "id": row["id"], "status": "ok", "finish_reason": "stop", "full_evidence": True,
        "input_key": MODULE.input_key(row, **contract),
        "audited_input_sha256": MODULE.audited_input_sha256(row),
        "judgement": {
            "protocol": MODULE.PROTOCOL,
            "facts": [{"fact_index": 0, "support": "entailed", "question_relevance": "direct",
                       "checked_evidence_ids": ["E1"], "citation_complete": True}],
            "relations": [], "set_support": "complete", "missing_requirements": [],
            "critical_unsupported_claims": 0, "irrelevant_claims": 0,
            "context_sufficiency": "sufficient", "action_appropriateness": "appropriate",
        },
    }
    return row, result, contract


def test_pending_rows_cannot_be_counted_as_reviewed_or_approved():
    row, result, contract = fixture()
    missing = copy.deepcopy(row)
    missing["id"] = "pending"
    frozen = MODULE.freeze([row, missing], [result], **contract)
    assert frozen["summary"]["ok"] == 1
    assert frozen["summary"]["pending"] == 1
    assert frozen["summary"]["complete_audit"] is False
    assert frozen["results"][1]["status"] == "pending"


def test_other_contract_and_superseded_error_do_not_poison_success():
    row, result, contract = fixture()
    records = [
        {**result, "status": "error", "error": "old failure"},
        {**result, "input_key": "different budget"},
        result,
    ]
    frozen = MODULE.freeze([row], records, **contract)
    assert frozen["summary"]["ignored_other_contract_records"] == 1
    assert frozen["summary"]["ok"] == 1
    assert frozen["summary"]["errors"] == 0


@pytest.mark.parametrize("change", ["sha", "truncated", "invalid_schema"])
def test_stale_or_invalid_success_is_rejected(change):
    row, result, contract = fixture()
    if change == "sha":
        result["audited_input_sha256"] = "stale"
    elif change == "truncated":
        result["finish_reason"] = "length"
    else:
        result["judgement"]["facts"][0]["checked_evidence_ids"] = ["E2"]
    with pytest.raises(ValueError):
        MODULE.freeze([row], [result], **contract)
