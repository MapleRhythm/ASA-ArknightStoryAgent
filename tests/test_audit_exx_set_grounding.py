import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_exx_set_grounding.py"
SPEC = importlib.util.spec_from_file_location("audit_exx_set_grounding", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_validate_set_judgement_accepts_joint_evidence() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲完成了任务。", "evidence_ids": ["E1", "E2"]}],
    }
    judgement = {
        "protocol": MODULE.PROTOCOL,
        "facts": [
            {
                "fact_index": 0,
                "support": "entailed",
                "checked_evidence_ids": ["E1", "E2"],
                "citation_complete": True,
            }
        ],
        "relations": [],
        "set_support": "complete",
        "missing_requirements": [],
        "critical_unsupported_claims": 0,
        "context_sufficiency": "sufficient",
        "action_appropriateness": "appropriate",
    }
    assert MODULE.validate_judgement(judgement, payload)["set_support"] == "complete"


def test_validate_set_judgement_rejects_mismatched_claim_ids() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲完成了任务。", "evidence_ids": ["E1"]}],
    }
    judgement = {
        "protocol": MODULE.PROTOCOL,
        "facts": [
            {
                "fact_index": 0,
                "support": "entailed",
                "checked_evidence_ids": ["E2"],
                "citation_complete": True,
            }
        ],
        "relations": [],
        "set_support": "complete",
        "missing_requirements": [],
        "critical_unsupported_claims": 0,
        "context_sufficiency": "sufficient",
        "action_appropriateness": "appropriate",
    }
    try:
        MODULE.validate_judgement(judgement, payload)
    except ValueError as exc:
        assert str(exc) == "set_judge_evidence_ids_mismatch"
    else:
        raise AssertionError("expected evidence ID mismatch")
