import importlib.util
import sys
import json
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
                "question_relevance": "direct",
                "checked_evidence_ids": ["E1", "E2"],
                "citation_complete": True,
            }
        ],
        "relations": [],
        "set_support": "complete",
        "missing_requirements": [],
        "critical_unsupported_claims": 0,
        "irrelevant_claims": 0,
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
                "question_relevance": "direct",
                "checked_evidence_ids": ["E2"],
                "citation_complete": True,
            }
        ],
        "relations": [],
        "set_support": "complete",
        "missing_requirements": [],
        "critical_unsupported_claims": 0,
        "irrelevant_claims": 0,
        "context_sufficiency": "sufficient",
        "action_appropriateness": "appropriate",
    }
    try:
        MODULE.validate_judgement(judgement, payload)
    except ValueError as exc:
        assert str(exc) == "set_judge_evidence_ids_mismatch"
    else:
        raise AssertionError("expected evidence ID mismatch")


def test_validate_set_judgement_rejects_complete_with_irrelevant_fact() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "甲完成任务。", "evidence_ids": ["E1"]},
            {"fact": "乙在白天离开。", "evidence_ids": ["E2"]},
        ],
    }
    judgement = {
        "protocol": MODULE.PROTOCOL,
        "facts": [
            {
                "fact_index": 0,
                "support": "entailed",
                "question_relevance": "direct",
                "checked_evidence_ids": ["E1"],
                "citation_complete": True,
            },
            {
                "fact_index": 1,
                "support": "entailed",
                "question_relevance": "irrelevant",
                "checked_evidence_ids": ["E2"],
                "citation_complete": True,
            },
        ],
        "relations": [],
        "set_support": "complete",
        "missing_requirements": [],
        "critical_unsupported_claims": 0,
        "irrelevant_claims": 1,
        "context_sufficiency": "sufficient",
        "action_appropriateness": "appropriate",
    }
    try:
        MODULE.validate_judgement(judgement, payload)
    except ValueError as exc:
        assert str(exc) == "invalid_complete_set_support"
    else:
        raise AssertionError("expected complete-support rejection")


def test_judge_records_server_model_without_claiming_backend_verification(monkeypatch):
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲完成任务。", "evidence_ids": ["E1"]}],
    }
    row = {
        "id": "source",
        "conversations": [{"value": "question: 谁完成任务？\nround: 1/2\nevidence:\n[E1]\n甲完成任务。\noutput_schema: grounded_action_exx_v1\n"}],
        "raw_output": json.dumps(payload, ensure_ascii=False),
    }
    judgement = {
        "protocol": MODULE.PROTOCOL,
        "facts": [{"fact_index": 0, "support": "entailed", "question_relevance": "direct",
                   "checked_evidence_ids": ["E1"], "citation_complete": True}],
        "relations": [], "set_support": "complete", "missing_requirements": [],
        "critical_unsupported_claims": 0, "irrelevant_claims": 0,
        "context_sufficiency": "sufficient", "action_appropriateness": "appropriate",
    }
    response = {
        "model": "server-reported-model", "id": "response-1",
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(judgement)}}],
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(response).encode()

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", lambda *a, **kw: FakeResponse())
    result = MODULE.judge_row(row, "https://judge.invalid", "test-not-a-key", "requested-model", 1)
    assert result["status"] == "ok"
    assert result["requested_model"] == "requested-model"
    assert result["response_model"] == "server-reported-model"
    assert result["response_id"] == "response-1"
