import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_relevance_clean_exx_data.py"
SPEC = importlib.util.spec_from_file_location("build_relevance_clean_exx_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cleaned_payload_drops_supported_irrelevant_facts() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "甲完成任务。", "evidence_ids": ["E1"]},
            {"fact": "乙在白天离开。", "evidence_ids": ["E2"]},
        ],
    }
    judgement = {
        "facts": [
            {"support": "entailed", "question_relevance": "direct", "citation_complete": True},
            {"support": "entailed", "question_relevance": "irrelevant", "citation_complete": True},
        ],
        "set_support": "complete",
        "context_sufficiency": "sufficient",
        "action_appropriateness": "appropriate",
    }
    # The full audit validator also requires the fields omitted from this
    # focused unit fixture; the subset operation itself is tested separately.
    assert MODULE.propose_repair(payload, judgement) == {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲完成任务。", "evidence_ids": ["E1"]}],
    }


def test_cleaned_payload_excludes_unsupported_only_answer() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲完成任务。", "evidence_ids": ["E1"]}],
    }
    assert MODULE.cleaned_payload(
        payload,
        {
            "facts": [{"support": "contradicted", "question_relevance": "direct"}],
            "set_support": "contradicted",
            "context_sufficiency": "insufficient",
            "action_appropriateness": "inappropriate",
            "relations": [],
            "missing_requirements": [],
            "critical_unsupported_claims": 1,
            "irrelevant_claims": 0,
            "protocol": "asa_exx_set_grounding_audit_v3",
        },
    ) is None


def test_heldout_overlap_checks_source_id_and_normalized_question() -> None:
    heldout = [
        {
            "id": "heldout-1",
            "meta": json.dumps({"source_id": "source-1"}),
            "conversations": [{"value": "问题：谁完成了任务？"}],
        }
    ]
    same_source = {
        "id": "other-id",
        "meta": json.dumps({"source_id": "source-1"}),
        "conversations": [{"value": "问题：完全不同的问法"}],
    }
    same_question = {
        "id": "other-source",
        "meta": "{}",
        "conversations": [{"value": "问题：谁完成了任务？"}],
    }
    assert MODULE.heldout_overlap(same_source, heldout)
    assert MODULE.heldout_overlap(same_question, heldout)
