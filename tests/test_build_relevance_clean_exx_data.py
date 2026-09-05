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
            {"support": "entailed", "question_relevance": "direct"},
            {"support": "entailed", "question_relevance": "irrelevant"},
        ]
    }
    assert MODULE.cleaned_payload(payload, judgement) == {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲完成任务。", "evidence_ids": ["E1"]}],
    }


def test_cleaned_payload_excludes_unsupported_only_answer() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲完成任务。", "evidence_ids": ["E1"]}],
    }
    assert MODULE.cleaned_payload(
        payload, {"facts": [{"support": "contradicted", "question_relevance": "direct"}]}
    ) is None
