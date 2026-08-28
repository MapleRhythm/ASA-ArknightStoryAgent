import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_exx_outputs.py"
SPEC = importlib.util.spec_from_file_location("evaluate_exx_outputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_reference_fact_similarity_rewards_exact_then_partial() -> None:
    exact = MODULE.reference_fact_similarity(["阿米娅带领罗德岛撤离。"], ["阿米娅带领罗德岛撤离。"])
    partial = MODULE.reference_fact_similarity(["阿米娅带领队伍前进。"], ["阿米娅带领罗德岛撤离。"])
    unrelated = MODULE.reference_fact_similarity(["天气晴朗。"], ["阿米娅带领罗德岛撤离。"])

    assert exact == 1.0
    assert exact > partial > unrelated


def test_cited_evidence_ids_flattens_supported_facts() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "a", "evidence_ids": ["E1", "E2"]},
            {"fact": "b", "evidence_ids": ["E2", "E3"]},
        ],
    }

    assert MODULE.cited_evidence_ids(payload) == {"E1", "E2", "E3"}


def test_claim_citation_alignment_requires_claim_local_ids() -> None:
    expected = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "阿米娅同意申请。", "evidence_ids": ["E1"]},
            {"fact": "杜宾安排训练。", "evidence_ids": ["E2"]},
        ],
    }
    exact = expected
    swapped = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "阿米娅同意申请。", "evidence_ids": ["E2"]},
            {"fact": "杜宾安排训练。", "evidence_ids": ["E1"]},
        ],
    }

    assert MODULE.cited_evidence_ids(exact) == MODULE.cited_evidence_ids(swapped)
    assert MODULE.claim_citation_alignment(exact, expected) == 1.0
    assert MODULE.claim_citation_alignment(swapped, expected) < 1.0


def test_claim_matching_uses_exact_maximum_not_greedy_pairs() -> None:
    assert MODULE.maximum_bipartite_score([[9, 9], [6, 8]]) == 17


def test_schema_rejects_duplicate_facts() -> None:
    payload = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "同一事实", "evidence_ids": ["E1"]},
            {"fact": "同一事实。", "evidence_ids": ["E1"]},
        ],
    }

    assert "fact_2_duplicate" in MODULE.validate_payload(payload, {"E1"})


def test_schema_strictly_validates_retrieve_and_abstain_payloads() -> None:
    incomplete_retrieve = {
        "next_action": "retrieve_more",
        "follow_up_hypothesis": {"question": "还需要什么？"},
    }
    empty_abstain = {"next_action": "abstain", "reason": ""}

    assert "follow_up_schema" in MODULE.validate_payload(incomplete_retrieve, {"E1"})
    assert "abstain_reason" in MODULE.validate_payload(empty_abstain, {"E1"})
