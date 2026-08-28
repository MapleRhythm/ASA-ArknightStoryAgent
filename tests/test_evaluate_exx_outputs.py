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
