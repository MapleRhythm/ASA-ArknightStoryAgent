import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_exx_calibrated.py"
SPEC = importlib.util.spec_from_file_location("evaluate_exx_calibrated", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_apply_calibration_drops_unsupported_and_trims_partial() -> None:
    original = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "甲做了A。", "evidence_ids": ["E1"]},
            {"fact": "乙做了B。", "evidence_ids": ["E2", "E3"]},
        ],
    }
    changes = [
        {"fact_index": 0, "change_type": "drop_fact", "label": "unsupported"},
        {
            "fact_index": 1,
            "change_type": "drop_evidence_ids",
            "label": "partial",
            "kept_evidence_ids": ["E3"],
        },
    ]
    calibrated, ignored, labels = MODULE.apply_calibration(original, changes, [])
    assert calibrated == {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "乙做了B。", "evidence_ids": ["E3"]}],
    }
    assert ignored == set()
    assert labels == ["unsupported", "partial"]


def test_ambiguity_is_not_counted_as_hard_binding_target() -> None:
    original = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲做了A。", "evidence_ids": ["E1"]}],
    }
    ambiguity = [
        {
            "fact_index": 0,
            "label": "ambiguous",
            "keep_eids": ["E1"],
        }
    ]
    calibrated, ignored, _ = MODULE.apply_calibration(original, [], ambiguity)
    assert calibrated["supported_facts"][0]["evidence_ids"] == ["E1"]
    assert ignored == {0}


def test_action_flip_removes_answer_facts() -> None:
    original = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "甲做了A。", "evidence_ids": ["E1"]}],
    }
    changes = [
        {"change_type": "action_flip", "new_action": "retrieve_more"},
    ]
    calibrated, _, _ = MODULE.apply_calibration(original, changes, [])
    assert calibrated == {"next_action": "retrieve_more"}
