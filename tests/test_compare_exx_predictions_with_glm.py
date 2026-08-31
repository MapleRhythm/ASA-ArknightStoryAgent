import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_exx_predictions_with_glm.py"
SPEC = importlib.util.spec_from_file_location("compare_exx_predictions_with_glm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_predictions(path: Path, ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": row_id,
                    "conversations": [
                        {"from": "human", "value": "prompt"},
                        {"from": "gpt", "value": "{}"},
                    ],
                }
                for row_id in ids
            ]
        ),
        encoding="utf-8",
    )


def test_blind_order_is_stable_and_row_specific() -> None:
    names = ["sft", "rule", "glm"]

    assert MODULE.blind_order("row-a", names, 7) == MODULE.blind_order("row-a", names, 7)
    assert set(MODULE.blind_order("row-a", names, 7)) == set(names)
    assert MODULE.blind_order("row-a", names, 7) != MODULE.blind_order("row-b", names, 7)


def test_aligned_rows_rejects_id_mismatch(tmp_path: Path) -> None:
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_predictions(left, ["a", "b"])
    write_predictions(right, ["a", "c"])

    with pytest.raises(ValueError, match="not ID-aligned"):
        MODULE.aligned_rows([("left", left), ("right", right)])


def test_paired_bootstrap_ci_is_deterministic_and_contains_mean() -> None:
    first = MODULE.paired_bootstrap_ci([0.5, -0.25, 0.75], samples=1000, seed=3)
    second = MODULE.paired_bootstrap_ci([0.5, -0.25, 0.75], samples=1000, seed=3)

    assert first == second
    assert first["samples"] == 3
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]


def test_build_summary_reports_pairwise_and_long_ready_fields() -> None:
    completed = [
        {
            "models": {
                "old": {
                    "score": 0.0,
                    "eligible": True,
                    "judged": True,
                    "protocol_adjusted_score": 0.0,
                    "judgement": None,
                },
                "new": {
                    "score": 0.5,
                    "eligible": True,
                    "judged": True,
                    "protocol_adjusted_score": 0.5,
                    "judgement": None,
                },
            }
        }
    ]

    summary = MODULE.build_summary(
        completed, ["new", "old"], bootstrap_samples=100, seed=7
    )

    pair = summary["pairwise"]["new_vs_old"]
    assert pair["semantic_left_wins"] == 1
    assert pair["semantic_delta"]["mean"] == 0.5
    assert pair["protocol_adjusted_delta"]["ci95_low"] == 0.5
