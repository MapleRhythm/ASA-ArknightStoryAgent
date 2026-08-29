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
