from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_binding_hardneg.py"
SPEC = importlib.util.spec_from_file_location("audit_binding_hardneg", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source_row(text: str) -> dict:
    return {
        "id": "row-1",
        "conversations": [
            {
                "value": (
                    "question: 测试\n"
                    "evidence:\n"
                    f"[E1]\n{text}\n"
                    "output_schema: grounded_action_exx_v1\n"
                )
            }
        ],
    }


def test_audit_quarantines_judged_prefix_when_source_continues() -> None:
    full = "前文" * 1000 + "尾部支持"
    hardneg = [
        {
            "record_id": "row-1",
            "fact_index": 0,
            "claim": "尾部支持",
            "hard_negatives": [
                {
                    "eid": "E1",
                    "text": full[:2000],
                    "glm_verdict": "unsupported",
                }
            ],
            "suspected_missed_positives": [],
        }
    ]
    report, quarantine = MODULE.build_report(hardneg, [source_row(full)])

    assert report["validation_errors"] == []
    assert report["candidates"] == 1
    assert report["judged_truncated_candidates"] == 1
    assert report["truncated_by_verdict"] == {"unsupported": 1}
    assert report["conservative_full_input_by_verdict"] == {}
    assert quarantine[0]["shown_chars"] == 2000
    assert quarantine[0]["full_chars"] == len(full)
