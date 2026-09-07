from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_binding_hardneg_full_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_binding_hardneg_full_evidence", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_reports_valid_transition(tmp_path: Path, monkeypatch) -> None:
    hardneg = tmp_path / "hardneg.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    rejudgements = tmp_path / "rejudgements.jsonl"
    report = tmp_path / "report.json"
    full_text = "旧前缀" + "完整尾部"
    write_jsonl(
        hardneg,
        [
            {
                "record_id": "row-1",
                "fact_index": 0,
                "claim": "甲导致乙",
                "hard_negatives": [
                    {
                        "eid": "E1",
                        "text": "旧前缀",
                        "glm_verdict": "unsupported",
                    }
                ],
            }
        ],
    )
    write_jsonl(
        quarantine,
        [
            {
                "record_id": "row-1",
                "fact_index": 0,
                "claim": "甲导致乙",
                "eid": "E1",
                "candidate_role": "hard_negative",
            }
        ],
    )
    write_jsonl(
        rejudgements,
        [
            {
                "record_id": "row-1",
                "fact_index": 0,
                "claim": "甲导致乙",
                "eid": "E1",
                "candidate_role": "hard_negative",
                "text": full_text,
                "evidence_chars": len(full_text),
                "evidence_sha256": MODULE.fingerprint(
                    "row-1", 0, "甲导致乙", "E1", full_text
                ),
                "full_evidence": True,
                "status": "ok",
                "glm_verdict": "supported",
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--hardneg",
            str(hardneg),
            "--quarantine",
            str(quarantine),
            "--rejudgements",
            str(rejudgements),
            "--report",
            str(report),
        ],
    )
    assert MODULE.main() == 0
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["completed"] == 1
    assert value["transition_matrix"] == {"unsupported -> supported": 1}
    assert value["validation_errors"] == []
