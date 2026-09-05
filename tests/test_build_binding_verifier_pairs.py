from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_binding_verifier_pairs.py"
)
SPEC = importlib.util.spec_from_file_location("build_binding_verifier_pairs", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_classify_with_ambiguity_adjudication() -> None:
    judgement = {
        "cited_eids": ["E1", "E2"],
        "merged": {"verdict": "unsupported"},
        "individual": {
            "E1": {"verdict": "supported"},
            "E2": {"verdict": "unsupported"},
        },
    }
    assert MODULE.classify_with_adjudication(judgement, None) == ("unsupported", [])
    assert MODULE.classify_with_adjudication(
        judgement,
        {"label": "supported_by_some", "keep_eids": ["E1"]},
    ) == ("supported_by_some", ["E1"])
    assert MODULE.classify_with_adjudication(
        judgement,
        {"label": "ambiguous", "keep_eids": ["E1"]},
    ) == ("ambiguous", [])


def test_build_includes_confirmed_missed_positive(tmp_path: Path, monkeypatch) -> None:
    hardneg = tmp_path / "hardneg.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    judgements = tmp_path / "judgements.jsonl"
    ambiguity = tmp_path / "ambiguity.jsonl"
    out_dir = tmp_path / "out"
    write_jsonl(
        manifest,
        [
            {
                "row_id": "row-1",
                "fact_index": 0,
                "claim": "甲导致乙",
                "cited_texts": {"E1": "原始正证据", "E2": "补充正证据"},
            }
        ],
    )
    write_jsonl(
        judgements,
        [
            {
                "fact_id": "fact-1",
                "row_id": "row-1",
                "fact_index": 0,
                "cited_eids": ["E1"],
                "merged": {"verdict": "supported"},
                "individual": {"E1": {"verdict": "supported"}},
            }
        ],
    )
    write_jsonl(
        ambiguity,
        [
            {
                "fact_id": "fact-1",
                "label": "supported_by_union",
                "keep_eids": ["E1"],
            }
        ],
    )
    write_jsonl(
        hardneg,
        [
            {
                "record_id": "row-1",
                "fact_index": 0,
                "claim": "甲导致乙",
                "query_type": "causality",
                "hard_negatives": [
                    {
                        "eid": "E9",
                        "text": "只谈甲，不支持因果",
                        "glm_verdict": "unsupported",
                    }
                ],
                "suspected_missed_positives": [
                    {
                        "eid": "E2",
                        "text": "补充正证据",
                        "glm_verdict": "supported",
                        "glm_reason": "直接支持",
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--hardneg",
            str(hardneg),
            "--manifest",
            str(manifest),
            "--judgements",
            str(judgements),
            "--ambiguity",
            str(ambiguity),
            "--include-suspected-missed-positives",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert MODULE.main() == 0
    pairs = MODULE.read_jsonl(out_dir / "pairwise.jsonl")
    assert len(pairs) == 2
    assert {row["positive_type"] for row in pairs} == {
        "gold",
        "suspected_missed_positive",
    }
    assert {tuple(row["positive_chain"]) for row in pairs} == {("E1",), ("E2",)}
