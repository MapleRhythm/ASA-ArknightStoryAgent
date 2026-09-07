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
    assert (
        MODULE.contradiction_kind(judgement)
        == "merged_unsupported_ind_supported"
    )
    assert "supported_by_some" not in MODULE.POSITIVE_GOLD_STATUSES


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
                "chunk_chars": 100,
                "merged": {"status": "ok", "verdict": "supported"},
                "individual": {
                    "E1": {"status": "ok", "verdict": "supported"}
                },
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


def test_build_drops_gold_when_judge_input_was_truncated(
    tmp_path: Path, monkeypatch
) -> None:
    hardneg = tmp_path / "hardneg.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    judgements = tmp_path / "judgements.jsonl"
    out_dir = tmp_path / "out"
    write_jsonl(
        manifest,
        [
            {
                "row_id": "row-1",
                "fact_index": 0,
                "claim": "甲导致乙",
                "cited_eids": ["E1"],
                "cited_texts": {"E1": "前文" * 400 + "甲导致乙"},
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
                "chunk_chars": 600,
                "merged": {"verdict": "supported"},
                "individual": {"E1": {"verdict": "supported"}},
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
                "hard_negatives": [
                    {
                        "eid": "E9",
                        "text": "无关证据",
                        "glm_verdict": "unsupported",
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
            "--out-dir",
            str(out_dir),
        ],
    )

    assert MODULE.main() == 0
    assert MODULE.read_jsonl(out_dir / "pairwise.jsonl") == []
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["stats"]["dropped_gold:judge_input_truncated"] == 1


def test_full_adjudication_requires_matching_evidence_fingerprint() -> None:
    task = {
        "claim": "甲导致乙",
        "cited_eids": ["E1"],
        "cited_texts": {"E1": "完整证据"},
    }
    valid = {
        "status": "ok",
        "full_evidence": True,
        "evidence_sha256": MODULE.evidence_fingerprint(task),
        "label": "supported_by_union",
        "keep_eids": ["E1"],
    }

    assert MODULE.full_adjudication_matches(task, valid)
    assert not MODULE.full_adjudication_matches(
        task, {**valid, "evidence_sha256": "stale"}
    )
    assert not MODULE.full_adjudication_matches(
        task, {**valid, "label": "invalid"}
    )
    assert not MODULE.full_adjudication_matches(
        task,
        {
            **valid,
            "label": "supported_by_union",
            "keep_eids": [],
        },
    )
    assert not MODULE.full_adjudication_matches(
        task,
        {
            **valid,
            "label": "supported_by_union",
            "keep_eids": ["E9"],
        },
    )


def test_legacy_2000_character_negative_is_not_training_safe() -> None:
    candidate = {
        "eid": "E9",
        "text": "负" * 2000,
        "glm_verdict": "unsupported",
    }
    assert (
        MODULE.negative_candidate_coverage_issue(
            "row-1", 0, "甲导致乙", candidate
        )
        == "judge_input_truncated"
    )
    trusted = {
        **candidate,
        "full_evidence": True,
    }
    trusted["evidence_sha256"] = MODULE.negative_evidence_fingerprint(
        "row-1", 0, "甲导致乙", trusted
    )
    assert (
        MODULE.negative_candidate_coverage_issue(
            "row-1", 0, "甲导致乙", trusted
        )
        is None
    )


def test_non_ok_gold_judgement_is_not_training_safe() -> None:
    judgement = {
        "cited_eids": ["E1"],
        "merged": {"status": "error", "verdict": "supported"},
        "individual": {"E1": {"status": "ok", "verdict": "supported"}},
    }
    assert MODULE.judgement_status_issues(judgement) == [
        "merged_status_not_ok"
    ]


def test_full_evidence_rejudgement_replaces_truncated_negative(
    tmp_path: Path, monkeypatch
) -> None:
    hardneg = tmp_path / "hardneg.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    judgements = tmp_path / "judgements.jsonl"
    sidecar = tmp_path / "full.jsonl"
    out_dir = tmp_path / "out"
    claim = "甲导致乙"
    original = "负" * 2000
    full = original + "完整尾部"
    write_jsonl(
        manifest,
        [
            {
                "row_id": "row-1",
                "fact_index": 0,
                "claim": claim,
                "cited_eids": ["E1"],
                "cited_texts": {"E1": "正证据"},
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
                "chunk_chars": 100,
                "merged": {"status": "ok", "verdict": "supported"},
                "individual": {"E1": {"status": "ok", "verdict": "supported"}},
            }
        ],
    )
    write_jsonl(
        hardneg,
        [
            {
                "record_id": "row-1",
                "fact_index": 0,
                "claim": claim,
                "hard_negatives": [
                    {"eid": "E9", "text": original, "glm_verdict": "unsupported"}
                ],
            }
        ],
    )
    write_jsonl(
        sidecar,
        [
            {
                "record_id": "row-1",
                "fact_index": 0,
                "claim": claim,
                "eid": "E9",
                "candidate_role": "hard_negative",
                "text": full,
                "evidence_chars": len(full),
                "evidence_sha256": MODULE.negative_evidence_fingerprint(
                    "row-1",
                    0,
                    claim,
                    {"eid": "E9", "text": full},
                ),
                "full_evidence": True,
                "status": "ok",
                "glm_verdict": "unsupported",
                "glm_reason": "完整证据不支持",
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
            "--full-evidence-rejudgements",
            str(sidecar),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert MODULE.main() == 0
    pairs = MODULE.read_jsonl(out_dir / "pairwise.jsonl")
    assert len(pairs) == 1
    assert pairs[0]["negative"] == full
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["stats"]["accepted_full_evidence_rejudgements"] == 1
    assert report["stats"]["full_evidence_rejudgement_verdict:unsupported"] == 1


def test_full_evidence_rejudgement_supported_hard_negative_is_excluded(
    tmp_path: Path, monkeypatch
) -> None:
    hardneg = tmp_path / "hardneg.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    judgements = tmp_path / "judgements.jsonl"
    sidecar = tmp_path / "full.jsonl"
    out_dir = tmp_path / "out"
    claim = "甲导致乙"
    original = "负" * 2000
    full = original + "支持尾部"
    write_jsonl(
        manifest,
        [
            {
                "row_id": "row-1",
                "fact_index": 0,
                "claim": claim,
                "cited_eids": ["E1"],
                "cited_texts": {"E1": "正证据"},
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
                "chunk_chars": 100,
                "merged": {"status": "ok", "verdict": "supported"},
                "individual": {"E1": {"status": "ok", "verdict": "supported"}},
            }
        ],
    )
    write_jsonl(
        hardneg,
        [
            {
                "record_id": "row-1",
                "fact_index": 0,
                "claim": claim,
                "hard_negatives": [
                    {"eid": "E9", "text": original, "glm_verdict": "unsupported"}
                ],
            }
        ],
    )
    write_jsonl(
        sidecar,
        [
            {
                "record_id": "row-1",
                "fact_index": 0,
                "claim": claim,
                "eid": "E9",
                "candidate_role": "hard_negative",
                "text": full,
                "evidence_chars": len(full),
                "evidence_sha256": MODULE.negative_evidence_fingerprint(
                    "row-1",
                    0,
                    claim,
                    {"eid": "E9", "text": full},
                ),
                "full_evidence": True,
                "status": "ok",
                "glm_verdict": "supported",
                "glm_reason": "完整证据支持",
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
            "--full-evidence-rejudgements",
            str(sidecar),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert MODULE.main() == 0
    assert MODULE.read_jsonl(out_dir / "pairwise.jsonl") == []
