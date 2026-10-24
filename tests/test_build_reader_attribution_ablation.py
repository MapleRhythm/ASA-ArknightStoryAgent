import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_reader_attribution_ablation.py"
SPEC = importlib.util.spec_from_file_location("attribution_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs():
    raw = "甲：不是乙的错。\n请别责怪她。\n丙：我明白了。"
    flat = " ".join(raw.splitlines())
    matrix = [{
        "id": "q::minimal_sufficient",
        "system": "system",
        "conversations": [
            {"from": "human", "value": f"question: 谁为乙辩护？\nevidence:\n[E1]\n{flat}\noutput_schema: test\nrules: test"},
            {"from": "gpt", "value": "PRIVATE_REFERENCE"},
        ],
        "diagnostic": {
            "source_id": "q", "condition": "minimal_sufficient",
            "question_family": "family", "evidence_id_to_corpus_id": {"E1": "doc"},
            "notes": "PRIVATE_NOTES",
        },
    }]
    hit = {"id": "doc", "clean_text": raw, "evidence_chain_text": "MUST_NOT_USE_CHAIN"}
    snapshots = [{"id": "q", "stages": {"prompt": [hit], "rerank": [hit]}}]
    return matrix, snapshots


def test_content_identity_no_gold_leakage_or_inferred_speaker():
    matrix, snapshots = inputs()
    rows, report = MODULE.build_rows(matrix, snapshots)
    assert len(rows) == 3
    assert rows[0]["conversations"][0] == matrix[0]["conversations"][0]
    assert "甲：不是乙的错。\n请别责怪她。" in rows[1]["conversations"][0]["value"]
    assert len({r["diagnostic"]["evidence_nonwhitespace_sha256"] for r in rows}) == 1
    for row in rows:
        prompt = row["conversations"][0]["value"]
        assert "PRIVATE" not in prompt and "MUST_NOT_USE_CHAIN" not in prompt
        assert row["diagnostic"]["evaluation_only"] is True
    assert "reading_guidance:" not in rows[1]["conversations"][0]["value"]
    assert "reading_guidance:" in rows[2]["conversations"][0]["value"]
    assert report["baseline_prompt_identical"]


def test_source_mismatch_and_duplicate_matrix_fail_closed():
    matrix, snapshots = inputs()
    with pytest.raises(ValueError, match="duplicate matrix"):
        MODULE.build_rows(matrix * 2, snapshots)
    snapshots[0]["stages"]["rerank"][0]["clean_text"] = "被修改的正文"
    with pytest.raises(ValueError, match="differs from source"):
        MODULE.build_rows(matrix, snapshots)
