import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_reader_evidence_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_reader_evidence_diagnostic", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _hit(doc_id: str, text: str) -> dict:
    return {
        "id": doc_id,
        "clean_text": text,
        "search_text": text,
        "activity_name": "活动",
        "story_name": "剧情",
        "stage_code": "ST-1",
        "source_path": f"stories/{doc_id}.txt",
    }


def _snapshot(
    row_id: str = "q1",
    family: str = "family-1",
) -> dict:
    doc_a = _hit("doc-a", "甲决定行动。")
    doc_b = _hit("doc-b", "甲行动是为了保护乙。")
    doc_c = _hit("doc-c", "同一场景中，丙谈到另一件事。")
    return {
        "id": row_id,
        "question": "甲为什么行动？",
        "question_family": family,
        "hypothesis": {
            "question": "甲为什么行动？",
            "query_type": "causality",
            "entities": ["甲"],
            "keywords": ["行动", "原因"],
            "expected_answer_type": "原因",
            "dialogue_context": "",
        },
        "stages": {
            "rerank": [doc_a, doc_b, doc_c],
            "prompt": [doc_a, doc_c],
            "fusion": [doc_a, doc_b, doc_c],
            "dense": [],
            "sparse": [],
            "minirag": [],
        },
    }


def _annotation(
    row_id: str = "q1",
    family: str = "family-1",
) -> dict:
    return {
        "id": row_id,
        "question_family": family,
        "review_status": "approved",
        "reviewer": "human-reviewer",
        "minimal_sufficient_evidence_ids": ["doc-a", "doc-b"],
        "distractor_evidence_ids": ["doc-c"],
        "answer_facts": [
            {
                "fact": "甲行动是为了保护乙。",
                "support_groups": [["doc-a", "doc-b"]],
            }
        ],
        "notes": "人工核对。",
    }


def _payload(row: dict) -> dict:
    return json.loads(row["conversations"][-1]["value"])


def test_build_matrix_creates_three_reviewed_conditions() -> None:
    rows, audit = MODULE.build_matrix(
        snapshots=[_snapshot()],
        annotations=[_annotation()],
        seed=20260907,
        max_chars_per_doc=1000,
        max_total_chars=8000,
        round_value="1/2",
    )

    by_condition = {row["diagnostic"]["condition"]: row for row in rows}
    assert set(by_condition) == set(MODULE.CONDITIONS)
    assert audit["matrix_rows"] == 3
    assert audit["question_families"] == 1

    minimal = by_condition["minimal_sufficient"]
    minimal_payload = _payload(minimal)
    assert minimal_payload["next_action"] == "answer_directly"
    cited = minimal_payload["supported_facts"][0]["evidence_ids"]
    mapped = minimal["diagnostic"]["evidence_id_to_corpus_id"]
    assert {mapped[evidence_id] for evidence_id in cited} == {"doc-a", "doc-b"}

    distracted = by_condition["sufficient_with_distractors"]
    assert _payload(distracted)["next_action"] == "answer_directly"
    assert set(distracted["diagnostic"]["visible_corpus_ids"]) == {
        "doc-a",
        "doc-b",
        "doc-c",
    }

    actual = by_condition["actual_prompt"]
    assert _payload(actual)["next_action"] == "retrieve_more"
    assert actual["diagnostic"]["reference_sufficiency"] == "insufficient"
    assert actual["diagnostic"]["visible_corpus_ids"] == ["doc-a", "doc-c"]
    assert "output_schema: grounded_action_exx_v1" in actual["conversations"][0]["value"]


def test_reviewed_distractor_cannot_be_an_accepted_support_document() -> None:
    annotation = _annotation()
    annotation["answer_facts"][0]["support_groups"].append(["doc-c"])

    with pytest.raises(
        MODULE.DiagnosticBuildError,
        match="distractors overlap an accepted support group",
    ):
        MODULE.build_matrix(
            snapshots=[_snapshot()],
            annotations=[annotation],
            seed=1,
            max_chars_per_doc=1000,
            max_total_chars=8000,
            round_value="1/2",
        )


def test_unapproved_or_duplicate_family_annotations_are_rejected() -> None:
    annotation = _annotation()
    annotation["review_status"] = "draft"
    with pytest.raises(MODULE.DiagnosticBuildError, match="must be approved"):
        MODULE.build_matrix(
            snapshots=[_snapshot()],
            annotations=[annotation],
            seed=1,
            max_chars_per_doc=1000,
            max_total_chars=8000,
            round_value="1/2",
        )

    with pytest.raises(MODULE.DiagnosticBuildError, match="duplicate reviewed"):
        MODULE.build_matrix(
            snapshots=[
                _snapshot("q1", "same-family"),
                _snapshot("q2", "same-family"),
            ],
            annotations=[
                _annotation("q1", "same-family"),
                _annotation("q2", "same-family"),
            ],
            seed=1,
            max_chars_per_doc=1000,
            max_total_chars=8000,
            round_value="1/2",
        )
