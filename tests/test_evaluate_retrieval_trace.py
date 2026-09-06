import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval_trace.py"
SPEC = importlib.util.spec_from_file_location("evaluate_retrieval_trace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stage_recall_and_prompt_loss_are_reported() -> None:
    rows = [
        {
            "id": "q1",
            "question": "问题一",
            "question_family": "family-a",
            "gold_evidence_ids": ["a", "b"],
            "stages": {
                "dense": ["a", "x"],
                "sparse": ["y", "b"],
                "fusion": ["a", "y", "b"],
                "rerank": ["a", "b"],
                "prompt": ["a"],
            },
        }
    ]

    result = MODULE.evaluate_rows(rows, [1, 2], 5)

    assert result["rows_evaluable"] == 1
    assert result["stages"]["fusion"]["recall"]["@2"] == 0.5
    assert result["stages"]["rerank"]["all_gold_rate"]["@2"] == 1.0
    assert result["stages"]["prompt"]["all_gold_rate"]["@2"] == 0.0
    assert result["family_macro"]["family-a"]["prompt"]["recall"]["@1"] == 0.5


def test_round_metrics_measure_novelty_and_duplicate_queries() -> None:
    row = {
        "id": "q1",
        "question": "问题一",
        "gold_evidence_ids": ["a", "b"],
        "stages": {
            "dense": ["a"],
            "sparse": ["a"],
            "fusion": ["a"],
            "rerank": ["a"],
            "prompt": ["a"],
        },
        "rounds": [
            {"round": 1, "query": "原问题", "doc_ids": ["a", "x"]},
            {"round": 2, "query": "原问题", "doc_ids": ["a", "b"]},
        ],
    }

    result = MODULE.evaluate_rows([row], [1, 2], 2)
    transition = result["round_rows"][0]["transitions"][0]

    assert transition["top_k_jaccard"] == 1 / 3
    assert transition["fresh_doc_rate"] == 0.5
    assert transition["fresh_recall"] == 0.5
    assert transition["query_exact_duplicate"] == 1


def test_object_stage_items_and_json_wrapper_are_supported(tmp_path: Path) -> None:
    path = tmp_path / "snapshots.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": "q1",
                        "gold": {"evidence": [{"id": "a"}]},
                        "retrieval_snapshot": {
                            "dense_hits": [{"doc_id": "a"}],
                            "sparse_hits": [{"doc_id": "a"}],
                            "fusion_hits": [{"doc_id": "a"}],
                            "rerank_hits": [{"doc_id": "a"}],
                            "prompt_evidence": [{"doc_id": "a"}],
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = MODULE.read_rows(path)
    result = MODULE.evaluate_rows(rows, [1], 5)
    assert result["rows_evaluable"] == 1
    assert result["stages"]["prompt"]["all_gold_rate"]["@1"] == 1.0


def test_missing_stage_is_reported_without_inventing_data() -> None:
    row = {
        "id": "bad",
        "gold_evidence_ids": ["a"],
        "stages": {
            "dense": ["a"],
            "sparse": ["a"],
            "fusion": ["a"],
            "rerank": ["a"],
        },
    }
    with pytest.raises(MODULE.SnapshotError, match="no valid stage snapshots"):
        MODULE.evaluate_rows([row], [1], 5)
