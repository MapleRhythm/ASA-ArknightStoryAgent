import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from asa_arknight_story_agent.config import QueryConfig


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_retrieval_stage_snapshots.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_retrieval_stage_snapshots", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _hit(doc_id: str, score: float) -> dict:
    return {
        "doc_index": int(score * 10),
        "fusion_score": score,
        "document": {
            "id": doc_id,
            "clean_text": f"{doc_id} 正文",
            "search_text": f"{doc_id} 正文",
        },
    }


def test_query_config_uses_runtime_values_without_changing_defaults() -> None:
    config = MODULE.query_config_from_runtime(
        {"dense_top_k": 120, "enable_neighbor_expansion": True}
    )
    assert config.dense_top_k == 120
    assert config.enable_neighbor_expansion
    assert config.rerank_top_k == QueryConfig().rerank_top_k


def test_manifest_hypothesis_and_gold_are_strictly_validated() -> None:
    row = {
        "question": "甲为什么行动？",
        "gold_evidence_ids": ["doc-a", "doc-a", "doc-b"],
        "hypothesis": {
            "query_type": "causality",
            "entities": ["甲"],
            "keywords": ["行动"],
            "expected_answer_type": "原因",
        },
    }
    hypothesis = MODULE.hypothesis_from_row(row)
    assert hypothesis.query_type == "causality"
    assert MODULE.gold_ids_from_row(row) == ["doc-a", "doc-b"]

    with pytest.raises(MODULE.SnapshotBuildError, match="unknown hypothesis"):
        MODULE.hypothesis_from_row(
            {
                "question": "问题",
                "hypothesis": {"invented": True},
            }
        )


def test_build_snapshot_records_all_required_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    dense = [_hit("dense-a", 0.8)]
    sparse = [_hit("sparse-b", 0.7)]
    fused = [_hit("gold-a", 0.9), _hit("gold-b", 0.8)]
    reranked = [_hit("gold-b", 1.0), _hit("gold-a", 0.9)]
    prompt = [reranked[0]]

    monkeypatch.setattr(
        MODULE,
        "search_queries",
        lambda **kwargs: (dense, sparse, []),
    )
    monkeypatch.setattr(
        MODULE,
        "_finalize_stage",
        lambda **kwargs: (fused, reranked),
    )
    monkeypatch.setattr(
        MODULE,
        "prepare_prompt_evidence",
        lambda **kwargs: prompt,
    )

    pipeline = MODULE.build_pipeline(
        retriever=object(),
        query_config=QueryConfig(
            minirag_chapter_isolation=False,
            minirag_auto_second_retrieval=False,
        ),
        inference_cfg={"prompt_evidence_top_k": 1},
    )
    result = MODULE.build_snapshot_row(
        pipeline=pipeline,
        row={
            "id": "q1",
            "question": "问题",
            "question_family": "family-a",
            "gold_evidence_ids": ["gold-a", "gold-b"],
            "queries": ["问题"],
        },
        row_index=0,
    )

    assert result["snapshot_version"] == MODULE.SNAPSHOT_VERSION
    assert [item["id"] for item in result["stages"]["dense"]] == ["dense-a"]
    assert [item["id"] for item in result["stages"]["sparse"]] == ["sparse-b"]
    assert [item["id"] for item in result["stages"]["fusion"]] == [
        "gold-a",
        "gold-b",
    ]
    assert [item["id"] for item in result["stages"]["rerank"]] == [
        "gold-b",
        "gold-a",
    ]
    assert [item["id"] for item in result["stages"]["prompt"]] == ["gold-b"]


def test_scoped_minirag_second_pass_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_dense = [_hit("dense-first", 0.8)]
    first_sparse = [_hit("sparse-first", 0.7)]
    first_fused = [_hit("fused-first", 0.9)]
    first_evidence = [_hit("evidence-first", 1.0)]
    graph_hits = [_hit("graph-hit", 0.6)]
    final_dense = [_hit("dense-final", 0.95)]
    final_sparse = [_hit("sparse-final", 0.85)]
    final_minirag = [_hit("minirag-final", 0.75)]
    final_fused = [_hit("fused-final", 1.1)]
    final_reranked = [_hit("reranked-final", 1.2)]
    finalize_calls: list[tuple[list[dict], list[dict], list[dict]]] = []

    monkeypatch.setattr(
        MODULE,
        "search_queries",
        lambda **kwargs: (first_dense, first_sparse, []),
    )

    def fake_finalize_stage(**kwargs):
        call = (
            kwargs["dense_hits"],
            kwargs["sparse_hits"],
            kwargs["minirag_hits"],
        )
        finalize_calls.append(call)
        if len(finalize_calls) == 1:
            return first_fused, first_evidence
        return final_fused, final_reranked

    monkeypatch.setattr(MODULE, "_finalize_stage", fake_finalize_stage)
    monkeypatch.setattr(
        MODULE,
        "build_minirag_scope_context",
        lambda **kwargs: SimpleNamespace(
            scope_info={"label": "主线"},
            sparse_storyline_scope="storyline",
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "minirag_scope_values",
        lambda *args: ("chapter", 1.5, True, True, "graph-scope"),
    )
    monkeypatch.setattr(
        MODULE,
        "run_minirag_second_pass",
        lambda **kwargs: SimpleNamespace(
            combined_dense_hits=final_dense,
            combined_sparse_hits=final_sparse,
            combined_minirag_hits=final_minirag,
            second_pass_queries=["补充查询"],
            use_scoped_candidates=True,
        ),
    )

    pipeline = MODULE.build_pipeline(
        retriever=object(),
        query_config=QueryConfig(
            minirag_chapter_isolation=True,
            minirag_auto_second_retrieval=True,
        ),
        inference_cfg={},
        enable_minirag=True,
    )
    pipeline._search_minirag_queries = lambda *args, **kwargs: graph_hits

    stages = MODULE._first_round_stages(
        pipeline=pipeline,
        question="问题",
        hypothesis=MODULE.hypothesis_from_row({"question": "问题"}),
        queries=["问题"],
    )

    assert stages[:5] == (
        final_dense,
        final_sparse,
        final_minirag,
        final_fused,
        final_reranked,
    )
    assert stages[5]["enabled"]
    assert stages[5]["second_pass_queries"] == ["补充查询"]
    assert finalize_calls == [
        (first_dense, first_sparse, []),
        (final_dense, final_sparse, final_minirag),
    ]


def test_snapshot_builder_rejects_enabled_web_context() -> None:
    with pytest.raises(MODULE.SnapshotBuildError, match="web context"):
        MODULE.build_pipeline(
            retriever=object(),
            query_config=QueryConfig(),
            inference_cfg={"web_context": {"enabled": True}},
        )
