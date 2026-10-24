from __future__ import annotations

import numpy as np

from goldenglow.retrieval.hybrid import (
    ArknightsHybridRetriever,
    tokenize_char_bigrams,
    tokenize_exact_terms,
)


def test_char_bigrams_do_not_cross_punctuation_or_lines() -> None:
    tokens = tokenize_char_bigrams("格拉尼。\n可萝尔")
    assert "格拉" in tokens
    assert "萝尔" in tokens
    assert "尼可" not in tokens


def test_stage_code_keeps_exact_and_compact_forms() -> None:
    tokens = tokenize_char_bigrams("GT-1 / main_12")
    assert "gt-1" in tokens
    assert "gt1" in tokens
    assert "main_12" in tokens
    assert "main12" in tokens
    assert "gt-1" in tokenize_exact_terms("GT-1")


def test_source_quotas_preserve_sparse_only_candidates() -> None:
    ranked = [
        {"doc_index": index, "fusion_score": 100.0 - index}
        for index in range(8)
    ]
    dense_hits = [{"doc_index": index} for index in range(4)]
    sparse_hits = [{"doc_index": index} for index in (7, 6, 5, 4)]
    selected = ArknightsHybridRetriever.apply_source_quotas(
        ranked,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        top_k=4,
        dense_min_quota=2,
        sparse_min_quota=2,
    )
    assert [item["doc_index"] for item in selected] == [0, 1, 6, 7]
    assert len(selected) == 4


def test_overlapping_source_quotas_are_rank_cutoffs() -> None:
    ranked = [
        {"doc_index": index, "fusion_score": 100.0 - index}
        for index in range(8)
    ]
    dense_hits = [{"doc_index": index} for index in (0, 1, 2, 3)]
    sparse_hits = [{"doc_index": index} for index in (0, 4, 5, 6)]
    selected = ArknightsHybridRetriever.apply_source_quotas(
        ranked,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        top_k=5,
        dense_min_quota=2,
        sparse_min_quota=3,
    )
    assert [item["doc_index"] for item in selected] == [0, 1, 2, 4, 5]


def test_dense_batch_results_preserve_query_rows() -> None:
    class FakeIndex:
        def search(self, vectors: np.ndarray, top_k: int):
            assert vectors.shape == (2, 3)
            assert top_k == 2
            return (
                np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
                np.array([[0, 1], [2, 1]], dtype=np.int64),
            )

    retriever = object.__new__(ArknightsHybridRetriever)
    retriever.index = FakeIndex()
    retriever.documents = [{"id": str(index)} for index in range(3)]
    rows = retriever._dense_hits_from_vectors(
        np.zeros((2, 3), dtype=np.float32),
        top_k=2,
    )
    assert [[hit["doc_index"] for hit in row] for row in rows] == [[0, 1], [2, 1]]
