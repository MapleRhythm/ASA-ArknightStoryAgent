from __future__ import annotations

from dataclasses import replace

from asa_arknight_story_agent.config import QueryConfig
from asa_arknight_story_agent.inference.retrieval.search import (
    search_queries,
    search_scoped_chapter_queries,
)


class _Retriever:
    def _hit(self, query: str, lane: str, rank: int = 0) -> dict:
        doc_index = abs(hash((query, lane))) % 10000 + 1
        return {
            "doc_index": doc_index,
            "document": {"id": f"{lane}:{query}"},
            "score": 1.0 - rank / 10,
        }

    def dense_search(self, query: str, *, top_k: int):
        return [self._hit(query, "dense")]

    def sparse_search(self, query: str, *, top_k: int, storyline_scope=None):
        return [self._hit(query, "sparse")]

    def minirag_search(self, query: str, *, top_k: int, chapter_scope=None):
        return [self._hit(query, "minirag")]

    def dense_search_chapter(self, query: str, *, top_k: int, chapter_scope: str):
        return [self._hit(query, "chapter_dense")]

    def sparse_search_chapter(self, query: str, *, top_k: int, chapter_scope: str):
        return [self._hit(query, "chapter_sparse")]


class _Pipeline:
    def __init__(self, workers: int):
        self.retriever = _Retriever()
        self.query_config = replace(
            QueryConfig(),
            retrieval_workers=workers,
        )


def _ids(items: list[dict]) -> list[str]:
    return [str(item["document"]["id"]) for item in items]


def test_parallel_search_preserves_legacy_merge_order() -> None:
    queries = ["阿米娅", "凯尔希", "罗德岛"]
    sequential = search_queries(
        pipeline=_Pipeline(1),
        queries=queries,
        minirag_chapter_scope="main_1",
        sparse_storyline_scope="故事A",
        enable_minirag=True,
    )
    parallel = search_queries(
        pipeline=_Pipeline(3),
        queries=queries,
        minirag_chapter_scope="main_1",
        sparse_storyline_scope="故事A",
        enable_minirag=True,
    )
    assert [_ids(lane) for lane in parallel] == [_ids(lane) for lane in sequential]


def test_parallel_scoped_search_preserves_lane_results() -> None:
    queries = ["阿米娅", "凯尔希"]
    sequential = search_scoped_chapter_queries(
        pipeline=_Pipeline(1),
        queries=queries,
        chapter_scope="main_1",
    )
    parallel = search_scoped_chapter_queries(
        pipeline=_Pipeline(2),
        queries=queries,
        chapter_scope="main_1",
    )
    assert [_ids(lane) for lane in parallel] == [_ids(lane) for lane in sequential]


def test_empty_query_list_does_not_touch_retriever() -> None:
    result = search_queries(
        pipeline=_Pipeline(4),
        queries=[],
        enable_minirag=True,
    )
    assert result == ([], [], [])
