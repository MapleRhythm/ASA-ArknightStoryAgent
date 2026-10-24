from __future__ import annotations

from goldenglow.config import QueryConfig
from goldenglow.inference.cpu_pipeline import CPUInferencePipeline


class _Generator:
    max_tokens = 512


def _pipeline(cap: int) -> CPUInferencePipeline:
    return CPUInferencePipeline(
        retriever=object(),
        generator=_Generator(),
        query_config=QueryConfig(max_retrieval_queries=cap),
    )


def test_retrieval_query_cap_deduplicates_and_caps() -> None:
    pipeline = _pipeline(2)
    assert pipeline.limit_retrieval_queries(["q1", "q1", "q2", "q3"]) == ["q1", "q2"]


def test_zero_retrieval_query_cap_keeps_all_unique_queries() -> None:
    pipeline = _pipeline(0)
    assert pipeline.limit_retrieval_queries(["q1", "q1", "q2", "q3"]) == ["q1", "q2", "q3"]
