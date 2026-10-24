from __future__ import annotations

from goldenglow.config import QueryConfig
from goldenglow.inference.cpu_pipeline import (
    CPUInferencePipeline,
    HypothesisDocument,
    _build_compound_subclaim_queries,
)


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


def test_compound_question_reserves_query_lane_for_each_subclaim() -> None:
    pipeline = _pipeline(4)
    question = "苏茜为什么买下绿意火花，后来又为什么把它改成理发店？"
    queries = [
        question,
        "苏茜 绿意火花",
        "苏茜",
        "绿意火花",
        "苏茜 买下绿意火花 原因",
        "苏茜 改成理发店 原因",
    ]

    selected = pipeline.limit_retrieval_queries(queries, question=question)

    assert selected[0] == question
    assert "苏茜 买下绿意火花 原因" in selected
    assert "苏茜 改成理发店 原因" in selected
    assert len(selected) == 4


def test_single_question_keeps_historical_query_order() -> None:
    pipeline = _pipeline(3)
    question = "凯尔希为什么阻止Mon3tr攻击？"
    queries = [question, "凯尔希 Mon3tr", "攻击 原因", "凯尔希"]

    assert pipeline.limit_retrieval_queries(queries, question=question) == queries[:3]


def test_compound_query_builder_keeps_pronoun_but_adds_primary_anchor() -> None:
    question = "苏茜为什么买下绿意火花，后来又为什么把它改成理发店？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["苏茜", "绿意火花"],
        keywords=["买下", "理发店"],
        expected_answer_type="原因与动机",
        dialogue_context="",
    )

    assert _build_compound_subclaim_queries(question, hypothesis) == [
        "苏茜为什么买下绿意火花",
        "苏茜 后来又为什么把它改成理发店",
    ]
