#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from goldenglow.config import QueryConfig
from goldenglow.inference.cpu_pipeline import (
    CPUInferencePipeline,
    HypothesisDocument,
    build_hypothesis,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever


RUN_ROOT = Path(
    "${REMOTE_ASA_ROOT}/rank_v8_coverage/runs/"
    "query_answerability_bge_t16_capped6_20260821"
)
ANSWERS_PATH = RUN_ROOT / "eval/e2e_novel_questions/answers.jsonl"
RUNTIME_PATH = RUN_ROOT / "eval/e2e_novel_questions/runtime_t16_capped6_e2e_enabled.json"
OUTPUT_PATH = RUN_ROOT / "eval/e2e_novel_questions/retrieval_only_profile.json"


class DummyGenerator:
    max_tokens = 900


def load_cases() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in ANSWERS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases: list[dict[str, Any]] = []
    for row in rows:
        trace = row.get("retrieval_trace") or []
        first = trace[0] if trace else None
        if first and isinstance(first.get("hypothesis"), dict):
            hypothesis = HypothesisDocument(**first["hypothesis"])
            queries = list(first.get("queries") or [])
            source = "saved_round1_trace"
        else:
            hypothesis = build_hypothesis(row["question"])
            queries = [row["question"]]
            source = "heuristic_fallback"
        cases.append(
            {
                "question": row["question"],
                "hypothesis": hypothesis,
                "queries": queries,
                "source": source,
                "round2": (
                    {
                        "hypothesis": HypothesisDocument(**trace[1]["hypothesis"]),
                        "queries": list(trace[1].get("queries") or []),
                    }
                    if len(trace) >= 2 and isinstance(trace[1].get("hypothesis"), dict)
                    else None
                ),
            }
        )
    return cases


def main() -> None:
    runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
    retrieval_cfg = dict(runtime.get("retrieval") or {})
    valid_fields = {field.name for field in fields(QueryConfig)}
    query_config = QueryConfig(
        **{name: value for name, value in retrieval_cfg.items() if name in valid_fields}
    )

    reranker_path = Path(str(retrieval_cfg["reranker_model_path"]))
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/model/embeddings/bge-small-zh-v1.5"),
        reranker_model_path=reranker_path,
        reranker_max_length=int(retrieval_cfg.get("reranker_max_length", 1024)),
        minirag_index_path=Path(
            "${REMOTE_HOME}/ASA-ArknightStoryAgent/indexes/arknights_story_minirag_v3/graph.json"
        ),
        device="cuda",
    )
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=DummyGenerator(),
        query_config=query_config,
        max_retrieval_rounds=2,
        prompt_evidence_top_k=int((runtime.get("inference") or {}).get("prompt_evidence_top_k", 10)),
        use_model_hypothesis=True,
        use_model_conclusion_generation=True,
    )

    original_pipeline_methods = [
        "_search_queries",
        "_search_minirag_queries",
        "_search_scoped_chapter_queries",
        "_finalize_hits",
        "_expand_fused_hits_with_neighbors",
        "_retrieve_first_round_with_scoped_minirag_expansion",
        "_retrieve_round",
    ]
    original_retriever_methods = [
        "dense_search",
        "sparse_search",
        "minirag_search",
        "dense_search_chapter",
        "sparse_search_chapter",
        "_collect_story_and_stage_neighbors",
    ]
    events: list[dict[str, Any]] = []
    current_case = -1
    current_round = ""
    sequence = 0

    def record(name: str, elapsed: float, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        nonlocal sequence
        sequence += 1
        payload: dict[str, Any] = {
            "seq": sequence,
            "case": current_case,
            "round": current_round,
            "name": name,
            "elapsed_sec": round(elapsed, 6),
        }
        if name in {"_search_queries", "_search_minirag_queries", "_search_scoped_chapter_queries"}:
            queries = args[0] if args else []
            payload["query_count"] = len(queries) if isinstance(queries, list) else None
            payload["enable_minirag"] = kwargs.get("enable_minirag")
            if name == "_search_scoped_chapter_queries":
                payload["chapter_scope"] = kwargs.get("chapter_scope")
        if name in {"dense_search", "sparse_search", "minirag_search", "dense_search_chapter", "sparse_search_chapter"}:
            payload["top_k"] = kwargs.get("top_k")
            if args and isinstance(args[0], str):
                payload["query_chars"] = len(args[0])
            if name.endswith("_chapter"):
                payload["chapter_scope"] = kwargs.get("chapter_scope")
        if name == "_finalize_hits" and len(args) >= 5:
            payload["dense_count"] = len(args[2])
            payload["sparse_count"] = len(args[3])
            payload["minirag_count"] = len(args[4])
        if name == "_expand_fused_hits_with_neighbors" and args:
            payload["input_count"] = len(args[0])
        if name == "_collect_story_and_stage_neighbors" and args:
            payload["input_count"] = len(args[0])
        if name == "score":
            documents = kwargs.get("documents")
            if documents is None and len(args) >= 2:
                documents = args[1]
            payload["document_count"] = len(documents) if isinstance(documents, list) else None
        events.append(payload)

    def install_wrapper(obj: Any, name: str) -> None:
        original = getattr(obj, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                record(name, time.perf_counter() - started, args, kwargs)

        setattr(obj, name, wrapped)

    for name in original_pipeline_methods:
        install_wrapper(pipeline, name)
    for name in original_retriever_methods:
        if hasattr(retriever, name):
            install_wrapper(retriever, name)
    if retriever.reranker is not None:
        install_wrapper(retriever.reranker, "score")

    case_results: list[dict[str, Any]] = []
    cases = load_cases()
    for case_index, case in enumerate(cases):
        current_case = case_index
        current_round = "round1_expansion"
        before = len(events)
        started = time.perf_counter()
        _, _, evidence, expansion = pipeline._retrieve_first_round_with_scoped_minirag_expansion(
            case["question"], case["hypothesis"], case["queries"]
        )
        round1_elapsed = time.perf_counter() - started
        round1_events = events[before:]
        round2_result: dict[str, Any] | None = None
        if case["round2"] is not None:
            current_round = "round2_global"
            before = len(events)
            started = time.perf_counter()
            _, _, round2_evidence = pipeline._retrieve_round(
                case["question"], case["round2"]["hypothesis"], case["round2"]["queries"]
            )
            round2_elapsed = time.perf_counter() - started
            round2_result = {
                "elapsed_sec": round(round2_elapsed, 6),
                "evidence_count": len(round2_evidence),
                "events": events[before:],
            }
        current_round = ""
        case_results.append(
            {
                "case_index": case_index,
                "question": case["question"],
                "source": case["source"],
                "round1_elapsed_sec": round(round1_elapsed, 6),
                "round1_evidence_count": len(evidence),
                "round1_expansion": expansion,
                "round1_events": round1_events,
                "round2": round2_result,
            }
        )
        print(
            json.dumps(
                {
                    "case": case_index + 1,
                    "round1_sec": round(round1_elapsed, 2),
                    "round2_sec": round(round2_result["elapsed_sec"], 2) if round2_result else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def sum_events(name: str, round_name: str | None = None) -> float:
        return sum(
            event["elapsed_sec"]
            for event in events
            if event["name"] == name and (round_name is None or event["round"] == round_name)
        )

    def values(name: str, round_name: str | None = None) -> list[float]:
        return [
            event["elapsed_sec"]
            for event in events
            if event["name"] == name and (round_name is None or event["round"] == round_name)
        ]

    method_summary: dict[str, Any] = {}
    for name in sorted({event["name"] for event in events}):
        vals = values(name)
        method_summary[name] = {
            "calls": len(vals),
            "total_sec": round(sum(vals), 6),
            "mean_sec": round(statistics.mean(vals), 6) if vals else 0.0,
            "p50_sec": round(statistics.median(vals), 6) if vals else 0.0,
            "max_sec": round(max(vals), 6) if vals else 0.0,
        }

    output = {
        "metadata": {
            "runtime_config": str(RUNTIME_PATH),
            "reranker": str(reranker_path),
            "device": "cuda:1 via CUDA_VISIBLE_DEVICES=1",
            "query_config": {
                field.name: getattr(query_config, field.name)
                for field in fields(QueryConfig)
            },
            "case_count": len(case_results),
            "round2_case_count": sum(1 for case in case_results if case["round2"] is not None),
        },
        "method_summary": method_summary,
        "events": events,
        "cases": case_results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "events": len(events)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
