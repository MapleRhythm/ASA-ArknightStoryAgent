#!/usr/bin/env python3
"""Compare first-round retrieval latency variants using saved ASA traces.

This helper is analysis-only: it does not modify project source or the runtime
configuration. Each variant reuses the saved question, hypothesis and initial
query set, then records latency and evidence-ID overlap with the full path.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

from goldenglow.config import QueryConfig
from goldenglow.inference import cpu_pipeline as pipeline_module
from goldenglow.inference.cpu_pipeline import CPUInferencePipeline, HypothesisDocument
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever


RUN_ROOT = Path(
    "${REMOTE_ASA_ROOT}/rank_v8_coverage/runs/"
    "query_answerability_bge_t16_capped6_20260821"
)
ANSWERS_PATH = RUN_ROOT / "eval/e2e_novel_questions/answers.jsonl"
RUNTIME_PATH = RUN_ROOT / "eval/e2e_novel_questions/runtime_t16_capped6_e2e_enabled.json"
DEFAULT_OUTPUT = RUN_ROOT / "eval/e2e_novel_questions/retrieval_latency_ablation.json"


class DummyGenerator:
    max_tokens = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on saved-trace cases.")
    return parser.parse_args()


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row_index, raw in enumerate(ANSWERS_PATH.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        row = json.loads(raw)
        trace = row.get("retrieval_trace") or []
        first = trace[0] if trace else None
        if not isinstance(first, dict) or not isinstance(first.get("hypothesis"), dict):
            continue
        cases.append(
            {
                "source_row_index": row_index,
                "question": str(row["question"]),
                "hypothesis": HypothesisDocument(**first["hypothesis"]),
                "queries": list(first.get("queries") or []),
            }
        )
    return cases


def build_query_config(runtime: dict[str, Any], **overrides: Any) -> QueryConfig:
    retrieval = dict(runtime.get("retrieval") or {})
    retrieval.update(overrides)
    valid_fields = {field.name for field in fields(QueryConfig)}
    return QueryConfig(**{name: value for name, value in retrieval.items() if name in valid_fields})


def evidence_ids(evidence: list[dict[str, Any]]) -> list[str]:
    return [str(item["document"].get("id") or "") for item in evidence if item.get("document")]


def latency_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean_sec": 0.0, "median_sec": 0.0, "p95_sec": 0.0, "min_sec": 0.0, "max_sec": 0.0}
    return {
        "count": len(ordered),
        "mean_sec": round(statistics.fmean(ordered), 6),
        "median_sec": round(statistics.median(ordered), 6),
        "p95_sec": round(ordered[int((len(ordered) - 1) * 0.95)], 6),
        "min_sec": round(ordered[0], 6),
        "max_sec": round(ordered[-1], 6),
    }


def rank_overlap(baseline: list[str], candidate: list[str], cutoff: int) -> dict[str, float | int | bool]:
    baseline_top = [item for item in baseline[:cutoff] if item]
    candidate_top = [item for item in candidate[:cutoff] if item]
    baseline_set = set(baseline_top)
    candidate_set = set(candidate_top)
    union = baseline_set | candidate_set
    return {
        "exact_order": baseline_top == candidate_top,
        "intersection": len(baseline_set & candidate_set),
        "jaccard": round(len(baseline_set & candidate_set) / len(union), 6) if union else 1.0,
    }


def install_compact_graph_query_builder() -> Callable[..., list[str]]:
    original = pipeline_module.build_minirag_expansion_queries

    def compact(*args: Any, **kwargs: Any) -> list[str]:
        # The full-evidence query can reach ~1.5k characters and dominates
        # BM25 time. Retain only the compact metadata/query variant.
        return [query for query in original(*args, **kwargs) if len(query) <= 420]

    pipeline_module.build_minirag_expansion_queries = compact
    return original


def retrieve_first_round(
    pipeline: CPUInferencePipeline,
    case: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if pipeline.query_config.minirag_chapter_isolation and pipeline.query_config.minirag_auto_second_retrieval:
        _, _, evidence, expansion = pipeline._retrieve_first_round_with_scoped_minirag_expansion(
            case["question"], case["hypothesis"], case["queries"]
        )
        return evidence, expansion
    _, _, evidence = pipeline._retrieve_round(
        case["question"], case["hypothesis"], case["queries"]
    )
    return evidence, None


def run_variant(
    *,
    name: str,
    retriever: ArknightsHybridRetriever,
    runtime: dict[str, Any],
    cases: list[dict[str, Any]],
    config_overrides: dict[str, Any],
    compact_graph_query: bool,
) -> dict[str, Any]:
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=DummyGenerator(),
        query_config=build_query_config(runtime, **config_overrides),
        max_retrieval_rounds=2,
        prompt_evidence_top_k=int((runtime.get("inference") or {}).get("prompt_evidence_top_k", 10)),
        use_model_hypothesis=True,
        use_model_conclusion_generation=True,
    )
    original_builder = install_compact_graph_query_builder() if compact_graph_query else None
    records: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            evidence, expansion = retrieve_first_round(pipeline, case)
            elapsed = time.perf_counter() - started
            ids = evidence_ids(evidence)
            records.append(
                {
                    "source_row_index": case["source_row_index"],
                    "question": case["question"],
                    "elapsed_sec": round(elapsed, 6),
                    "evidence_ids": ids,
                    "top3_ids": ids[:3],
                    "expansion": expansion,
                }
            )
            print(
                json.dumps(
                    {"variant": name, "case": index, "total": len(cases), "elapsed_sec": round(elapsed, 3)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if original_builder is not None:
            pipeline_module.build_minirag_expansion_queries = original_builder

    return {
        "name": name,
        "config_overrides": config_overrides,
        "compact_graph_query": compact_graph_query,
        "latency": latency_summary([float(record["elapsed_sec"]) for record in records]),
        "records": records,
    }


def main() -> int:
    args = parse_args()
    runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
    cases = load_cases()
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No saved-trace cases with first-round retrieval trace found.")

    retrieval = dict(runtime.get("retrieval") or {})
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/model/embeddings/bge-small-zh-v1.5"),
        reranker_model_path=Path(str(retrieval["reranker_model_path"])),
        reranker_max_length=int(retrieval.get("reranker_max_length", 1024)),
        minirag_index_path=Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/indexes/arknights_story_minirag_v3/graph.json"),
        device="cuda",
    )
    variants = [
        run_variant(
            name="full_current",
            retriever=retriever,
            runtime=runtime,
            cases=cases,
            config_overrides={},
            compact_graph_query=False,
        ),
        run_variant(
            name="compact_graph_query",
            retriever=retriever,
            runtime=runtime,
            cases=cases,
            config_overrides={},
            compact_graph_query=True,
        ),
        run_variant(
            name="compact_graph_query_no_scoped_search",
            retriever=retriever,
            runtime=runtime,
            cases=cases,
            config_overrides={"enable_scoped_chapter_search": False},
            compact_graph_query=True,
        ),
        run_variant(
            name="no_chapter_expansion",
            retriever=retriever,
            runtime=runtime,
            cases=cases,
            config_overrides={"minirag_auto_second_retrieval": False, "enable_scoped_chapter_search": False},
            compact_graph_query=False,
        ),
    ]

    baseline_by_row = {
        int(record["source_row_index"]): record
        for variant in variants
        if variant["name"] == "full_current"
        for record in variant["records"]
    }
    for variant in variants:
        for record in variant["records"]:
            baseline = baseline_by_row[int(record["source_row_index"])]
            record["baseline_overlap"] = {
                f"top{cutoff}": rank_overlap(baseline["evidence_ids"], record["evidence_ids"], cutoff)
                for cutoff in (3, 10, 32)
            }

    concise: dict[str, Any] = {}
    for variant in variants:
        concise[variant["name"]] = {
            "latency": variant["latency"],
            "top3_mean_jaccard": round(
                statistics.fmean(record["baseline_overlap"]["top3"]["jaccard"] for record in variant["records"]),
                6,
            ),
            "top10_mean_jaccard": round(
                statistics.fmean(record["baseline_overlap"]["top10"]["jaccard"] for record in variant["records"]),
                6,
            ),
            "top3_exact_match_rate": round(
                statistics.fmean(float(record["baseline_overlap"]["top3"]["exact_order"]) for record in variant["records"]),
                6,
            ),
        }
    report = {
        "description": "Saved-trace first-round latency ablation. Runtime source was not modified.",
        "runtime": str(RUNTIME_PATH),
        "case_count": len(cases),
        "variants": variants,
        "summary": concise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": concise}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
