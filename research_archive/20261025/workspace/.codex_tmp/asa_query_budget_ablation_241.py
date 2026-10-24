#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import statistics
import time
from pathlib import Path
from typing import Any

from goldenglow.config import QueryConfig
from goldenglow.inference.cpu_pipeline import (
    CPUInferencePipeline,
    HypothesisDocument,
    _resolve_referential_question,
    build_follow_up_hypothesis_queries,
    build_retrieval_query,
    expand_queries_with_main_chapter_terms,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever


TARGET_GROUPS: dict[str, list[list[str]]] = {
    "血骑士退役后为何选择在卡西米尔乡间买田生活，村民最初又怎样认出他？": [
        ["activities/act13side/level_act13side_hidden_st03#chunk-0009"],
    ],
    "斯凯为什么认为雅斯彭不适合继续经营“幸运瘤兽”，这项决定会怎样影响受雇的体验员？": [
        ["activities/act47side/level_act47side_01_beg#chunk-0009"],
    ],
    "托希娅为什么拒绝教小贝蒂做生意？她如何解释帮助竞争对手与自身利益的冲突？": [
        ["activities/act20mini/level_act20mini_st02#chunk-0007"],
        ["activities/act20mini/level_act20mini_st02#chunk-0008"],
    ],
    "格特鲁德为什么没有按原计划彻底毁掉车尔尼，又向别格勒承认了哪件与哥哥有关的事？": [
        ["activities/act18side/level_act18side_08_beg#chunk-0007"],
    ],
    "伊芙利特用哪些具体回忆劝迷迭香停手，这些回忆共同说明了什么？": [
        ["activities/act25side/level_act25side_06_end#chunk-0004"],
    ],
    "凯尔希为什么没有让Mon3tr攻击“皇帝的利刃”，她又如何评价内卫作为人类的意义？": [
        ["activities/act18d0/level_act18d0_08_end#chunk-0000"],
        ["activities/act18d0/level_act18d0_08_end#chunk-0005"],
    ],
    "乔迪为何判断自己看到的并非普通梦境，而是Ishar-mla的回忆？": [
        ["activities/act34side/level_act34side_st03#chunk-0015"],
    ],
    "娜塔莉娅怎样解释乌萨斯依靠战争维持运转，这与她在学校组织学生的经历有什么联系？": [
        ["activities/act10d5/level_act10d5_st04#chunk-0018"],
    ],
}


class DummyGenerator:
    max_tokens = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def current_queries(question: str, hypothesis: HypothesisDocument) -> list[str]:
    queries = [
        _resolve_referential_question(question, hypothesis.entities),
        build_retrieval_query(hypothesis),
    ]
    queries.extend(build_follow_up_hypothesis_queries(question, hypothesis))
    return dedupe(expand_queries_with_main_chapter_terms(queries))


def compact_queries(question: str, hypothesis: HypothesisDocument) -> list[str]:
    primary = hypothesis.entities[0] if hypothesis.entities else ""
    other_entities = [item for item in hypothesis.entities[1:5] if item and len(item) <= 12]
    keywords = [item for item in hypothesis.keywords[:8] if item and len(item) <= 12]
    return dedupe(
        [
            question,
            " ".join([primary, *other_entities[:2]]),
            " ".join([primary, *keywords[:3]]),
            " ".join([*other_entities[:2], *keywords[:3]]),
        ]
    )


def ids(items: list[dict[str, Any]]) -> list[str]:
    return [str((item.get("document") or {}).get("id") or item.get("id") or "") for item in items]


def target_diagnostics(target_groups: list[list[str]], ranked_ids: list[str]) -> dict[str, Any]:
    ranks: list[int | None] = []
    for group in target_groups:
        found = [ranked_ids.index(target) + 1 for target in group if target in ranked_ids]
        ranks.append(min(found) if found else None)
    return {
        "group_ranks": ranks,
        "groups_hit": sum(rank is not None for rank in ranks),
        "groups_total": len(ranks),
        "full_coverage": bool(ranks) and all(rank is not None for rank in ranks),
    }


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]


def main() -> int:
    args = parse_args()
    runtime = json.loads(args.runtime.read_text())
    retrieval = dict(runtime["retrieval"])
    valid = {field.name for field in fields(QueryConfig)}
    query_config = QueryConfig(**{key: value for key, value in retrieval.items() if key in valid})
    index_dir = Path(retrieval["index_dir"])
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=Path(retrieval["embedding_model_path"]),
        reranker_model_path=Path(retrieval["reranker_model_path"]),
        reranker_max_length=int(retrieval.get("reranker_max_length", 1024)),
        documents_path=index_dir / "documents.jsonl",
        faiss_index_path=index_dir / "faiss.index",
        bm25_tokens_path=index_dir / "bm25_tokens.pkl",
        sparse_index_path=index_dir / "sparse_index.pkl",
        index_metadata_path=index_dir / "index_meta.json",
        device=str(retrieval.get("device", "cuda")),
    )
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=DummyGenerator(),
        query_config=query_config,
        max_retrieval_rounds=1,
        prompt_evidence_top_k=10,
    )
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    cases = [row for row in rows if row["question"] in TARGET_GROUPS]

    # Warm model kernels and tokenization outside measured runs.
    retriever.dense_search("罗德岛", top_k=2)
    retriever.sparse_search("罗德岛", top_k=2)
    assert retriever.reranker is not None
    retriever.reranker.score("罗德岛", ["罗德岛"], batch_size=1)

    variants: list[tuple[str, Any]] = [
        ("current16", lambda full, compact: full[:16]),
        ("prefix8", lambda full, compact: full[:8]),
        ("prefix4", lambda full, compact: full[:4]),
        ("prefix2", lambda full, compact: full[:2]),
        ("question_only", lambda full, compact: [compact[0]]),
        ("compact4", lambda full, compact: compact[:4]),
    ]
    output_variants: list[dict[str, Any]] = []
    for variant_name, selector in variants:
        records: list[dict[str, Any]] = []
        for row in cases:
            question = str(row["question"])
            hypothesis = HypothesisDocument(**row["hypothesis"])
            full = current_queries(question, hypothesis)
            compact = compact_queries(question, hypothesis)
            selected = selector(full, compact)
            started = time.perf_counter()
            dense_hits, sparse_hits, evidence = pipeline._retrieve_round(
                question,
                hypothesis,
                selected,
                enable_minirag=False,
            )
            elapsed = time.perf_counter() - started
            evidence_ids = ids(evidence)
            record = {
                "question": question,
                "query_count": len(selected),
                "queries": selected,
                "elapsed_sec": round(elapsed, 6),
                "dense": target_diagnostics(TARGET_GROUPS[question], ids(dense_hits)),
                "sparse": target_diagnostics(TARGET_GROUPS[question], ids(sparse_hits)),
                "top32": target_diagnostics(TARGET_GROUPS[question], evidence_ids[:32]),
                "top10": target_diagnostics(TARGET_GROUPS[question], evidence_ids[:10]),
                "top_ids": evidence_ids[:10],
            }
            records.append(record)
            print(json.dumps({"variant": variant_name, "question": question, "elapsed": round(elapsed, 3), "top32": record["top32"]}, ensure_ascii=False), flush=True)
        elapsed_values = [float(record["elapsed_sec"]) for record in records]
        output_variants.append(
            {
                "name": variant_name,
                "latency": {
                    "mean_sec": round(statistics.fmean(elapsed_values), 6),
                    "median_sec": round(statistics.median(elapsed_values), 6),
                    "p95_sec": round(percentile(elapsed_values, 0.95), 6),
                    "max_sec": round(max(elapsed_values), 6),
                },
                "full_coverage_top32": sum(record["top32"]["full_coverage"] for record in records),
                "full_coverage_top10": sum(record["top10"]["full_coverage"] for record in records),
                "group_hits_top32": sum(record["top32"]["groups_hit"] for record in records),
                "group_total": sum(record["top32"]["groups_total"] for record in records),
                "records": records,
            }
        )
    payload = {"case_count": len(cases), "query_config": {key: value for key, value in retrieval.items() if key in valid}, "variants": output_variants}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{key: value for key, value in variant.items() if key != "records"} for variant in output_variants], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
