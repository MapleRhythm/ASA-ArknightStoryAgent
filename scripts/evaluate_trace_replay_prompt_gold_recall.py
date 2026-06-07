#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"
if TRAIN_PYTHON_OVERLAY_DIR.exists():
    sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import (  # noqa: E402
    BM25_TOKENS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_DIR,
    FAISS_INDEX_PATH,
    INDEX_ROOT,
    MINIRAG_GRAPH_PATH,
    RERANKER_MODEL_DIR,
)
from goldenglow.inference.cpu_pipeline import (  # noqa: E402
    CPUInferencePipeline,
    HypothesisDocument,
    merge_evidence_keep_order,
    render_short_evidence_brief,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402
from scripts.analyze_soda_gold_evidence_topk import (  # noqa: E402
    cumulative_counts,
    extract_gold_units,
    first_hit_rank,
    extract_evidence_lines,
)
from scripts.evaluate_multiround_retrieval_recall import (  # noqa: E402
    build_query_config,
    load_runtime_config,
    resolve_path,
)


DEFAULT_TRACE = PROJECT_ROOT / "data/processed/llama_factory/soda_eval50_len1800_api_verifier_v1_noweb_gpu3_merged/teacher_full_chain.jsonl"
DEFAULT_AUDIT = PROJECT_ROOT / "data/processed/llama_factory/soda_eval50_len1800_blackbox_v1_noweb_gpu3_merged/audit_records.jsonl"
DEFAULT_RUNTIME = PROJECT_ROOT / "configs/runtime_inference_gpu.json"


class NullArgs:
    def __getattr__(self, _name: str) -> None:
        return None


class DummyGenerator:
    max_tokens = 1

    def describe_runtime(self) -> dict[str, Any]:
        return {"generator_backend": "dummy-trace-replay"}

    def generate(self, *_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("trace replay does not call generation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved retrieval traces and measure prompt-visible gold evidence.")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--audit-records", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--top-ks", default="1,3,5,8,10,12")
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument(
        "--modes",
        default="current",
        help="Comma-separated: current,no_scoped,no_sweep,no_scoped_no_sweep.",
    )
    return parser.parse_args()


def parse_top_ks(raw: str) -> list[int]:
    return sorted({int(item) for item in raw.split(",") if item.strip()})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_gold_units(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    question_re = re.compile(r"(?m)^question:\s*(.+?)\s*$")
    round_re = re.compile(r"(?m)^round:\s*(.+?)\s*$")
    gold: dict[tuple[str, str], list[dict[str, str]]] = {}
    for record in read_jsonl(path):
        if record.get("task_type") != "conclusion_generation" or not bool(record.get("kto_tag")):
            continue
        prompt = str((record.get("conversations") or [{}])[0].get("value") or "")
        question_match = question_re.search(prompt)
        round_match = round_re.search(prompt)
        if not question_match or not round_match:
            continue
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
        units = extract_gold_units(str(source.get("gold") or ""))
        if units:
            gold[(question_match.group(1).strip(), round_match.group(1).strip())] = units
    return gold


def build_pipeline(
    *,
    retriever: ArknightsHybridRetriever,
    runtime_config: dict[str, Any],
    mode: str,
) -> CPUInferencePipeline:
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    inference_cfg = runtime_config.get("inference", {}) if isinstance(runtime_config.get("inference"), dict) else {}
    qcfg = build_query_config(NullArgs(), retrieval_cfg, rerank_top_k=int(retrieval_cfg.get("rerank_top_k", 32)))
    if mode in {"no_scoped", "no_scoped_no_sweep"}:
        qcfg.enable_scoped_chapter_search = False
    if mode in {"no_sweep", "no_scoped_no_sweep"}:
        qcfg.enable_same_story_sweep = False
    return CPUInferencePipeline(
        retriever=retriever,
        generator=DummyGenerator(),
        query_config=qcfg,
        max_retrieval_rounds=int(inference_cfg.get("max_retrieval_rounds", 2)),
        prompt_evidence_top_k=int(inference_cfg.get("prompt_evidence_top_k", 12)),
        prompt_evidence_max_chars_per_doc=int(inference_cfg.get("prompt_evidence_max_chars_per_doc", 1800)),
        prompt_conclusion_evidence_max_total_chars=int(
            inference_cfg.get("prompt_conclusion_evidence_max_total_chars", 24000)
        ),
        enable_mmr=bool(inference_cfg.get("enable_mmr", False)),
        mmr_lambda=float(inference_cfg.get("mmr_lambda", 0.72)),
        enable_pyramid_order=bool(inference_cfg.get("enable_pyramid_order", False)),
        enable_evidence_pinning=bool(inference_cfg.get("enable_evidence_pinning", False)),
        enable_crag_refinement=bool(inference_cfg.get("enable_crag_refinement", False)),
        crag_refine_top_sentences=int(inference_cfg.get("crag_refine_top_sentences", 4)),
        crag_refine_max_sentences=int(inference_cfg.get("crag_refine_max_sentences", 24)),
        conclusion_prompt_mode=str(inference_cfg.get("conclusion_prompt_mode", "minimal")),
        web_context_config={"enabled": False},
    )


def evaluate_mode(
    *,
    pipeline: CPUInferencePipeline,
    trace_records: list[dict[str, Any]],
    gold_by_key: dict[tuple[str, str], list[dict[str, str]]],
    top_ks: list[int],
    threshold: float,
) -> dict[str, Any]:
    prompt_rows: list[dict[str, Any]] = []
    gold_unit_ranks: list[int | None] = []
    prompt_any_ranks: list[int | None] = []
    prompt_all_ranks: list[int | None] = []
    by_round: dict[str, dict[str, list[Any]]] = {}
    scope_records = 0
    local_dense_counts: list[int] = []
    local_sparse_counts: list[int] = []

    for record_index, record in enumerate(trace_records, start=1):
        question = str(record.get("question") or "")
        retained_chapter_scope: str | None = None
        retained_storyline_scope: str | None = None
        retained_scope_evidence: list[dict[str, Any]] = []
        scope_retention_enabled = False
        for step in record.get("retrieval_trace") or []:
            round_id = f"{step.get('round')}/2"
            units = gold_by_key.get((question, round_id))
            if not units:
                continue
            hypothesis = HypothesisDocument(**step["hypothesis"])
            queries = [str(item) for item in (step.get("queries") or []) if str(item).strip()]
            if (
                int(step.get("round") or 1) == 1
                and pipeline.query_config.minirag_chapter_isolation
                and pipeline.query_config.minirag_auto_second_retrieval
            ):
                _, _, evidence, expansion_record = pipeline._retrieve_first_round_with_scoped_minirag_expansion(
                    question,
                    hypothesis,
                    queries,
                )
                if expansion_record is not None:
                    scope_records += 1
                    retained_chapter_scope = str(expansion_record.get("chapter_scope") or "").strip() or None
                    retained_storyline_scope = str(expansion_record.get("storyline_scope") or "").strip() or None
                    scope_retention_enabled = bool(expansion_record.get("use_scoped_candidates") and retained_chapter_scope)
                    local_dense_counts.append(int(expansion_record.get("scoped_local_dense_hit_count") or 0))
                    local_sparse_counts.append(int(expansion_record.get("scoped_local_sparse_hit_count") or 0))
            else:
                _, _, evidence = pipeline._retrieve_round(
                    question,
                    hypothesis,
                    queries,
                    minirag_chapter_scope=retained_chapter_scope if scope_retention_enabled else None,
                    candidate_chapter_scope=retained_chapter_scope if scope_retention_enabled else None,
                    sparse_storyline_scope=retained_storyline_scope if scope_retention_enabled else None,
                )
            if scope_retention_enabled and retained_scope_evidence and int(step.get("round") or 1) > 1:
                evidence = merge_evidence_keep_order(
                    retained_scope_evidence,
                    evidence,
                    limit=max(pipeline.query_config.reranker_candidate_top_k, pipeline.prompt_evidence_top_k * 2),
                )
            if int(step.get("round") or 1) == 1 and scope_retention_enabled:
                retained_scope_evidence = list(evidence)

            selected = pipeline.prepare_prompt_evidence(question, hypothesis, evidence)
            rendered = (
                "evidence_brief:\n"
                + render_short_evidence_brief(
                    selected,
                    max_chars_per_doc=pipeline.prompt_evidence_max_chars_per_doc,
                    max_total_chars=pipeline.prompt_conclusion_evidence_max_total_chars,
                )
                + "\noutput_schema:"
            )
            evidence_lines = extract_evidence_lines(rendered)
            ranks_for_prompt: list[int | None] = []
            for unit in units:
                rank, _, _ = first_hit_rank(unit["text"], evidence_lines, threshold=threshold)
                ranks_for_prompt.append(rank)
                gold_unit_ranks.append(rank)
            hits = [rank for rank in ranks_for_prompt if rank is not None]
            any_rank = min(hits) if hits else None
            all_rank = max(hits) if len(hits) == len(ranks_for_prompt) else None
            coverage = len(hits) / len(ranks_for_prompt) if ranks_for_prompt else 0.0
            prompt_any_ranks.append(any_rank)
            prompt_all_ranks.append(all_rank)
            prompt_rows.append(
                {
                    "record_index": record_index,
                    "question": question,
                    "round": round_id,
                    "gold_units": len(ranks_for_prompt),
                    "hit_units": len(hits),
                    "unit_ranks": ranks_for_prompt,
                    "any_rank": any_rank,
                    "all_rank": all_rank,
                    "coverage": round(coverage, 4),
                }
            )
            bucket = by_round.setdefault(round_id, {"unit": [], "any": [], "all": [], "coverage": []})
            bucket["unit"].extend(ranks_for_prompt)
            bucket["any"].append(any_rank)
            bucket["all"].append(all_rank)
            bucket["coverage"].append(coverage)

    coverages = [row["coverage"] for row in prompt_rows]
    return {
        "counts": {
            "prompts": len(prompt_rows),
            "questions": len({row["question"] for row in prompt_rows}),
            "gold_units": len(gold_unit_ranks),
            "scope_records": scope_records,
        },
        "gold_unit_cumulative": cumulative_counts(gold_unit_ranks, top_ks),
        "prompt_any_gold_cumulative": cumulative_counts(prompt_any_ranks, top_ks),
        "prompt_all_gold_cumulative": cumulative_counts(prompt_all_ranks, top_ks),
        "prompt_gold_coverage": {
            "mean": round(statistics.mean(coverages), 4) if coverages else 0.0,
            "p50": round(statistics.median(coverages), 4) if coverages else 0.0,
            "full_coverage_count": sum(value >= 1.0 for value in coverages),
            "zero_coverage_count": sum(value <= 0.0 for value in coverages),
        },
        "rounds": {
            round_id: {
                "gold_unit_cumulative": cumulative_counts(payload["unit"], top_ks),
                "prompt_any_gold_cumulative": cumulative_counts(payload["any"], top_ks),
                "prompt_all_gold_cumulative": cumulative_counts(payload["all"], top_ks),
                "coverage_mean": round(statistics.mean(payload["coverage"]), 4) if payload["coverage"] else 0.0,
                "prompts": len(payload["coverage"]),
            }
            for round_id, payload in sorted(by_round.items())
        },
        "scoped_local_stats": {
            "dense_mean": round(statistics.mean(local_dense_counts), 2) if local_dense_counts else 0.0,
            "sparse_mean": round(statistics.mean(local_sparse_counts), 2) if local_sparse_counts else 0.0,
            "dense_max": max(local_dense_counts, default=0),
            "sparse_max": max(local_sparse_counts, default=0),
        },
        "prompt_rows": prompt_rows,
    }


def main() -> int:
    args = parse_args()
    runtime_path = args.runtime_config if args.runtime_config.is_absolute() else PROJECT_ROOT / args.runtime_config
    runtime_config = load_runtime_config(runtime_path)
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    device = str(args.device or retrieval_cfg.get("device") or "cuda")
    reranker_model = None
    if not args.no_reranker and bool(retrieval_cfg.get("enable_reranker", True)):
        configured = retrieval_cfg.get("reranker_model_path") or retrieval_cfg.get("reranker_model")
        reranker_model = resolve_path(args.reranker_model if args.reranker_model is not None else configured, default=RERANKER_MODEL_DIR)
    index_dir = args.index_dir if args.index_dir.is_absolute() else PROJECT_ROOT / args.index_dir
    sys.stderr.write(f"[load] retriever device={device} reranker={reranker_model}\n")
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=reranker_model,
        reranker_max_length=int(retrieval_cfg.get("reranker_max_length", 1024)),
        documents_path=index_dir / "documents.jsonl" if (index_dir / "documents.jsonl").exists() else DOCUMENTS_PATH,
        faiss_index_path=index_dir / "faiss.index" if (index_dir / "faiss.index").exists() else FAISS_INDEX_PATH,
        bm25_tokens_path=index_dir / "bm25_tokens.pkl" if (index_dir / "bm25_tokens.pkl").exists() else BM25_TOKENS_PATH,
        minirag_index_path=Path(retrieval_cfg.get("minirag_index_path") or MINIRAG_GRAPH_PATH),
        device=device,
    )
    trace_path = args.trace if args.trace.is_absolute() else PROJECT_ROOT / args.trace
    audit_path = args.audit_records if args.audit_records.is_absolute() else PROJECT_ROOT / args.audit_records
    records = read_jsonl(trace_path)
    if args.sample_offset:
        records = records[args.sample_offset :]
    if args.sample is not None:
        records = records[: args.sample]
    gold_by_key = load_gold_units(audit_path)
    top_ks = parse_top_ks(args.top_ks)
    output: dict[str, Any] = {
        "settings": {
            "trace": str(trace_path),
            "audit_records": str(audit_path),
            "runtime_config": str(runtime_path),
            "device": device,
            "reranker_model": str(reranker_model) if reranker_model else None,
            "records": len(records),
            "sample": args.sample,
            "sample_offset": args.sample_offset,
            "top_ks": top_ks,
            "match_threshold": args.match_threshold,
        },
        "modes": {},
    }
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        sys.stderr.write(f"[eval] mode={mode} records={len(records)}\n")
        pipeline = build_pipeline(retriever=retriever, runtime_config=runtime_config, mode=mode)
        output["modes"][mode] = evaluate_mode(
            pipeline=pipeline,
            trace_records=records,
            gold_by_key=gold_by_key,
            top_ks=top_ks,
            threshold=args.match_threshold,
        )
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({mode: output["modes"][mode]["prompt_gold_coverage"] for mode in output["modes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
