#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import random
from pathlib import Path
import sys
import time
from typing import Any

from tqdm import tqdm


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
from goldenglow.inference.cpu_pipeline import CPUInferencePipeline  # noqa: E402
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402
from scripts.evaluate_multiround_retrieval_recall import (  # noqa: E402
    DEFAULT_RUNTIME_CONFIG_PATH,
    build_generator,
    build_query_config,
    config_value,
    first_hit,
    load_runtime_config,
    resolve_path,
)
from scripts.evaluate_retrieval_recall import extract_gold_text, load_listwise, normalize_text, parse_mode_weights  # noqa: E402


FULL_CHAIN_TASK_TYPE = "full_chain_generation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/opd_candidates/qwen35_4b_full_chain_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sample_records(records: list[dict[str, Any]], *, sample: int | None, seed: int) -> list[dict[str, Any]]:
    filtered = [
        record
        for record in records
        if str(record.get("query") or "").strip() and extract_gold_text(record)
    ]
    if sample is None or sample >= len(filtered):
        return filtered
    rng = random.Random(seed)
    indexed = list(enumerate(filtered))
    rng.shuffle(indexed)
    return [record for _, record in indexed[:sample]]


def hit_payload(hit: tuple[int, str, float] | None) -> dict[str, Any] | None:
    if hit is None:
        return None
    return {"rank": int(hit[0]), "source": str(hit[1]), "score": float(hit[2])}


def top_doc_ids(evidence: list[dict[str, Any]], *, limit: int = 10) -> list[str]:
    return [str(item.get("id") or "") for item in evidence[:limit]]


def compact_evidence(evidence: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in evidence[:limit]:
        compact.append(
            {
                "doc_id": str(item.get("id") or ""),
                "activity_name": str(item.get("activity_name") or ""),
                "story_name": str(item.get("story_name") or ""),
                "stage_code": str(item.get("stage_code") or ""),
                "evidence_chain_roles": item.get("evidence_chain_roles"),
                "evidence_chain_score": item.get("evidence_chain_score"),
                "rerank_score": item.get("rerank_score"),
                "evidence_chain_text": str(item.get("evidence_chain_text") or "")[:1600],
                "clean_text": str(item.get("clean_text") or "")[:1200],
            }
        )
    return compact


def evidence_for_hit(item: dict[str, Any]) -> dict[str, Any]:
    if isinstance(item.get("document"), dict):
        return item
    return {
        "document": {
            "id": str(item.get("id") or item.get("doc_id") or ""),
            "clean_text": str(item.get("clean_text") or ""),
        },
        "evidence_chain_text": str(item.get("evidence_chain_text") or ""),
    }


def compact_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for step in trace:
        conclusion = step.get("conclusion") if isinstance(step.get("conclusion"), dict) else {}
        output.append(
            {
                "round": step.get("round"),
                "queries": step.get("queries") or [],
                "planner_action": step.get("planner_action") or "",
                "hypothesis_task_type": step.get("hypothesis_task_type") or "",
                "hypothesis": step.get("hypothesis") or {},
                "conclusion_task_type": step.get("conclusion_task_type") or "",
                "conclusion": conclusion,
                "missing_slots": step.get("missing_slots") or conclusion.get("missing_slots") or [],
                "follow_up_hypothesis_task_type": step.get("follow_up_hypothesis_task_type") or "",
                "follow_up_hypothesis": step.get("follow_up_hypothesis"),
                "next_round_queries": step.get("next_round_queries") or [],
                "evidence_summary": step.get("evidence_summary") or [],
            }
        )
    return output


def build_retrieval_metrics(
    *,
    evidence: list[dict[str, Any]],
    gold_text: str,
    max_k: int,
    args: argparse.Namespace,
    rounds_run: int,
    answer: str,
    generation_error: str,
) -> dict[str, Any]:
    hit_items = [evidence_for_hit(item) for item in evidence]
    hit = first_hit(hit_items, gold_text, max_k=max_k, args=args)
    final_action = ""
    return {
        "max_k": max_k,
        "final_hit": hit_payload(hit),
        "final_rank_value": int(hit[0]) if hit is not None else max_k + 1,
        "hit_at_1": bool(hit is not None and hit[0] <= 1),
        "hit_at_5": bool(hit is not None and hit[0] <= 5),
        "hit_at_10": bool(hit is not None and hit[0] <= 10),
        "hit_at_20": bool(hit is not None and hit[0] <= 20),
        "hit_at_50": bool(hit is not None and hit[0] <= 50),
        "top_doc_ids": top_doc_ids(evidence),
        "rounds_run": rounds_run,
        "answered": bool(str(answer or "").strip()),
        "generation_error": generation_error,
        "final_action": final_action,
    }


def make_candidate_record(
    *,
    candidate_id: str,
    question: str,
    dialogue_context: str,
    result_payload: dict[str, Any],
    evidence: list[dict[str, Any]],
    retrieval_metrics: dict[str, Any],
    gold_text: str,
    source_record: dict[str, Any],
    source_index: int,
    run_index: int,
    latency: float,
    generation_error: str,
) -> dict[str, Any]:
    trace = compact_trace(result_payload.get("retrieval_trace") or [])
    if trace:
        last_conclusion = trace[-1].get("conclusion")
        if isinstance(last_conclusion, dict):
            retrieval_metrics["final_action"] = str(last_conclusion.get("next_action") or "")
    return {
        "candidate_id": candidate_id,
        "task_type": FULL_CHAIN_TASK_TYPE,
        "question": question,
        "dialogue_context": dialogue_context,
        "prompt": "runtime_full_chain",
        "candidate": {
            "question": question,
            "answer": str(result_payload.get("answer") or ""),
            "intent": str(result_payload.get("intent") or ""),
            "final_hypothesis": result_payload.get("hypothesis") or {},
            "retrieval_trace": trace,
            "retrieval_query": str(result_payload.get("retrieval_query") or ""),
        },
        "evidence": compact_evidence(evidence),
        "retrieval_metrics": retrieval_metrics,
        "gold": normalize_text(gold_text)[:1600],
        "source": {
            "source_index": source_index,
            "run_index": run_index,
            "query_type": str(source_record.get("query_type") or ""),
            "answer": str(source_record.get("answer") or ""),
            "answer_focus": str(source_record.get("answer_focus") or ""),
        },
        "runtime": result_payload.get("model_runtime") or {},
        "latency": round(latency, 3),
        "generation_error": generation_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full runtime-chain OPD candidates from the trained 4B model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--listwise", type=Path, default=Path("data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/reranker_listwise.jsonl"))
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG_PATH)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--runs-per-question", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--dialogue-context", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--reranker-max-length", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--fusion-top-k", type=int, default=None)
    parser.add_argument("--reranker-candidate-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=50)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--minirag-top-k", type=int, default=None)
    parser.add_argument("--minirag-weight", type=float, default=None)
    parser.add_argument("--minirag-mode-weights", type=parse_mode_weights, default=None)
    parser.add_argument("--minirag-index", type=Path, default=None)
    parser.add_argument("--minirag-fusion-mode", choices=("score", "append"), default=None)
    parser.add_argument("--enable-neighbor-expansion", action="store_true", default=None)
    parser.add_argument("--disable-neighbor-expansion", dest="enable_neighbor_expansion", action="store_false")
    parser.add_argument("--neighbor-max-seed-docs", type=int, default=None)
    parser.add_argument("--neighbor-story-window", type=int, default=None)
    parser.add_argument("--neighbor-activity-story-sort-window", type=int, default=None)
    parser.add_argument("--prompt-evidence-top-k", type=int, default=None)
    parser.add_argument("--enable-mmr", action="store_true", default=None)
    parser.add_argument("--disable-mmr", dest="enable_mmr", action="store_false")
    parser.add_argument("--mmr-lambda", type=float, default=None)
    parser.add_argument("--enable-pyramid-order", action="store_true", default=None)
    parser.add_argument("--disable-pyramid-order", dest="enable_pyramid_order", action="store_false")
    parser.add_argument("--enable-crag-refinement", action="store_true", default=None)
    parser.add_argument("--disable-crag-refinement", dest="enable_crag_refinement", action="store_false")
    parser.add_argument("--crag-refine-top-sentences", type=int, default=None)
    parser.add_argument("--crag-refine-max-sentences", type=int, default=None)
    parser.add_argument("--self-consistency-samples", type=int, default=None)
    parser.add_argument("--self-consistency-temperature", type=float, default=None)
    parser.add_argument("--backend", choices=("vllm", "llama.cpp"), default=None)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--llama-cli", type=Path, default=None)
    parser.add_argument("--gguf-model", type=Path, default=None)
    parser.add_argument("--lora-gguf", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--llama-device", default=None)
    parser.add_argument("--llama-gpu-layers", default=None)
    parser.add_argument("--llama-batch-size", type=int, default=None)
    parser.add_argument("--llama-ubatch-size", type=int, default=None)
    parser.add_argument("--llama-flash-attn", choices=("on", "off", "auto"), default=None)
    parser.add_argument("--ctx-size", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--jaccard-threshold", type=float, default=0.25)
    parser.add_argument("--overlap-threshold", type=float, default=0.32)
    parser.add_argument("--min-overlap-grams", type=int, default=60)
    parser.add_argument("--min-candidate-grams", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = args.runtime_config if args.runtime_config.is_absolute() else PROJECT_ROOT / args.runtime_config
    runtime_config = load_runtime_config(runtime_config_path)
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    inference_cfg = runtime_config.get("inference", {}) if isinstance(runtime_config.get("inference"), dict) else {}
    generator_cfg = runtime_config.get("generator", {}) if isinstance(runtime_config.get("generator"), dict) else {}

    listwise_path = args.listwise if args.listwise.is_absolute() else PROJECT_ROOT / args.listwise
    all_records = load_listwise(listwise_path)
    records = sample_records(all_records, sample=args.sample, seed=args.seed)
    index_dir = args.index_dir if args.index_dir.is_absolute() else PROJECT_ROOT / args.index_dir
    device = str(config_value(args.device, retrieval_cfg, "device", "cuda"))
    max_k = int(args.rerank_top_k)
    max_rounds = int(config_value(args.max_rounds, inference_cfg, "max_retrieval_rounds", 3))

    enable_reranker = bool(retrieval_cfg.get("enable_reranker", True)) and not args.no_reranker
    configured_reranker = retrieval_cfg.get("reranker_model_path") or retrieval_cfg.get("reranker_model")
    reranker_model = (
        resolve_path(args.reranker_model if args.reranker_model is not None else configured_reranker, default=RERANKER_MODEL_DIR)
        if enable_reranker
        else None
    )
    minirag_index = resolve_path(
        args.minirag_index if args.minirag_index is not None else retrieval_cfg.get("minirag_index_path"),
        default=MINIRAG_GRAPH_PATH if bool(retrieval_cfg.get("enable_minirag", True)) else None,
    )
    print(f"[load] full-chain records={len(records)} index={index_dir} device={device}", flush=True)
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=reranker_model,
        reranker_max_length=int(config_value(args.reranker_max_length, retrieval_cfg, "reranker_max_length", 1024)),
        documents_path=index_dir / "documents.jsonl" if (index_dir / "documents.jsonl").exists() else DOCUMENTS_PATH,
        faiss_index_path=index_dir / "faiss.index" if (index_dir / "faiss.index").exists() else FAISS_INDEX_PATH,
        bm25_tokens_path=index_dir / "bm25_tokens.pkl" if (index_dir / "bm25_tokens.pkl").exists() else BM25_TOKENS_PATH,
        minirag_index_path=minirag_index,
        device=device,
    )
    query_config = build_query_config(args, retrieval_cfg, rerank_top_k=max_k)
    generator = build_generator(args, generator_cfg)
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=generator,
        query_config=query_config,
        max_retrieval_rounds=max_rounds,
        prompt_evidence_top_k=int(config_value(args.prompt_evidence_top_k, inference_cfg, "prompt_evidence_top_k", 8)),
        enable_mmr=bool(config_value(args.enable_mmr, inference_cfg, "enable_mmr", False)),
        mmr_lambda=float(config_value(args.mmr_lambda, inference_cfg, "mmr_lambda", 0.72)),
        enable_pyramid_order=bool(config_value(args.enable_pyramid_order, inference_cfg, "enable_pyramid_order", False)),
        enable_crag_refinement=bool(config_value(args.enable_crag_refinement, inference_cfg, "enable_crag_refinement", False)),
        crag_refine_top_sentences=int(config_value(args.crag_refine_top_sentences, inference_cfg, "crag_refine_top_sentences", 4)),
        crag_refine_max_sentences=int(config_value(args.crag_refine_max_sentences, inference_cfg, "crag_refine_max_sentences", 24)),
        self_consistency_samples=int(config_value(args.self_consistency_samples, inference_cfg, "self_consistency_samples", 1)),
        self_consistency_temperature=float(config_value(args.self_consistency_temperature, inference_cfg, "self_consistency_temperature", 0.7)),
    )

    candidates_path = output_dir / "candidates.jsonl"
    failures_path = output_dir / "failed_candidates.jsonl"
    stats: Counter[str] = Counter()
    per_record: list[dict[str, Any]] = []
    started = time.time()
    total_runs = len(records) * max(1, args.runs_per_question)
    progress = tqdm(total=total_runs, desc="opd full-chain generation", unit="chain")
    for source_index, record in enumerate(records):
        question = str(record.get("query") or "").strip()
        gold_text = extract_gold_text(record) or ""
        if not question or not gold_text:
            stats["skipped"] += 1
            continue
        for run_index in range(max(1, args.runs_per_question)):
            candidate_id = f"{source_index:06d}-{FULL_CHAIN_TASK_TYPE}-{run_index:02d}"
            run_started = time.time()
            try:
                result = pipeline.run(question, args.dialogue_context)
                result_payload = asdict(result)
                evidence = result_payload.get("evidence") or []
                generation_error = ""
                retrieval_metrics = build_retrieval_metrics(
                    evidence=evidence,
                    gold_text=gold_text,
                    max_k=max_k,
                    args=args,
                    rounds_run=len(result_payload.get("retrieval_trace") or []),
                    answer=str(result_payload.get("answer") or ""),
                    generation_error=generation_error,
                )
                record_payload = make_candidate_record(
                    candidate_id=candidate_id,
                    question=question,
                    dialogue_context=args.dialogue_context,
                    result_payload=result_payload,
                    evidence=evidence,
                    retrieval_metrics=retrieval_metrics,
                    gold_text=gold_text,
                    source_record=record,
                    source_index=source_index,
                    run_index=run_index,
                    latency=time.time() - run_started,
                    generation_error=generation_error,
                )
                append_jsonl(candidates_path, record_payload)
                stats["completed"] += 1
                stats[f"hit@{max_k}:{bool(retrieval_metrics['final_hit'])}"] += 1
                stats[f"final_action:{retrieval_metrics.get('final_action') or 'unknown'}"] += 1
                per_record.append(
                    {
                        "candidate_id": candidate_id,
                        "question": question,
                        "final_hit": retrieval_metrics["final_hit"],
                        "final_action": retrieval_metrics.get("final_action") or "",
                        "rounds_run": retrieval_metrics["rounds_run"],
                        "latency": record_payload["latency"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                stats["failed"] += 1
                append_jsonl(
                    failures_path,
                    {
                        "candidate_id": candidate_id,
                        "question": question,
                        "error": f"{type(exc).__name__}: {exc}",
                        "created_at": int(time.time()),
                    },
                )
            progress.update(1)
            if args.progress_every > 0 and (stats["completed"] + stats["failed"]) % args.progress_every == 0:
                progress.set_postfix({"ok": stats["completed"], "failed": stats["failed"]})
    progress.close()

    summary = {
        "records": len(records),
        "runs_per_question": max(1, args.runs_per_question),
        "total_runs": total_runs,
        "output_dir": str(output_dir),
        "runtime_config": str(runtime_config_path),
        "listwise": str(listwise_path),
        "max_k": max_k,
        "max_rounds": max_rounds,
        "query_config": asdict(query_config),
        "generator_runtime": generator.describe_runtime(),
        "stats": dict(stats),
        "wall_seconds": round(time.time() - started, 2),
        "per_record": per_record,
    }
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
