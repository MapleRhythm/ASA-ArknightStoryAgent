#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import re
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
    QueryConfig,
    RERANKER_MODEL_DIR,
)
from goldenglow.inference.cpu_pipeline import (  # noqa: E402
    CONCLUSION_TASK_TYPE,
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    INITIAL_HYPOTHESIS_TASK_TYPE,
    CPUInferencePipeline,
    ConclusionResult,
    HypothesisDocument,
    LlamaCppRunner,
    VllmRunner,
    _resolve_referential_question,
    build_conclusion_prompt,
    build_follow_up_hypothesis_prompt,
    build_hypothesis,
    build_hypothesis_prompt,
    build_retrieval_query,
    build_unresolved_points,
    extract_json_object,
    merge_hypotheses,
    normalize_conclusion_payload,
    normalize_hypothesis_payload,
    repair_json_like_output,
    render_dialogue_context_for_prompt,
    validate_conclusion_grounding,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402
from scripts.evaluate_multiround_retrieval_recall import (  # noqa: E402
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_GGUF_MODEL_PATH,
    DEFAULT_LLAMA_CLI_PATH,
    DEFAULT_RUNTIME_CONFIG_PATH,
    DEFAULT_VLLM_LORA_PATH,
    build_generator,
    build_generation_trace_record,
    build_query_config,
    config_value,
    first_hit,
    infer_missing_slots,
    load_runtime_config,
    merge_evidence_pool,
    resolve_path,
)
from scripts.evaluate_retrieval_recall import extract_gold_text, load_listwise, normalize_text, parse_mode_weights  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/opd_candidates/qwen35_4b_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def rank_value(hit: tuple[int, str, float] | None, *, miss_rank: int) -> int:
    return int(hit[0]) if hit is not None else miss_rank


def build_retrieval_metrics(
    *,
    before_hit: tuple[int, str, float] | None,
    after_hit: tuple[int, str, float] | None,
    before_top_doc_ids: list[str],
    after_top_doc_ids: list[str],
    max_k: int,
) -> dict[str, Any]:
    before_rank = rank_value(before_hit, miss_rank=max_k + 1)
    after_rank = rank_value(after_hit, miss_rank=max_k + 1)
    rank_delta = before_rank - after_rank
    return {
        "max_k": max_k,
        "before_hit": hit_payload(before_hit),
        "after_hit": hit_payload(after_hit),
        "before_rank_value": before_rank,
        "after_rank_value": after_rank,
        "rank_delta": rank_delta,
        "miss_to_hit": before_hit is None and after_hit is not None,
        "hit_to_top20": before_rank > 20 and after_rank <= 20,
        "hit_to_top10": before_rank > 10 and after_rank <= 10,
        "improved": rank_delta > 0,
        "before_top_doc_ids": before_top_doc_ids[:10],
        "after_top_doc_ids": after_top_doc_ids[:10],
    }


def normalize_raw_json_output(raw_output: str) -> dict[str, Any] | None:
    repaired = repair_json_like_output(raw_output)
    return extract_json_object(repaired)


def generate_json_candidates(
    generator: LlamaCppRunner | VllmRunner,
    prompt: str,
    *,
    samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for sample_index in range(samples):
        started = time.time()
        try:
            raw = generator.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            )
            payload = normalize_raw_json_output(raw)
            outputs.append(
                {
                    "sample_index": sample_index,
                    "ok": payload is not None,
                    "payload": payload,
                    "raw_output": raw,
                    "latency": round(time.time() - started, 3),
                    "error": "" if payload is not None else "invalid_json",
                }
            )
        except Exception as exc:  # noqa: BLE001
            outputs.append(
                {
                    "sample_index": sample_index,
                    "ok": False,
                    "payload": None,
                    "raw_output": "",
                    "latency": round(time.time() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return outputs


def retrieve_with_hypothesis(
    pipeline: CPUInferencePipeline,
    *,
    question: str,
    hypothesis: HypothesisDocument,
    include_question_query: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    queries = [build_retrieval_query(hypothesis)]
    if include_question_query:
        queries = [_resolve_referential_question(question, hypothesis.entities), *queries]
    _, _, hits = pipeline._retrieve_round(question, hypothesis, queries)
    return queries, hits


def top_doc_ids(hits: list[dict[str, Any]], *, limit: int = 10) -> list[str]:
    return [str((item.get("document") or {}).get("id") or "") for item in hits[:limit]]


def make_candidate_record(
    *,
    candidate_id: str,
    task_type: str,
    question: str,
    prompt: str,
    candidate_payload: dict[str, Any],
    evidence: list[dict[str, Any]],
    retrieval_metrics: dict[str, Any] | None,
    gold_text: str,
    source_record: dict[str, Any],
    source_index: int,
    sample_index: int,
    raw_output: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "task_type": task_type,
        "question": question,
        "prompt": prompt,
        "candidate": candidate_payload,
        "evidence": [
            {
                "doc_id": str((item.get("document") or {}).get("id") or ""),
                "evidence_chain_text": str(item.get("evidence_chain_text") or "")[:1600],
                "clean_text": str((item.get("document") or {}).get("clean_text") or "")[:1200],
            }
            for item in evidence[:8]
        ],
        "retrieval_metrics": retrieval_metrics or {},
        "gold": normalize_text(gold_text)[:1200],
        "source": {
            "source_index": source_index,
            "sample_index": sample_index,
            "query_type": str(source_record.get("query_type") or ""),
            "answer": str(source_record.get("answer") or ""),
            "answer_focus": str(source_record.get("answer_focus") or ""),
        },
        "raw_output": raw_output,
    }


def build_trace_for_prompt(
    *,
    round_index: int,
    hypothesis: HypothesisDocument,
    queries: list[str],
    hits: list[dict[str, Any]],
    missing_slots: list[str],
) -> dict[str, Any]:
    return build_generation_trace_record(
        round_index=round_index,
        hypothesis=hypothesis,
        queries=queries,
        hits=hits,
        planner_action="retrieval_completed",
        missing_slots=missing_slots,
    )


def process_record(
    record: dict[str, Any],
    *,
    source_index: int,
    pipeline: CPUInferencePipeline,
    output_dir: Path,
    samples_per_prompt: int,
    max_k: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = str(record.get("query") or "").strip()
    dialogue_context = args.dialogue_context
    gold_text = extract_gold_text(record) or ""
    if not question or not gold_text:
        return {"question": question, "skipped": True, "reason": "missing question or gold"}

    stats: Counter[str] = Counter()
    records_path = output_dir / "candidates.jsonl"
    invalid_path = output_dir / "invalid_candidates.jsonl"
    trace_path = output_dir / "retrieval_traces.jsonl"

    initial_prompt = build_hypothesis_prompt(question, dialogue_context)
    initial_hypothesis = pipeline.build_hypothesis(question, dialogue_context)
    initial_queries, initial_hits = retrieve_with_hypothesis(
        pipeline,
        question=question,
        hypothesis=initial_hypothesis,
        include_question_query=True,
    )
    initial_hit = first_hit(initial_hits, gold_text, max_k=max_k, args=args)
    missing_slots = infer_missing_slots(
        question=question,
        hypothesis=initial_hypothesis,
        evidence=initial_hits,
        query_type=initial_hypothesis.query_type,
    )
    retrieval_trace = [
        build_trace_for_prompt(
            round_index=1,
            hypothesis=initial_hypothesis,
            queries=initial_queries,
            hits=initial_hits,
            missing_slots=missing_slots,
        )
    ]

    append_jsonl(
        records_path,
        make_candidate_record(
            candidate_id=f"{source_index:06d}-{INITIAL_HYPOTHESIS_TASK_TYPE}-00",
            task_type=INITIAL_HYPOTHESIS_TASK_TYPE,
            question=question,
            prompt=initial_prompt,
            candidate_payload=asdict(initial_hypothesis),
            evidence=[],
            retrieval_metrics={
                "initial_hit": hit_payload(initial_hit),
                "initial_top_doc_ids": top_doc_ids(initial_hits),
                "max_k": max_k,
            },
            gold_text=gold_text,
            source_record=record,
            source_index=source_index,
            sample_index=0,
            raw_output="",
        ),
    )
    stats[f"task:{INITIAL_HYPOTHESIS_TASK_TYPE}"] += 1

    previous_conclusion = ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=missing_slots,
        clarification_question="",
        follow_up_hypothesis=None,
    )
    follow_prompt = build_follow_up_hypothesis_prompt(
        question=question,
        current_hypothesis=initial_hypothesis,
        evidence=initial_hits,
        unresolved_points=build_unresolved_points(
            question,
            initial_hypothesis,
            initial_hits,
            retrieval_trace,
            missing_slots,
        ),
        retrieval_trace=retrieval_trace,
        previous_conclusion=previous_conclusion,
        current_round=2,
        max_retrieval_rounds=args.max_rounds,
        prompt_evidence_top_k=pipeline.prompt_evidence_top_k,
        prompt_evidence=pipeline.prepare_prompt_evidence(question, initial_hypothesis, initial_hits),
    )
    follow_samples = generate_json_candidates(
        pipeline.generator,
        follow_prompt,
        samples=samples_per_prompt,
        max_tokens=min(args.follow_max_tokens, pipeline.generator.max_tokens),
        temperature=args.sample_temperature,
        top_p=args.sample_top_p,
        repeat_penalty=args.sample_repeat_penalty,
    )
    for sample in follow_samples:
        sample_index = int(sample["sample_index"])
        candidate_id = f"{source_index:06d}-{FOLLOW_UP_HYPOTHESIS_TASK_TYPE}-{sample_index:02d}"
        if not sample["ok"] or not isinstance(sample.get("payload"), dict):
            append_jsonl(invalid_path, {"candidate_id": candidate_id, "task_type": FOLLOW_UP_HYPOTHESIS_TASK_TYPE, **sample})
            stats["invalid_follow_up"] += 1
            continue
        try:
            follow = normalize_hypothesis_payload(
                sample["payload"],
                question=question,
                dialogue_context=dialogue_context,
                current_intent=initial_hypothesis.intent,
            )
            merged = merge_hypotheses(initial_hypothesis, follow)
            after_queries, after_hits = retrieve_with_hypothesis(
                pipeline,
                question=question,
                hypothesis=merged,
                include_question_query=False,
            )
            after_hit = first_hit(after_hits, gold_text, max_k=max_k, args=args)
            metrics = build_retrieval_metrics(
                before_hit=initial_hit,
                after_hit=after_hit,
                before_top_doc_ids=top_doc_ids(initial_hits),
                after_top_doc_ids=top_doc_ids(after_hits),
                max_k=max_k,
            )
            metrics["queries"] = {"before": initial_queries, "after": after_queries}
            append_jsonl(
                records_path,
                make_candidate_record(
                    candidate_id=candidate_id,
                    task_type=FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
                    question=question,
                    prompt=follow_prompt,
                    candidate_payload={
                        "question": follow.question,
                        "query_type": follow.query_type,
                        "entities": follow.entities,
                        "keywords": follow.keywords,
                        "expected_answer_type": follow.expected_answer_type,
                        "dialogue_context": follow.dialogue_context,
                    },
                    evidence=initial_hits,
                    retrieval_metrics=metrics,
                    gold_text=gold_text,
                    source_record=record,
                    source_index=source_index,
                    sample_index=sample_index,
                    raw_output=str(sample.get("raw_output") or ""),
                ),
            )
            append_jsonl(
                trace_path,
                {
                    "candidate_id": candidate_id,
                    "question": question,
                    "initial_hypothesis": asdict(initial_hypothesis),
                    "follow_up_hypothesis": asdict(follow),
                    "merged_hypothesis": asdict(merged),
                    "retrieval_metrics": metrics,
                },
            )
            stats[f"task:{FOLLOW_UP_HYPOTHESIS_TASK_TYPE}"] += 1
            if metrics["improved"]:
                stats["follow_up_improved"] += 1
            if metrics["miss_to_hit"]:
                stats["follow_up_miss_to_hit"] += 1
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                invalid_path,
                {
                    "candidate_id": candidate_id,
                    "task_type": FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
                    "error": f"{type(exc).__name__}: {exc}",
                    **sample,
                },
            )
            stats["invalid_follow_up"] += 1

    conclusion_prompt = build_conclusion_prompt(
        question,
        initial_hypothesis,
        initial_hits,
        retrieval_trace,
        1,
        args.max_rounds,
        pipeline.prompt_evidence_top_k,
        prompt_evidence=pipeline.prepare_prompt_evidence(question, initial_hypothesis, initial_hits),
    )
    conclusion_samples = generate_json_candidates(
        pipeline.generator,
        conclusion_prompt,
        samples=samples_per_prompt,
        max_tokens=min(args.conclusion_max_tokens, pipeline.generator.max_tokens),
        temperature=args.sample_temperature,
        top_p=args.sample_top_p,
        repeat_penalty=args.sample_repeat_penalty,
    )
    for sample in conclusion_samples:
        sample_index = int(sample["sample_index"])
        candidate_id = f"{source_index:06d}-{CONCLUSION_TASK_TYPE}-{sample_index:02d}"
        if not sample["ok"] or not isinstance(sample.get("payload"), dict):
            append_jsonl(invalid_path, {"candidate_id": candidate_id, "task_type": CONCLUSION_TASK_TYPE, **sample})
            stats["invalid_conclusion"] += 1
            continue
        try:
            conclusion = normalize_conclusion_payload(
                sample["payload"],
                question=question,
                dialogue_context=dialogue_context,
                current_intent=initial_hypothesis.intent,
                max_round_reached=False,
            )
            conclusion = validate_conclusion_grounding(
                question=question,
                hypothesis=initial_hypothesis,
                evidence=pipeline.prepare_prompt_evidence(question, initial_hypothesis, initial_hits),
                conclusion=conclusion,
                max_round_reached=False,
            )
            metrics: dict[str, Any] = {
                "max_k": max_k,
                "before_hit": hit_payload(initial_hit),
                "before_rank_value": rank_value(initial_hit, miss_rank=max_k + 1),
                "before_top_doc_ids": top_doc_ids(initial_hits),
                "next_action": conclusion.next_action,
            }
            if conclusion.follow_up_hypothesis is not None:
                merged = merge_hypotheses(initial_hypothesis, conclusion.follow_up_hypothesis)
                after_queries, after_hits = retrieve_with_hypothesis(
                    pipeline,
                    question=question,
                    hypothesis=merged,
                    include_question_query=False,
                )
                after_hit = first_hit(after_hits, gold_text, max_k=max_k, args=args)
                metrics.update(
                    build_retrieval_metrics(
                        before_hit=initial_hit,
                        after_hit=after_hit,
                        before_top_doc_ids=top_doc_ids(initial_hits),
                        after_top_doc_ids=top_doc_ids(after_hits),
                        max_k=max_k,
                    )
                )
                metrics["queries"] = {"before": initial_queries, "after": after_queries}
                evidence_for_record = initial_hits
            else:
                metrics["after_hit"] = None
                metrics["rank_delta"] = 0
                evidence_for_record = initial_hits
            candidate_payload = {
                "question": question,
                "next_action": conclusion.next_action,
                "answer": conclusion.answer,
                "missing_slots": conclusion.missing_slots,
                "clarification_question": conclusion.clarification_question,
                "follow_up_hypothesis": (
                    {
                        "question": conclusion.follow_up_hypothesis.question,
                        "query_type": conclusion.follow_up_hypothesis.query_type,
                        "entities": conclusion.follow_up_hypothesis.entities,
                        "keywords": conclusion.follow_up_hypothesis.keywords,
                        "expected_answer_type": conclusion.follow_up_hypothesis.expected_answer_type,
                        "dialogue_context": conclusion.follow_up_hypothesis.dialogue_context,
                    }
                    if conclusion.follow_up_hypothesis is not None
                    else None
                ),
            }
            append_jsonl(
                records_path,
                make_candidate_record(
                    candidate_id=candidate_id,
                    task_type=CONCLUSION_TASK_TYPE,
                    question=question,
                    prompt=conclusion_prompt,
                    candidate_payload=candidate_payload,
                    evidence=evidence_for_record,
                    retrieval_metrics=metrics,
                    gold_text=gold_text,
                    source_record=record,
                    source_index=source_index,
                    sample_index=sample_index,
                    raw_output=str(sample.get("raw_output") or ""),
                ),
            )
            stats[f"task:{CONCLUSION_TASK_TYPE}"] += 1
            if metrics.get("improved"):
                stats["conclusion_follow_up_improved"] += 1
            if metrics.get("miss_to_hit"):
                stats["conclusion_follow_up_miss_to_hit"] += 1
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                invalid_path,
                {
                    "candidate_id": candidate_id,
                    "task_type": CONCLUSION_TASK_TYPE,
                    "error": f"{type(exc).__name__}: {exc}",
                    **sample,
                },
            )
            stats["invalid_conclusion"] += 1

    return {
        "question": question,
        "initial_hit": hit_payload(initial_hit),
        "initial_top_doc_ids": top_doc_ids(initial_hits),
        "stats": dict(stats),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OPD candidates from the trained 4B model with real retrieval metrics.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--listwise", type=Path, default=Path("data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/reranker_listwise.jsonl"))
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG_PATH)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
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
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--sample-top-p", type=float, default=0.9)
    parser.add_argument("--sample-repeat-penalty", type=float, default=1.05)
    parser.add_argument("--follow-max-tokens", type=int, default=384)
    parser.add_argument("--conclusion-max-tokens", type=int, default=512)
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
    print(f"[load] retriever records={len(records)} index={index_dir} device={device}", flush=True)
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
        max_retrieval_rounds=args.max_rounds,
        prompt_evidence_top_k=int(config_value(args.prompt_evidence_top_k, inference_cfg, "prompt_evidence_top_k", 8)),
        enable_mmr=bool(config_value(args.enable_mmr, inference_cfg, "enable_mmr", False)),
        mmr_lambda=float(config_value(args.mmr_lambda, inference_cfg, "mmr_lambda", 0.72)),
        enable_pyramid_order=bool(config_value(args.enable_pyramid_order, inference_cfg, "enable_pyramid_order", False)),
        enable_crag_refinement=bool(config_value(args.enable_crag_refinement, inference_cfg, "enable_crag_refinement", False)),
        crag_refine_top_sentences=int(config_value(args.crag_refine_top_sentences, inference_cfg, "crag_refine_top_sentences", 4)),
        crag_refine_max_sentences=int(config_value(args.crag_refine_max_sentences, inference_cfg, "crag_refine_max_sentences", 24)),
        self_consistency_samples=1,
    )

    stats: Counter[str] = Counter()
    per_record: list[dict[str, Any]] = []
    started = time.time()
    for index, record in enumerate(tqdm(records, desc="opd candidate generation", unit="question")):
        source_index = index
        try:
            result = process_record(
                record,
                source_index=source_index,
                pipeline=pipeline,
                output_dir=output_dir,
                samples_per_prompt=args.samples_per_prompt,
                max_k=max_k,
                args=args,
            )
            per_record.append(result)
            stats.update(result.get("stats") or {})
            stats["completed_questions"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["failed_questions"] += 1
            append_jsonl(
                output_dir / "failed_questions.jsonl",
                {
                    "source_index": source_index,
                    "question": str(record.get("query") or ""),
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": int(time.time()),
                },
            )
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(f"[progress] {index + 1}/{len(records)} stats={dict(stats)}", flush=True)

    summary = {
        "records": len(records),
        "output_dir": str(output_dir),
        "runtime_config": str(runtime_config_path),
        "listwise": str(listwise_path),
        "samples_per_prompt": args.samples_per_prompt,
        "max_k": max_k,
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
