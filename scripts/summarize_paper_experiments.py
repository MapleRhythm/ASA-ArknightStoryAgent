#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/paper_experiments"
DEFAULT_PAPER_SOURCE = PROJECT_ROOT / "docs/asa_rag_soda_paper.md"

RELEASE_VARIANTS = [
    "full_current_v6",
    "no_neighbor_v6",
    "no_minirag_v6",
    "old_reranker_full",
    "skip_rerank_full",
]

EASY_HARD_VARIANTS = [
    "full_current_v6",
    "no_neighbor_v6",
    "no_minirag_v6",
    "old_reranker_full",
    "skip_rerank_full",
]

FAILURE_VARIANTS = [
    "failure_full_current_v6",
    "failure_no_neighbor_v6",
    "failure_no_minirag_v6",
    "failure_old_reranker_full",
    "failure_skip_rerank_full",
]

MULTIROUND_RUNS = [
    ("sft_quality_fix3_plus_api175_brief_v2", "sft_quality_fix3_plus_api175_brief_v2_sample50_tp2.json"),
    ("sft_quality_fix3", "sft_quality_fix3_sample50_tp2.json"),
    ("sft_baseline", "sft_baseline_sample50_tp2.json"),
    ("kto_v2_full_runtime", "kto_v2_full_runtime_sample50_tp2.json"),
    ("kto_v3_full_runtime", "kto_v3_full_runtime_sample50_tp2.json"),
]

FINAL_RUNTIME_RUNS = [
    ("api_qc_sft_4b_cutoff3072_mem52", "outputs/eval_api_qc_sft_4b_cutoff3072_mem52_20260603_134858/summary.json"),
    ("quote80_sft_v1", "outputs/eval_quote80_sft_v1_20260605_095553/summary.json"),
    ("kto_quote80_verifier_cut5632", "outputs/eval50_hard10_kto_quote80_verifier_cut5632_filtered_20260605_184448/summary.json"),
    (
        "soda_targeted_human_v3_current_chain_latest_kto",
        "outputs/eval50_hard10_soda_targeted_human_v3_current_chain_latest_kto_retryenv_20260606_235346/summary.json",
    ),
    ("mergedbase_cutoff6656", "outputs/eval50_hard10_mergedbase_cutoff6656_trainenv_20260607_175322/summary.json"),
]

API_NO_VERIFIER_RUNS = [
    "baseline_standard",
    "improved_standard_retryfix",
    "question_retrieve_refine_full10",
]

RERANKER_EVALS = [
    ("base_bge_reranker_v2_m3", "outputs/evidence_chain_reranker_eval/base_eval.json"),
    ("rank_mix_v5_warm", "outputs/evidence_chain_reranker_eval/rank_mix_v5_new_eval.json"),
    ("rank_mix_v6_small_patch", "outputs/evidence_chain_reranker_eval/rank_mix_v6_small_patch_eval.json"),
]

LATENCY_RUNS = [
    ("cpu", "outputs/retrieval_latency_reasoning_cpu.json"),
    ("gpu", "outputs/retrieval_latency_train_gpu.json"),
]

SOURCE_ORACLE_RUNS = [
    ("eval50_minirag_v3_prerank_oracle", "outputs/retrieval_eval/eval50_len1800_minirag_v3_prerank_oracle.json"),
]

PROMPT_REPLAY_RUNS = [
    ("scoped_sweep_replay", "outputs/soda_flow_reports/eval50_trace_replay_gpu_reranker_merged_scoped_sweep.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation outputs used by the ASA RAG/SODA paper.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-source", type=Path, default=DEFAULT_PAPER_SOURCE)
    parser.add_argument(
        "--suite-summary",
        type=Path,
        default=None,
        help="Optional suite_summary.json from scripts/run_paper_ablation_suite.py to include current-run results.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write partial output instead of failing when an expected artifact is missing.",
    )
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def rel(path: Path | str) -> str:
    resolved = resolve(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path: Path | str, *, allow_missing: bool = False) -> Any:
    resolved = resolve(path)
    if not resolved.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(rel(resolved))
    return json.loads(resolved.read_text(encoding="utf-8"))


def read_jsonl(path: Path | str, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    resolved = resolve(path)
    if not resolved.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(rel(resolved))
    records: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
    return records


def pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def metric_row(tag: str, payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload["overall"]
    recall = overall.get("recall", {})
    count = int(overall.get("count") or 0)
    missed = int(overall.get("missed") or 0)
    return {
        "tag": tag,
        "count": count,
        "missed": missed,
        "missed_text": f"{missed}/{count}" if count else str(missed),
        "mrr": overall.get("mrr", overall.get("mrr_global_round_rank")),
        "mean_first_hit_rank": overall.get("mean_first_hit_rank"),
        "r@1": recall.get("@1"),
        "r@5": recall.get("@5"),
        "r@10": recall.get("@10"),
        "r@20": recall.get("@20"),
        "r@32": recall.get("@32"),
        "r@50": recall.get("@50"),
        "wall_seconds": payload.get("wall_seconds"),
        "path": rel(payload.get("_source_path", "")),
    }


def load_retrieval_ablation(allow_missing: bool) -> dict[str, Any]:
    base = Path("outputs/ablation_release_20260531")
    rows = []
    by_type: dict[str, Any] = {}
    for tag in RELEASE_VARIANTS:
        path = base / f"{tag}.json"
        payload = read_json(path, allow_missing=allow_missing)
        if payload is None:
            continue
        payload["_source_path"] = path
        rows.append(metric_row(tag, payload))
        by_type[tag] = payload.get("by_query_type", {})
    return {
        "rows": rows,
        "by_query_type": by_type,
        "report": rel(base / "report.md"),
    }


def load_easy_hard(allow_missing: bool) -> dict[str, Any]:
    base = Path("outputs/ablation_easy_hard_20260531")
    rows = []
    for split in ("easy", "hard"):
        for variant in EASY_HARD_VARIANTS:
            tag = f"{split}_{variant}"
            path = base / f"{tag}.json"
            payload = read_json(path, allow_missing=allow_missing)
            if payload is None:
                continue
            payload["_source_path"] = path
            row = metric_row(variant, payload)
            row["split"] = split
            row["raw_tag"] = tag
            rows.append(row)
    split_summary = read_json(
        "data/processed/evidence_chain_reranker/easy_hard_split_v1/split_summary.json",
        allow_missing=allow_missing,
    )
    return {
        "rows": rows,
        "split_summary": split_summary,
        "report": rel(base / "report.md"),
    }


def load_failure_hard(allow_missing: bool) -> dict[str, Any]:
    base = Path("outputs/ablation_failure_hard_20260531")
    rows = []
    for tag in FAILURE_VARIANTS:
        path = base / f"{tag}.json"
        payload = read_json(path, allow_missing=allow_missing)
        if payload is None:
            continue
        payload["_source_path"] = path
        rows.append(metric_row(tag, payload))
    pool_summary = read_json(
        "data/processed/evidence_chain_reranker/failure_hard_pool_v1/failure_hard_pool_summary.json",
        allow_missing=allow_missing,
    )
    return {
        "rows": rows,
        "pool_summary": pool_summary,
        "report": rel(base / "report.md"),
    }


def load_reranker_diagnostics(allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    for label, path in RERANKER_EVALS:
        payload = read_json(path, allow_missing=allow_missing)
        if payload is None:
            continue
        pairwise = payload.get("pairwise", {})
        listwise = payload.get("listwise", {})
        rows.append(
            {
                "model": label,
                "pairwise_count": pairwise.get("count"),
                "pairwise_accuracy": pairwise.get("accuracy"),
                "pairwise_mean_margin": pairwise.get("mean_margin"),
                "listwise_count": listwise.get("count"),
                "listwise_top1": listwise.get("top1_accuracy"),
                "listwise_mrr": listwise.get("mrr"),
                "listwise_mean_margin": listwise.get("mean_gold_vs_best_negative_margin"),
                "path": rel(path),
            }
        )
    return rows


def load_source_oracles(allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    for label, path in SOURCE_ORACLE_RUNS:
        payload = read_json(path, allow_missing=allow_missing)
        if payload is None:
            continue
        sources = (payload.get("source_oracle") or {}).get("sources", {})
        for source_name, stats in sources.items():
            recall = stats.get("recall", {})
            rows.append(
                {
                    "run": label,
                    "source": source_name,
                    "count": stats.get("count"),
                    "missed": stats.get("missed"),
                    "mrr": stats.get("mrr"),
                    "mean_first_hit_rank": stats.get("mean_first_hit_rank"),
                    "r@5": recall.get("@5"),
                    "r@12": recall.get("@12"),
                    "r@32": recall.get("@32"),
                    "r@80": recall.get("@80"),
                    "r@120": recall.get("@120"),
                    "path": rel(path),
                }
            )
    return rows


def load_prompt_replay_ablation(allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    for label, path in PROMPT_REPLAY_RUNS:
        payload = read_json(path, allow_missing=allow_missing)
        if payload is None:
            continue
        settings = payload.get("settings", {})
        for mode, mode_payload in (payload.get("modes") or {}).items():
            counts = mode_payload.get("counts", {})
            coverage = mode_payload.get("prompt_gold_coverage", {})
            scoped = mode_payload.get("scoped_local_stats", {})
            rows.append(
                {
                    "run": label,
                    "mode": mode,
                    "prompts": counts.get("prompts"),
                    "questions": counts.get("questions"),
                    "gold_units": counts.get("gold_units"),
                    "scope_records": counts.get("scope_records"),
                    "gold_unit@12": ((mode_payload.get("gold_unit_cumulative") or {}).get("@12") or {}).get("ratio"),
                    "prompt_any@12": ((mode_payload.get("prompt_any_gold_cumulative") or {}).get("@12") or {}).get("ratio"),
                    "prompt_all@12": ((mode_payload.get("prompt_all_gold_cumulative") or {}).get("@12") or {}).get("ratio"),
                    "coverage_mean": coverage.get("mean"),
                    "coverage_p50": coverage.get("p50"),
                    "full_coverage_count": coverage.get("full_coverage_count"),
                    "zero_coverage_count": coverage.get("zero_coverage_count"),
                    "scoped_dense_mean": scoped.get("dense_mean"),
                    "scoped_sparse_mean": scoped.get("sparse_mean"),
                    "match_threshold": settings.get("match_threshold"),
                    "path": rel(path),
                }
            )
    return rows


def load_prompt_gold(allow_missing: bool) -> dict[str, Any]:
    path = "outputs/soda_flow_reports/eval50_len1800_v2_scoped_sweep_soda_lora_gpu3_merged_gold_topk.json"
    payload = read_json(path, allow_missing=allow_missing)
    if payload is None:
        return {"path": rel(path)}
    payload["path"] = rel(path)
    return payload


def load_soda_verifier(allow_missing: bool) -> dict[str, Any]:
    blackbox_dir = Path("data/processed/llama_factory/soda_eval50_len1800_blackbox_v2_scoped_sweep_soda_lora_gpu3_merged")
    verifier_dir = Path("data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_merged")
    blackbox_summary = read_json(blackbox_dir / "build_summary.json", allow_missing=allow_missing) or {}
    verifier_summary = read_json(verifier_dir / "build_summary.json", allow_missing=allow_missing) or {}
    verifier_records = read_jsonl(verifier_dir / "api_verifier_records.jsonl", allow_missing=allow_missing)
    blackbox_records = read_jsonl(blackbox_dir / "audit_records.jsonl", allow_missing=allow_missing)

    question_keys = {
        str((record.get("meta") or {}).get("question_key") or "")
        for record in blackbox_records
        if isinstance(record.get("meta"), dict) and (record.get("meta") or {}).get("question_key")
    }
    verifier_stats: Counter[str] = Counter()
    for record in verifier_records:
        verdict = record.get("verifier") if isinstance(record.get("verifier"), dict) else {}
        verifier_stats[f"correct_action:{verdict.get('correct_action') or '<missing>'}"] += 1
        verifier_stats[f"evidence_sufficient:{bool(verdict.get('evidence_sufficient'))}"] += 1
        verifier_stats[f"student_action_error:{verdict.get('student_action_error') or '<missing>'}"] += 1
        verifier_stats[f"teacher_action_error:{verdict.get('teacher_action_error') or '<missing>'}"] += 1
        if verdict.get("teacher_answer_uses_prior_knowledge"):
            verifier_stats["teacher_answer_uses_prior_knowledge"] += 1
        verifier_stats[f"use_for_training:{bool(verdict.get('use_for_training', True))}"] += 1

    return {
        "blackbox_dataset_dir": rel(blackbox_dir),
        "verifier_dataset_dir": rel(verifier_dir),
        "question_count": len(question_keys) or None,
        "blackbox_build_summary": blackbox_summary,
        "verifier_build_summary": verifier_summary,
        "verifier_stats": dict(sorted(verifier_stats.items())),
        "verifier_record_count": len(verifier_records),
        "audit_report": rel("outputs/soda_flow_reports/eval50_len1800_v2_scoped_sweep_soda_lora_gpu3_merged_api_verifier_audit.md"),
    }


def load_multiround_runtime(allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    for label, filename in MULTIROUND_RUNS:
        path = Path("outputs/eval_multiround_retrieval") / filename
        payload = read_json(path, allow_missing=allow_missing)
        if payload is None:
            continue
        overall = payload.get("overall", {})
        recall = overall.get("recall", {})
        rows.append(
            {
                "model_runtime": label,
                "count": overall.get("count"),
                "missed": overall.get("missed"),
                "mrr": overall.get("mrr_global_round_rank"),
                "r@1": recall.get("@1"),
                "r@5": recall.get("@5"),
                "r@10": recall.get("@10"),
                "r@50": recall.get("@50"),
                "generation_error_count": payload.get("generation_error_count"),
                "wall_seconds": payload.get("wall_seconds"),
                "path": rel(path),
            }
        )
    return rows


def load_api_no_verifier(allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    base = Path("outputs/api_no_verifier_ablation_20260602")
    for tag in API_NO_VERIFIER_RUNS:
        path = base / tag / "summary.json"
        payload = read_json(path, allow_missing=allow_missing)
        if payload is None:
            continue
        rows.append(
            {
                "run": tag,
                "question_count": payload.get("question_count"),
                "success_count": payload.get("success_count"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "runtime_config": rel(payload.get("runtime_config", "")),
                "records_path": rel(payload.get("records_path", "")),
                "summary_path": rel(path),
            }
        )
    return rows


def load_final_runtime_actions(allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    for label, path in FINAL_RUNTIME_RUNS:
        payload = read_json(path, allow_missing=allow_missing)
        if payload is None:
            continue
        for summary in payload.get("summaries", []):
            if not isinstance(summary, dict):
                continue
            final_actions = summary.get("final_actions", {})
            action_sequences = summary.get("action_sequences", {})
            rows.append(
                {
                    "run": label,
                    "dataset": summary.get("name"),
                    "count": summary.get("count"),
                    "errors": summary.get("errors"),
                    "abstain_like": summary.get("abstain_like"),
                    "answer_directly": final_actions.get("answer_directly"),
                    "abstain": final_actions.get("abstain"),
                    "retrieve_answer": action_sequences.get("retrieve_more>answer_directly"),
                    "retrieve_abstain": action_sequences.get("retrieve_more>abstain"),
                    "avg_elapsed_sec": summary.get("avg_elapsed_sec"),
                    "path": rel(path),
                }
            )
    return rows


def load_latency(allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    for label, path in LATENCY_RUNS:
        payload = read_json(path, allow_missing=allow_missing)
        if payload is None:
            continue
        stage = payload.get("stage_latency_summary", {})
        rows.append(
            {
                "run": label,
                "device": (payload.get("environment") or {}).get("device"),
                "query_count": (payload.get("environment") or {}).get("query_count"),
                "repeat": (payload.get("environment") or {}).get("repeat"),
                "end_to_end_mean_ms": (stage.get("end_to_end_s") or {}).get("mean_ms"),
                "dense_mean_ms": (stage.get("dense_total_s") or {}).get("mean_ms"),
                "sparse_mean_ms": (stage.get("sparse_total_s") or {}).get("mean_ms"),
                "rerank_mean_ms": (stage.get("rerank_s") or {}).get("mean_ms"),
                "load_total_ms": (payload.get("load_timings_ms") or {}).get("load_total_s"),
                "path": rel(path),
            }
        )
    return rows


def load_suite_summary(path: Path | None, *, allow_missing: bool) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = read_json(path, allow_missing=allow_missing)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {rel(path)}")
    payload["_source_path"] = rel(path)
    return payload


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def render_retrieval_table(rows: list[dict[str, Any]], *, include_wall: bool = False) -> str:
    headers = ["Variant", "MRR", "Missed", "Mean first hit", "R@1", "R@5", "R@10", "R@20", "R@32"]
    if include_wall:
        headers.append("Wall s")
    table_rows = []
    for row in rows:
        missed_text = row.get("missed_text")
        if missed_text is None and row.get("count") is not None and row.get("missed") is not None:
            missed_text = f"{row.get('missed')}/{row.get('count')}"
        item = [
            f"`{row['tag']}`",
            pct(row.get("mrr"), 4),
            missed_text,
            pct(row.get("mean_first_hit_rank"), 3),
            pct(row.get("r@1"), 4),
            pct(row.get("r@5"), 4),
            pct(row.get("r@10"), 4),
            pct(row.get("r@20"), 4),
            pct(row.get("r@32"), 4),
        ]
        if include_wall:
            item.append(pct(row.get("wall_seconds"), 2))
        table_rows.append(item)
    return markdown_table(headers, table_rows)


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "# ASA RAG/SODA Experiment Summary",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "This file is generated from existing experiment artifacts by `scripts/summarize_paper_experiments.py`.",
        "",
        "## Source Artifacts",
        "",
    ]
    for source in summary["source_artifacts"]:
        lines.append(f"- `{source}`")

    suite = summary.get("paper_ablation_suite")
    if suite:
        lines.extend(["", "## Current Paper Ablation Suite", ""])
        lines.append(
            markdown_table(
                ["Field", "Value"],
                [
                    ["Run ID", f"`{suite.get('run_id')}`"],
                    ["Success", suite.get("success")],
                    ["Smoke", suite.get("smoke")],
                    ["Commands", len(suite.get("commands") or [])],
                    ["Run dir", f"`{suite.get('run_dir')}`"],
                ],
            )
        )
        if suite.get("ablation_release"):
            lines.extend(["", "### Suite Release Static Retrieval", ""])
            lines.append(render_retrieval_table(suite["ablation_release"], include_wall=True))
        if suite.get("ablation_easy_hard"):
            lines.extend(["", "### Suite Easy/Hard Static Retrieval", ""])
            easy_hard_suite_rows = []
            for row in suite["ablation_easy_hard"]:
                tag = str(row.get("tag") or "")
                split = "hard" if tag.startswith("hard_") else "easy" if tag.startswith("easy_") else ""
                easy_hard_suite_rows.append(
                    [
                        split,
                        f"`{tag}`",
                        pct(row.get("mrr"), 4),
                        f"{row.get('missed')}/{row.get('count')}",
                        pct(row.get("r@1"), 4),
                        pct(row.get("r@5"), 4),
                        pct(row.get("r@10"), 4),
                        pct(row.get("r@32"), 4),
                    ]
                )
            lines.append(
                markdown_table(
                    ["Split", "Variant", "MRR", "Missed", "R@1", "R@5", "R@10", "R@32"],
                    easy_hard_suite_rows,
                )
            )
        if suite.get("ablation_failure_hard"):
            lines.extend(["", "### Suite Failure-Hard Static Retrieval", ""])
            lines.append(render_retrieval_table(suite["ablation_failure_hard"], include_wall=True))
        if suite.get("source_oracle"):
            lines.extend(["", "### Suite Pre-Rerank Source Oracle", ""])
            lines.append(
                markdown_table(
                    ["Run", "Source", "Count", "Missed", "MRR", "R@5", "R@32", "R@120"],
                    [
                        [
                            f"`{row.get('run')}`",
                            row.get("source"),
                            row.get("count"),
                            row.get("missed"),
                            pct(row.get("mrr"), 4),
                            pct(row.get("r@5"), 4),
                            pct(row.get("r@32"), 4),
                            pct(row.get("r@120"), 4),
                        ]
                        for row in suite["source_oracle"]
                    ],
                )
            )
        if suite.get("prompt_replay"):
            lines.extend(["", "### Suite Prompt Evidence Replay", ""])
            lines.append(
                markdown_table(
                    ["Mode", "Prompts", "Questions", "Gold units", "Gold @12", "Any @12", "All @12", "Coverage"],
                    [
                        [
                            f"`{row.get('mode')}`",
                            row.get("prompts"),
                            row.get("questions"),
                            row.get("gold_units"),
                            pct(row.get("gold_unit@12"), 4),
                            pct(row.get("prompt_any@12"), 4),
                            pct(row.get("prompt_all@12"), 4),
                            pct(row.get("coverage_mean"), 4),
                        ]
                        for row in suite["prompt_replay"]
                    ],
                )
            )
        if suite.get("latency"):
            lines.extend(["", "### Suite Retrieval Latency", ""])
            lines.append(
                markdown_table(
                    ["Device", "Queries", "Repeat", "E2E mean ms", "Sparse ms", "Rerank ms"],
                    [
                        [
                            row.get("device"),
                            row.get("query_count"),
                            row.get("repeat"),
                            pct(row.get("end_to_end_mean_ms"), 3),
                            pct(row.get("sparse_mean_ms"), 3),
                            pct(row.get("rerank_mean_ms"), 3),
                        ]
                        for row in suite["latency"]
                    ],
                )
            )

    if not suite:
        lines.extend(["", "## Static Retrieval Ablation", ""])
        lines.append(render_retrieval_table(summary["retrieval_ablation"]["rows"], include_wall=True))

        lines.extend(["", "## Easy/Hard Split Retrieval Ablation", ""])
        easy_hard_rows = []
        for row in summary["easy_hard_ablation"]["rows"]:
            easy_hard_rows.append(
                [
                    row["split"],
                    f"`{row['tag']}`",
                    pct(row.get("mrr"), 4),
                    row.get("missed_text"),
                    pct(row.get("r@1"), 4),
                    pct(row.get("r@5"), 4),
                    pct(row.get("r@10"), 4),
                    pct(row.get("r@32"), 4),
                ]
            )
        lines.append(
            markdown_table(
                ["Split", "Variant", "MRR", "Missed", "R@1", "R@5", "R@10", "R@32"],
                easy_hard_rows,
            )
        )

        lines.extend(["", "## Failure-Driven Hard Pool", ""])
        lines.append(render_retrieval_table(summary["failure_hard_ablation"]["rows"]))

    lines.extend(["", "## Standalone Reranker Diagnostics", ""])
    lines.append(
        markdown_table(
            [
                "Model",
                "Pairwise n",
                "Pairwise acc",
                "Mean margin",
                "Listwise n",
                "Top1",
                "MRR",
                "Listwise margin",
            ],
            [
                [
                    f"`{row['model']}`",
                    row.get("pairwise_count"),
                    pct(row.get("pairwise_accuracy"), 4),
                    pct(row.get("pairwise_mean_margin"), 4),
                    row.get("listwise_count"),
                    pct(row.get("listwise_top1"), 4),
                    pct(row.get("listwise_mrr"), 4),
                    pct(row.get("listwise_mean_margin"), 4),
                ]
                for row in summary["reranker_diagnostics"]
            ],
        )
    )
    lines.append("")
    lines.append("Note: standalone reranker diagnostics use the eval files recorded with each model, so dataset size differs across rows.")

    lines.extend(["", "## Pre-Rerank Source Oracle", ""])
    lines.append(
        markdown_table(
            ["Run", "Source", "Count", "Missed", "MRR", "Mean rank", "R@5", "R@12", "R@32", "R@80", "R@120"],
            [
                [
                    f"`{row['run']}`",
                    row.get("source"),
                    row.get("count"),
                    row.get("missed"),
                    pct(row.get("mrr"), 4),
                    pct(row.get("mean_first_hit_rank"), 3),
                    pct(row.get("r@5"), 4),
                    pct(row.get("r@12"), 4),
                    pct(row.get("r@32"), 4),
                    pct(row.get("r@80"), 4),
                    pct(row.get("r@120"), 4),
                ]
                for row in summary["source_oracles"]
            ],
        )
    )

    prompt = summary["prompt_gold_coverage"]
    coverage = prompt.get("prompt_gold_coverage", {})
    lines.extend(["", "## Prompt Gold Coverage", ""])
    lines.append(
        markdown_table(
            ["Metric", "Value"],
            [
                ["Conclusion prompts", (prompt.get("counts") or {}).get("conclusion_prompts")],
                ["Questions", (prompt.get("counts") or {}).get("questions")],
                ["Gold units", (prompt.get("counts") or {}).get("gold_units_prompt_level")],
                ["Gold unit @1", pct(((prompt.get("gold_unit_cumulative") or {}).get("@1") or {}).get("ratio"), 4)],
                ["Gold unit @5", pct(((prompt.get("gold_unit_cumulative") or {}).get("@5") or {}).get("ratio"), 4)],
                ["Gold unit @12", pct(((prompt.get("gold_unit_cumulative") or {}).get("@12") or {}).get("ratio"), 4)],
                ["Prompt any gold @12", pct(((prompt.get("prompt_any_gold_cumulative") or {}).get("@12") or {}).get("ratio"), 4)],
                ["Prompt all gold @12", pct(((prompt.get("prompt_all_gold_cumulative") or {}).get("@12") or {}).get("ratio"), 4)],
                ["Question any gold @12", pct(((prompt.get("question_any_gold_cumulative") or {}).get("@12") or {}).get("ratio"), 4)],
                ["Question all gold @12", pct(((prompt.get("question_all_gold_cumulative") or {}).get("@12") or {}).get("ratio"), 4)],
                ["Prompt mean coverage", pct(coverage.get("mean"), 4)],
                ["Prompt zero coverage", pct(coverage.get("zero_coverage_ratio"), 4)],
                ["Prompt full coverage", pct(coverage.get("full_coverage_ratio"), 4)],
            ],
        )
    )

    lines.extend(["", "## Prompt Evidence Replay Ablation", ""])
    lines.append(
        markdown_table(
            [
                "Run",
                "Mode",
                "Prompts",
                "Questions",
                "Gold units",
                "Gold unit @12",
                "Any @12",
                "All @12",
                "Mean coverage",
                "Full count",
                "Zero count",
            ],
            [
                [
                    f"`{row['run']}`",
                    f"`{row['mode']}`",
                    row.get("prompts"),
                    row.get("questions"),
                    row.get("gold_units"),
                    pct(row.get("gold_unit@12"), 4),
                    pct(row.get("prompt_any@12"), 4),
                    pct(row.get("prompt_all@12"), 4),
                    pct(row.get("coverage_mean"), 4),
                    row.get("full_coverage_count"),
                    row.get("zero_coverage_count"),
                ]
                for row in summary["prompt_replay_ablation"]
            ],
        )
    )

    soda = summary["soda_verifier"]
    verifier_build = soda.get("verifier_build_summary") or {}
    verifier_stats = soda.get("verifier_stats") or {}
    lines.extend(["", "## Verifier-Aware SODA Dataset", ""])
    lines.append(
        markdown_table(
            ["Metric", "Value"],
            [
                ["Questions", soda.get("question_count")],
                ["Rollout raw pairs", (soda.get("blackbox_build_summary") or {}).get("raw_pairs")],
                ["Rollout KTO records", (soda.get("blackbox_build_summary") or {}).get("records_total")],
                ["Verifier records", verifier_build.get("verifier_records") or soda.get("verifier_record_count")],
                ["Teacher full-chain records", verifier_build.get("teacher_full_chain_records")],
                ["Final KTO records", verifier_build.get("records_total")],
                ["Train records", verifier_build.get("records_train")],
                ["Val records", verifier_build.get("records_val")],
                ["Student unsupported answer", verifier_stats.get("student_action_error:unsupported_answer")],
                ["Student over-retrieve", verifier_stats.get("student_action_error:over_retrieve")],
                ["Student premature answer", verifier_stats.get("student_action_error:premature_answer")],
                ["Teacher prior knowledge risk", verifier_stats.get("teacher_answer_uses_prior_knowledge")],
                ["Teacher unsupported answer", verifier_stats.get("teacher_action_error:unsupported_answer")],
            ],
        )
    )

    lines.extend(["", "## Historical Multiround Retrieval Runtime", ""])
    lines.append(
        markdown_table(
            ["Model / runtime", "Count", "Missed", "MRR", "R@1", "R@5", "R@10", "R@50", "Errors"],
            [
                [
                    f"`{row['model_runtime']}`",
                    row.get("count"),
                    row.get("missed"),
                    pct(row.get("mrr"), 4),
                    pct(row.get("r@1"), 4),
                    pct(row.get("r@5"), 4),
                    pct(row.get("r@10"), 4),
                    pct(row.get("r@50"), 4),
                    row.get("generation_error_count"),
                ]
                for row in summary["multiround_runtime"]
            ],
        )
    )

    lines.extend(["", "## API No-Verifier Runs", ""])
    lines.append(
        markdown_table(
            ["Run", "Questions", "Success", "Elapsed s", "Runtime config"],
            [
                [
                    f"`{row['run']}`",
                    row.get("question_count"),
                    row.get("success_count"),
                    pct(row.get("elapsed_seconds"), 3),
                    f"`{row['runtime_config']}`",
                ]
                for row in summary["api_no_verifier"]
            ],
        )
    )

    lines.extend(["", "## Final Runtime Action Summaries", ""])
    lines.append(
        markdown_table(
            [
                "Run",
                "Dataset",
                "Count",
                "Errors",
                "Abstain-like",
                "Answer",
                "Abstain",
                "Ret>Answer",
                "Ret>Abstain",
                "Avg sec",
            ],
            [
                [
                    f"`{row['run']}`",
                    row.get("dataset"),
                    row.get("count"),
                    row.get("errors"),
                    row.get("abstain_like"),
                    row.get("answer_directly"),
                    row.get("abstain"),
                    row.get("retrieve_answer"),
                    row.get("retrieve_abstain"),
                    pct(row.get("avg_elapsed_sec"), 3),
                ]
                for row in summary["final_runtime_actions"]
            ],
        )
    )

    lines.extend(["", "## Retrieval Latency", ""])
    lines.append(
        markdown_table(
            ["Run", "Device", "Queries", "Repeat", "E2E mean ms", "Dense ms", "Sparse ms", "Rerank ms", "Load total ms"],
            [
                [
                    row.get("run"),
                    row.get("device"),
                    row.get("query_count"),
                    row.get("repeat"),
                    pct(row.get("end_to_end_mean_ms"), 3),
                    pct(row.get("dense_mean_ms"), 3),
                    pct(row.get("sparse_mean_ms"), 3),
                    pct(row.get("rerank_mean_ms"), 3),
                    pct(row.get("load_total_ms"), 3),
                ]
                for row in summary["retrieval_latency"]
            ],
        )
    )

    reproduce_cmd = "python scripts/summarize_paper_experiments.py"
    suite = summary.get("paper_ablation_suite")
    if isinstance(suite, dict) and suite.get("_source_path"):
        reproduce_cmd += f" --suite-summary {suite['_source_path']}"
    lines.extend(["", "## Reproduce This Summary", "", "```bash", reproduce_cmd, "```", ""])
    return "\n".join(lines)


def collect_source_artifacts(summary: dict[str, Any]) -> list[str]:
    sources: set[str] = set()
    suite = summary.get("paper_ablation_suite")
    if isinstance(suite, dict):
        if suite.get("_source_path"):
            sources.add(suite["_source_path"])
        for collection in (
            "ablation_release",
            "ablation_easy_hard",
            "ablation_failure_hard",
            "source_oracle",
            "prompt_replay",
            "latency",
        ):
            for row in suite.get(collection, []):
                if row.get("path"):
                    sources.add(row["path"])
        for group in ("soda_audit", "paper_experiments"):
            for path in (suite.get(group) or {}).values():
                if path:
                    sources.add(path)
    if not isinstance(suite, dict):
        for section in (
            "retrieval_ablation",
            "easy_hard_ablation",
            "failure_hard_ablation",
        ):
            if summary.get(section, {}).get("report"):
                sources.add(summary[section]["report"])
            for row in summary.get(section, {}).get("rows", []):
                if row.get("path"):
                    sources.add(row["path"])
    for collection in (
        "reranker_diagnostics",
        "source_oracles",
        "prompt_replay_ablation",
        "multiround_runtime",
        "api_no_verifier",
        "final_runtime_actions",
        "retrieval_latency",
    ):
        for row in summary.get(collection, []):
            for key in ("path", "summary_path"):
                if row.get(key):
                    sources.add(row[key])
    for section in ("prompt_gold_coverage", "soda_verifier"):
        item = summary.get(section, {})
        for key in ("path", "audit_report", "blackbox_dataset_dir", "verifier_dataset_dir"):
            if item.get(key):
                sources.add(item[key])
    return sorted(sources)


def write_generated_paper(source: Path, output: Path, tables_md: str) -> None:
    body = source.read_text(encoding="utf-8")
    appendix = [
        "",
        "---",
        "",
        "# 自动生成实验附录",
        "",
        "以下内容由 `scripts/summarize_paper_experiments.py` 从当前实验产物生成。",
        "",
        tables_md,
    ]
    output.write_text(body.rstrip() + "\n" + "\n".join(appendix) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_source = resolve(args.paper_source)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "retrieval_ablation": load_retrieval_ablation(args.allow_missing),
        "easy_hard_ablation": load_easy_hard(args.allow_missing),
        "failure_hard_ablation": load_failure_hard(args.allow_missing),
        "reranker_diagnostics": load_reranker_diagnostics(args.allow_missing),
        "source_oracles": load_source_oracles(args.allow_missing),
        "prompt_gold_coverage": load_prompt_gold(args.allow_missing),
        "prompt_replay_ablation": load_prompt_replay_ablation(args.allow_missing),
        "soda_verifier": load_soda_verifier(args.allow_missing),
        "multiround_runtime": load_multiround_runtime(args.allow_missing),
        "api_no_verifier": load_api_no_verifier(args.allow_missing),
        "final_runtime_actions": load_final_runtime_actions(args.allow_missing),
        "retrieval_latency": load_latency(args.allow_missing),
        "paper_ablation_suite": load_suite_summary(args.suite_summary, allow_missing=args.allow_missing),
    }
    summary["source_artifacts"] = collect_source_artifacts(summary)

    tables_md = render_markdown(summary)
    summary_path = output_dir / "summary.json"
    tables_path = output_dir / "tables.md"
    paper_path = output_dir / "asa_rag_soda_paper.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tables_path.write_text(tables_md + "\n", encoding="utf-8")
    if paper_source.exists():
        write_generated_paper(paper_source, paper_path, tables_md)
    elif not args.allow_missing:
        raise FileNotFoundError(rel(paper_source))

    print(f"Wrote {rel(summary_path)}")
    print(f"Wrote {rel(tables_path)}")
    if paper_path.exists():
        print(f"Wrote {rel(paper_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
