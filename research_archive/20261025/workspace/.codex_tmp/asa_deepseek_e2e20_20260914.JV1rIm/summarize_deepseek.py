"""Recompute paired counts from raw results and explicit manual review, not F1."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latency(values):
    ordered = sorted(values)
    return {
        "n": len(values), "mean_sec": round(statistics.mean(values), 3),
        "median_sec": round(statistics.median(values), 3),
        "p90_nearest_rank_sec": ordered[math.ceil(len(values) * 0.9) - 1],
        "min_sec": min(values), "max_sec": max(values),
        "sum_sec": round(sum(values), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = jsonl(args.run_dir / "results.jsonl")
    calls = [x for x in jsonl(args.run_dir / "calls.jsonl") if x["event"] == "generation_end"]
    audit = jsonl(args.run_dir / "pipeline_audit.jsonl")
    reviews = json.loads(args.review.read_text())
    baseline = json.loads(args.baseline_summary.read_text())
    completion = json.loads((args.run_dir / "completion.json").read_text())
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    assert len(rows) == len(reviews) == 20
    assert [x["question_index"] for x in reviews] == list(range(1, 21))
    assert completion["questions_match"]
    assert hashlib.sha256((args.run_dir / "results.jsonl").read_bytes()).hexdigest() == (
        completion["results_sha256"]
    )
    assert manifest["questions_sha256"] == baseline["question_selection_sha256"]
    assert manifest["rubric_sha256"] == baseline["frozen_protocol_sha256"]
    snapshots = [x for x in audit if x["event"] == "prompt_evidence"]
    retrievals = [x for x in audit if x["event"] == "retrieval"]
    http_calls = [x for x in calls if "request_number" in x]
    with_usage = [x for x in calls if x.get("usage")]
    evidence_calls = [x for x in calls if x["stage"] != "user_question_hypothesis_generation"]
    guard_calls = [x for x in evidence_calls if "qwen_guard_input_tokens" in x]
    per_question = []
    for index, (row, review, old) in enumerate(zip(rows, reviews, baseline["per_question"]), 1):
        assert row["question"] == old["question"]
        qcalls = [x for x in calls if x["question_index"] == index]
        qretrievals = [x for x in retrievals if x["question_index"] == index]
        old_correct = old["category"] == "correct"
        new_correct = review["category"] == "correct"
        transition = (
            "both_correct" if old_correct and new_correct else
            "deepseek_only_correct" if new_correct else
            "4b_only_correct" if old_correct else "both_not_fully_correct"
        )
        action = (row.get("retrieval_trace") or [{}])[-1].get("planner_action", "error")
        per_question.append({
            **review, "question": row["question"], "answer": row.get("answer", ""),
            "error": row.get("error", ""), "elapsed_sec": row["elapsed_sec"],
            "retrieval_rounds": len(qretrievals),
            "queries_per_round": [len(x["queries"]) for x in qretrievals],
            "model_calls": len(qcalls), "final_action": action,
            "api_error_count": sum(x["status"] != "ok" for x in qcalls),
            "4b_answer": old["answer"], "4b_category": old["category"],
            "4b_elapsed_sec": old["elapsed_sec"], "strict_transition": transition,
            "stage_durations_sec": row.get("stage_durations_sec", {}),
        })
    strict = sum(x["category"] == "correct" for x in reviews)
    loose = sum(x["loosely_usable"] for x in reviews)
    assert all(x["loosely_usable"] for x in reviews if x["category"] == "correct")
    usage = Counter()
    for call in with_usage:
        usage.update({k: v for k, v in call["usage"].items() if isinstance(v, int)})
    p = strict / 20
    z = 1.959963984540054
    denominator = 1 + z*z/20
    center = (p + z*z/40) / denominator
    margin = z * math.sqrt(p*(1-p)/20 + z*z/(4*20*20)) / denominator
    summary = {
        "experiment": "DeepSeek full E2E same20 Chinese development regression, 2026-09-14",
        "denominator": 20,
        "strict_correct_count": strict, "strict_correct_percent": strict * 5,
        "loosely_usable_count": loose, "loosely_usable_percent": loose * 5,
        "categories": dict(Counter(x["category"] for x in reviews)),
        "final_actions": dict(Counter(x["final_action"] for x in per_question)),
        "pipeline_completed": sum(not x.get("error") for x in rows),
        "runtime_errors": sum(bool(x.get("error")) for x in rows),
        "model_call_count": len(calls), "http_request_count": len(http_calls),
        "http_status_counts": dict(Counter(x.get("http_status") for x in http_calls)),
        "generation_error_count": sum(x["status"] != "ok" for x in calls),
        "questions_with_any_api_error": [x["question_index"] for x in per_question if x["api_error_count"]],
        "finish_reason_counts": dict(Counter(x.get("finish_reason") for x in http_calls)),
        "returned_models": dict(Counter(x.get("returned_model") for x in http_calls)),
        "context_overflows": sum(x.get("error_type") == "PromptContextOverflowError" for x in calls),
        "reasoning_content_nonempty_count": sum(bool(x.get("reasoning_content_nonempty")) for x in calls),
        "snapshot_count": len(snapshots),
        "snapshot_evidence_counts": dict(Counter(x["evidence_count"] for x in snapshots)),
        "snapshot_full_body_checks": sum(len(x["body_checks"]) for x in snapshots),
        "snapshot_body_missing_count": sum(not b["full_body_present"] for x in snapshots for b in x["body_checks"]),
        "http_full_body_checks": sum(len(x.get("outgoing_evidence_body_checks", [])) for x in http_calls),
        "http_body_missing_count": sum(not b["full_body_present"] for x in http_calls for b in x.get("outgoing_evidence_body_checks", [])),
        "provider_prompt_tokens_min": min(x["usage"]["prompt_tokens"] for x in evidence_calls if x.get("usage")),
        "provider_prompt_tokens_max": max(x["usage"]["prompt_tokens"] for x in evidence_calls if x.get("usage")),
        "common_qwen_guard_input_min": min(x["qwen_guard_input_tokens"] for x in guard_calls),
        "common_qwen_guard_input_max": max(x["qwen_guard_input_tokens"] for x in guard_calls),
        "common_qwen_guard_input_plus_output_max": max(x["qwen_guard_input_tokens"] + x["max_output_tokens"] for x in guard_calls),
        "usage_totals": dict(usage),
        "latency_all_questions": latency([x["elapsed_sec"] for x in rows]),
        "latency_excluding_first": latency([x["elapsed_sec"] for x in rows[1:]]),
        "under_20_seconds_count": sum(x["elapsed_sec"] < 20 for x in rows),
        "retrieval_round_counts": dict(Counter(x["retrieval_rounds"] for x in per_question)),
        "retrieval_queries_per_round": [len(x["queries"]) for x in retrievals],
        "runner_total_elapsed_sec": completion["total_elapsed_sec"],
        "4b_strict_percent": baseline["strict_correct_percent"],
        "4b_loose_percent": baseline["loosely_usable_percent"],
        "strict_delta_percentage_points": strict * 5 - baseline["strict_correct_percent"],
        "paired_transitions": dict(Counter(x["strict_transition"] for x in per_question)),
        "iid_wilson95_strict_percent_illustrative_only": [
            round((center-margin)*100, 2), round((center+margin)*100, 2)
        ],
        "interval_caveat": "These are selected correlated development cases, not an IID sample; interval is illustrative, not a production coverage guarantee.",
        "quality_review_method": "Same main-assistant manual original-story rubric; no API judge; final answers only",
        "per_question": per_question,
    }
    with args.output.open("x") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
