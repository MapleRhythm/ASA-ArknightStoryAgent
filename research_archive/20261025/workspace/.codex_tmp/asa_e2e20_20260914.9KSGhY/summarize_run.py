"""Recompute mechanics from raw audit and quality counts from explicit human review."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latency(values):
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean_sec": round(statistics.mean(ordered), 3),
        "median_sec": round(statistics.median(ordered), 3),
        "p90_nearest_rank_sec": ordered[math.ceil(0.90 * len(ordered)) - 1],
        "min_sec": min(ordered),
        "max_sec": max(ordered),
        "total_sec": round(sum(ordered), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.run_dir / "results.jsonl")
    calls = [x for x in read_jsonl(args.run_dir / "calls.jsonl")
             if x["event"] == "generation_end"]
    audit = read_jsonl(args.run_dir / "pipeline_audit.jsonl")
    reviews = json.loads(args.review.read_text())
    assert len(rows) == len(reviews) == 20
    assert [x["question_index"] for x in reviews] == list(range(1, 21))
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    completion = json.loads((args.run_dir / "completion.json").read_text())
    assert completion["questions_match"]
    assert hashlib.sha256((args.run_dir / "results.jsonl").read_bytes()).hexdigest() == (
        completion["results_sha256"]
    )
    snapshots = [x for x in audit if x["event"] == "prompt_evidence"]
    retrievals = [x for x in audit if x["event"] == "retrieval"]
    actual_calls = [x for x in calls if "actual_input_tokens" in x]
    evidence_calls = [x for x in actual_calls if x["stage"] != "user_question_hypothesis_generation"]
    all_latencies = [x["elapsed_sec"] for x in rows]
    actions = [x.get("retrieval_trace", [{}])[-1].get("planner_action", "error") for x in rows]
    category_counts = Counter(x["category"] for x in reviews)
    strict = sum(x["category"] == "correct" for x in reviews)
    usable = sum(x["loosely_usable"] for x in reviews)
    assert all(x["loosely_usable"] for x in reviews if x["category"] == "correct")
    per_question = []
    for index, (row, review) in enumerate(zip(rows, reviews), 1):
        qcalls = [x for x in calls if x["question_index"] == index]
        qretrievals = [x for x in retrievals if x["question_index"] == index]
        per_question.append({
            **review,
            "question": row["question"],
            "answer": row["answer"],
            "runtime_error": row.get("error", ""),
            "elapsed_sec": row["elapsed_sec"],
            "retrieval_rounds": len(qretrievals),
            "queries_per_round": [len(x["queries"]) for x in qretrievals],
            "final_action": actions[index - 1],
            "model_calls": len(qcalls),
            "actual_input_tokens": [x.get("actual_input_tokens") for x in qcalls],
            "stage_durations_sec": row["stage_durations_sec"],
        })
    result = {
        "experiment": "4B full pipeline Chinese development regression, 2026-09-14",
        "denominator": 20,
        "question_selection_sha256": manifest["questions_sha256"],
        "frozen_protocol_sha256": manifest["protocol_sha256"],
        "runtime_error_count": sum(bool(x.get("error")) for x in rows),
        "pipeline_completion_count": sum(not x.get("error") for x in rows),
        "final_action_counts": dict(Counter(actions)),
        "human_category_counts": dict(category_counts),
        "strict_correct_count": strict,
        "strict_correct_percent": 100 * strict / 20,
        "loosely_usable_count": usable,
        "loosely_usable_percent": 100 * usable / 20,
        "incorrect_abstention_count": sum(x["category"] == "abstain" for x in reviews),
        "unanswerable_question_index": 14,
        "system_abstain_on_answerable_count": sum(
            action == "abstain" and index != 14
            for index, action in enumerate(actions, 1)
        ),
        "excerpt_only_fallback_count": sum(
            x["answer"].startswith("当前证据只能确认以下片段事实") for x in rows
        ),
        "correct_or_partial_boundary_answer_count": sum(
            x["question_index"] == 14 and x["category"] in ("correct", "partial")
            for x in reviews
        ),
        "latency_including_first_question_cold_start": latency(all_latencies),
        "latency_excluding_first_question": latency(all_latencies[1:]),
        "under_20_seconds_count": sum(x < 20 for x in all_latencies),
        "retrieval_round_counts": dict(Counter(x["retrieval_rounds"] for x in per_question)),
        "retrieval_query_counts": [len(x["queries"]) for x in retrievals],
        "model_generation_calls": len(calls),
        "generation_error_count": sum(x["status"] != "ok" for x in calls),
        "context_overflow_count": sum(
            x.get("error_type") == "PromptContextOverflowError" for x in calls
        ),
        "output_finish_reasons": dict(Counter(
            choice["finish_reason"] for x in calls for choice in x.get("engine_outputs", [])
        )),
        "prompt_evidence_snapshot_count": len(snapshots),
        "prompt_evidence_counts": dict(Counter(x["evidence_count"] for x in snapshots)),
        "evidence_body_checks": sum(len(x["body_checks"]) for x in snapshots),
        "incomplete_evidence_body_count": sum(
            not check["full_body_present"] for x in snapshots for check in x["body_checks"]
        ),
        "full_prompt_to_engine_token_mismatch_count": sum(
            not x["full_prompt_token_match"] for x in actual_calls
        ),
        "engine_response_input_mismatch_count": sum(
            not x["engine_input_matches"] for x in actual_calls if "engine_input_matches" in x
        ),
        "evidence_input_tokens_min": min(x["actual_input_tokens"] for x in evidence_calls),
        "evidence_input_tokens_max": max(x["actual_input_tokens"] for x in evidence_calls),
        "max_input_plus_output_budget": max(
            x["actual_input_tokens"] + x["max_output_tokens"] for x in actual_calls
        ),
        "runner_total_elapsed_sec": completion["total_elapsed_sec"],
        "quality_review_method": "Main assistant manually checked original Chinese story files; no API judge",
        "per_question": per_question,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
