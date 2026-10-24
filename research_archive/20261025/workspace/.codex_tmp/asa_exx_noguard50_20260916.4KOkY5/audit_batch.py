"""Mechanical auditing only: no semantic answer grading and no model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    rows = read_jsonl(directory / "results.jsonl")
    calls = read_jsonl(directory / "calls.jsonl")
    audit = read_jsonl(directory / "audit.jsonl")
    manifest = json.loads((directory / "manifest.json").read_text())
    completion = json.loads((directory / "completion.json").read_text())
    assert len(rows) == completion["count"] == 50
    assert completion["exit_code"] == 0 and completion["runtime_errors"] == 0
    assert completion["questions_match"]
    assert sha(directory / "results.jsonl") == completion["results_sha256"]
    assert sha(directory / "questions.txt") == manifest["questions_sha256"]
    assert manifest["questions_sha256"] == "be3c8bfb6cf1b9ac3ab391bb7ff039d673b8aa8ca598c20457bb6ac5d3110184"
    assert manifest["old_answers_or_gold_sent"] is False
    assert manifest["answer_validation_enabled"] is False
    requests = [event for event in calls if event["event"] == "request"]
    responses = [event for event in calls if event["event"] == "response"]
    engines = [event for event in calls if event["event"] == "engine_result"]
    errors = [event for event in calls if event["event"] == "error"]
    contexts = [event for event in audit if event["event"] == "prompt_evidence"]
    retrievals = [event for event in audit if event["event"] == "retrieval"]
    bypasses = [event for event in audit if event["event"] == "answer_validation_bypassed"]
    unexpected_checks = [event for event in audit if event["event"] == "unexpected_answer_validator"]
    problems = []
    if not len(requests) == len(responses) == len(engines) or errors:
        problems.append("generation call counts/errors")
    if unexpected_checks:
        problems.append("answer validator unexpectedly executed")
    if not bypasses or not all(event["input_equals_output"] for event in bypasses):
        problems.append("answer bypass absent or changed model conclusion")
    if not all(event["actual_input_matches"] for event in engines):
        problems.append("actual token input differs")
    if not all(event["count_matches_top20"] and len(event["documents"]) == 20
               and all(check["full"] for check in event["checks"]) for event in contexts):
        problems.append("full top20 evidence mismatch")
    request_hashes = {
        hashlib.sha256(event["prompt"].encode()).hexdigest() for event in requests
    }
    if not all(event["prompt_sha256"] in request_hashes for event in contexts):
        problems.append("evidence builder prompt not actually sent")
    for event in responses:
        diag = event["diagnostics"]
        if diag.get("input_truncated") or diag.get("output_budget_reduced"):
            problems.append(f"q{event['question_index']} truncation/budget reduction")
        if diag["prompt_tokens"] + diag["requested_max_tokens"] > diag["max_model_len"]:
            problems.append(f"q{event['question_index']} context overflow")
    evidence_requests = [event for event in requests if "\nevidence:\n" in event["prompt"]]
    for event in evidence_requests:
        block = event["prompt"].split("\nevidence:\n", 1)[1].rsplit("\noutput_schema:", 1)[0]
        if re.findall(r"(?m)^\[(E\d+)\]", block) != [f"E{i}" for i in range(1, 21)]:
            problems.append(f"q{event['question_index']} evidence numbering")
    timings = [row["elapsed_sec"] for row in rows]
    summary = {
        "semantic_grading_performed": False,
        "completion": completion,
        "mechanical_audit_passed": not problems,
        "problems": problems,
        "model_calls": len(responses),
        "answer_validation_bypasses": len(bypasses),
        "unexpected_answer_validator_calls": len(unexpected_checks),
        "all_bypassed_conclusions_unchanged": all(event["input_equals_output"] for event in bypasses),
        "actual_input_token_checks": len(engines),
        "retrieval_calls": len(retrievals),
        "evidence_prompts": len(contexts),
        "full_document_occurrences": sum(len(event["documents"]) for event in contexts),
        "finish_reasons": dict(Counter(choice["finish_reason"]
                                      for event in engines for choice in event["choices"])),
        "max_input_tokens": max(event["input_tokens"] for event in engines),
        "max_output_tokens": max(choice["tokens"] for event in engines for choice in event["choices"]),
        "generation_budgets": dict(Counter(event["max_tokens"] for event in engines)),
        "elapsed_sec": {"batch_including_loading": completion["elapsed_sec"],
                        "per_question_mean": statistics.mean(timings),
                        "first_question_including_generator_loading": timings[0],
                        "warm_mean_excluding_first": statistics.mean(timings[1:]),
                        "warm_median_excluding_first": statistics.median(timings[1:]),
                        "min": min(timings), "max": max(timings)},
        "per_question": [],
        "artifact_sha256": {path.name: sha(path) for path in sorted(directory.iterdir())
                            if path.is_file()},
    }
    for index, row in enumerate(rows, 1):
        generated = [event for event in responses if event["question_index"] == index]
        traces = row.get("retrieval_trace", [])
        summary["per_question"].append({
            "q": index, "question_key": f"eval50_{index - 1:03d}",
            "question": row["question"], "answer": row["answer"], "error": row.get("error"),
            "elapsed_sec": row["elapsed_sec"], "rounds": len(traces),
            "model_calls": len(generated),
            "final_action": traces[-1]["conclusion"]["next_action"] if traces else None,
            "grounding_statuses": [trace["conclusion"].get("generation_diagnostics", {})
                                  .get("grounding_status") for trace in traces],
            "finish_reasons": [event["diagnostics"].get("finish_reason") for event in generated],
        })
        if any(trace["conclusion"].get("generation_diagnostics", {}).get("grounding_status")
               not in ("disabled_for_experiment", None) for trace in traces):
            problems.append(f"q{index} non-disabled answer validation status")
    summary["mechanical_audit_passed"] = not problems
    with args.out.open("x") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
