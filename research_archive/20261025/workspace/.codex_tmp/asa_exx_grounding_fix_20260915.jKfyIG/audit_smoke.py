"""Mechanically audit fresh targeted inference without assigning semantic scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / "manifest.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    assert completion["exit_code"] == 0
    assert completion["questions_match"] and completion["count"] == 8
    assert hashlib.sha256((root / "results.jsonl").read_bytes()).hexdigest() == completion["results_sha256"]
    results = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    responses = [item for item in calls if item["event"] == "response"]
    requests = [item for item in calls if item["event"] == "request"]
    assert len(responses) == len(requests)
    full_inputs = []
    for request in requests:
        prompt = request["prompt"]
        if "\nevidence:\n" not in prompt:
            continue
        evidence = prompt.split("\nevidence:\n", 1)[1].rsplit("\noutput_schema:", 1)[0]
        labels = re.findall(r"(?m)^\[(E\d+)\]", evidence)
        assert labels == [f"E{i}" for i in range(1, 21)]
        full_inputs.append({"q": request["q"], "call": request["call"], "count": len(labels)})
    for response in responses:
        diag = response["diagnostics"]
        assert not diag.get("input_truncated") and not diag.get("output_budget_reduced")
        assert diag["prompt_tokens"] + diag["requested_max_tokens"] <= diag["max_model_len"]
    question_rows = []
    for q, result in zip(manifest["qids"], results):
        trace = result["retrieval_trace"]
        last = trace[-1]["conclusion"]
        question_rows.append({
            "q": q, "question": result["question"], "answer": result["answer"],
            "action": last["next_action"],
            "rounds": len(trace), "elapsed_sec": result["elapsed_sec"],
            "repair": [record for step in trace for record in step.get("grounding_repair_attempts", [])],
            "last_diagnostics": last.get("generation_diagnostics", {}),
        })
    summary = {
        "kind": manifest["kind"], "completion": completion, "questions": question_rows,
        "calls": len(responses), "evidence_prompts": len(full_inputs),
        "finish_reasons": dict(Counter(item["diagnostics"]["finish_reason"] for item in responses)),
        "maximum_input_tokens": max(item["diagnostics"]["prompt_tokens"] for item in responses),
        "mean_latency": sum(item["elapsed_sec"] for item in results) / len(results),
        "actions": dict(Counter(row["action"] for row in question_rows)),
        "repair_statuses": dict(Counter(record["status"] for row in question_rows for record in row["repair"])),
    }
    with args.out.open("x") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "questions"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
