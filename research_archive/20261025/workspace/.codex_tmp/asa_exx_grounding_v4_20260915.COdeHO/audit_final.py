import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.directory / "results.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (args.directory / "calls.jsonl").read_text().splitlines()]
    completion = json.loads((args.directory / "completion.json").read_text())
    manifest = json.loads((args.directory / "manifest.json").read_text())
    assert len(rows) == completion["count"] == 2 and manifest["qids"] == [13, 19]
    assert completion["runtime_errors"] == 0 and completion["exit_code"] == 0 and completion["questions_match"]
    assert hashlib.sha256((args.directory / "results.jsonl").read_bytes()).hexdigest() == completion["results_sha256"]
    responses = [r for r in events if r["event"] == "response"]
    requests = [r for r in events if r["event"] == "request"]
    assert len(requests) == len(responses)
    body_count = 0
    for request in requests:
        prompt = request["prompt"]
        assert "repair_task:" not in prompt
        if "\nevidence:\n" in prompt:
            brief = prompt.split("\nevidence:\n", 1)[1].rsplit("\noutput_schema:", 1)[0]
            assert re.findall(r"(?m)^\[(E\d+)\]", brief) == [f"E{i}" for i in range(1, 21)]
            body_count += 20
    for response in responses:
        diag = response["diagnostics"]
        assert not diag["input_truncated"] and not diag["output_budget_reduced"]
        assert diag["prompt_tokens"] + diag["requested_max_tokens"] <= diag["max_model_len"]
        assert isinstance(json.loads(response["text"]), dict)
    report = {
        "kind": "final_two_question_regression_not_accuracy_estimate",
        "completion": completion, "calls": len(responses), "evidence_items": body_count,
        "finish_reasons": dict(Counter(r["diagnostics"]["finish_reason"] for r in responses)),
        "max_input_tokens": max(r["diagnostics"]["prompt_tokens"] for r in responses),
        "max_output_tokens": max(r["diagnostics"]["generated_tokens"] for r in responses),
        "per_question": [
            {"q": q, "answer": row["answer"], "elapsed_sec": row["elapsed_sec"],
             "rounds": len(row["retrieval_trace"]),
             "action": row["retrieval_trace"][-1]["conclusion"]["next_action"]}
            for q, row in zip([13, 19], rows)
        ],
    }
    with args.out.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
