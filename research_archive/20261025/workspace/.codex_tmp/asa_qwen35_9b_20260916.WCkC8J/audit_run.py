"""Offline mechanical audit, never a runtime answer judge."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert not args.out.exists()
    directory = args.directory
    rows = read(directory / "results.jsonl")
    calls = read(directory / "calls.jsonl")
    audits = read(directory / "audit.jsonl")
    completion = json.loads((directory / "completion.json").read_text())
    requests = [x for x in calls if x["event"] == "request"]
    responses = [x for x in calls if x["event"] == "response"]
    engines = [x for x in calls if x["event"] == "engine_result"]
    contexts = [x for x in audits if x["event"] == "prompt_evidence"]
    bypasses = [x for x in audits if x["event"] == "answer_validation_bypassed"]
    hypothesis_checks = [x for x in audits if x["event"] == "hypothesis_schema"]
    hypothesis_unwraps = [x for x in audits if x["event"] == "hypothesis_schema_unwrapped"]
    bad = []
    if not (len(rows) == completion["count"] == completion["expected_count"]
            and completion["exit_code"] == 0 and completion["runtime_errors"] == 0
            and completion["questions_match"]):
        bad.append("batch not fully complete")
    with (directory / "results.jsonl").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != completion["results_sha256"]:
            bad.append("result checksum")
    if len(requests) != len(responses) or len(responses) != len(engines):
        bad.append("generation event mismatch")
    if any(x["event"] == "error" for x in calls):
        bad.append("generation errors")
    if any(x["event"] == "unexpected_answer_validator" for x in audits):
        bad.append("unexpected validation")
    if not bypasses or not all(x["input_equals_output"] for x in bypasses):
        bad.append("bypass absent or not identity")
    if any(not x["valid"] for x in hypothesis_checks):
        bad.append("hypothesis schema fallback")
    hashes = {hashlib.sha256(x["original_prompt"].encode()).hexdigest() for x in requests}
    for context in contexts:
        if len(context["documents"]) != 20 or not all(c["full"] for c in context["checks"]):
            bad.append(f"q{context['q']} full top20")
        if context["prompt_sha256"] not in hashes:
            bad.append(f"q{context['q']} prompt not sent")
    if not all(x["actual_input_matches"] for x in engines):
        bad.append("input token mismatch")
    for x in responses:
        d = x["diagnostics"]
        if d["input_truncated"] or d["output_budget_reduced"]:
            bad.append(f"q{x['q']} truncated")
        if d["prompt_tokens"] + d["requested_max_tokens"] > d["max_model_len"]:
            bad.append(f"q{x['q']} context overflow")
    for x in requests:
        if not x["native_chat_template"] or x["thinking"] is not False:
            bad.append("incorrect chat mode")
        if "\nevidence:\n" in x["prompt"]:
            old = x["original_prompt"].split("\nevidence:\n", 1)[1].rsplit("\noutput_schema:", 1)[0]
            new = x["prompt"].split("\nevidence:\n", 1)[1].rsplit("\noutput_schema:", 1)[0]
            if old != new:
                bad.append("evidence changed by chat conversion")
    result = {
        "mechanical_audit_passed": not bad, "problems": bad, "completion": completion,
        "model_calls": len(engines), "answer_validation_bypasses": len(bypasses),
        "unexpected_answer_validator_calls": sum(x["event"] == "unexpected_answer_validator" for x in audits),
        "evidence_prompts": len(contexts), "full_documents": sum(len(x["documents"]) for x in contexts),
        "hypothesis_schema_checks": len(hypothesis_checks),
        "hypothesis_schema_valid": sum(x["valid"] for x in hypothesis_checks),
        "hypothesis_schema_unwrapped": len(hypothesis_unwraps),
        "max_input_tokens": max(x["input_tokens"] for x in engines),
        "finish_reasons": dict(Counter(c["finish_reason"] for x in engines for c in x["choices"])),
        "per_question_mean_sec": statistics.mean(r["elapsed_sec"] for r in rows),
        "warm_mean_sec": statistics.mean(r["elapsed_sec"] for r in rows[1:]) if len(rows) > 1 else None,
        "semantic_grading_performed": False,
        "per_question": [{"q": i, "question": r["question"], "answer": r["answer"],
                          "elapsed_sec": r["elapsed_sec"], "error": r.get("error"),
                          "rounds": len(r.get("retrieval_trace", [])),
                          "action": r["retrieval_trace"][-1]["conclusion"]["next_action"]}
                         for i, r in enumerate(rows, 1)],
    }
    with args.out.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(int(bool(bad)))


if __name__ == "__main__":
    main()
