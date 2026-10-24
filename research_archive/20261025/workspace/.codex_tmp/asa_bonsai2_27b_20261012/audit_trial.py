"""Mechanical audit only; semantic review remains a separate offline task."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from bonsai_api import messages_from_prompt


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit_reader(path):
    rows = read(path / "results.jsonl")
    details = []
    for r in rows:
        messages = messages_from_prompt(r["original_prompt"])
        response = r.get("api_response", {})
        choices = response.get("choices", [])
        output = choices[0].get("message", {}) if choices else {}
        evidence = messages[1]["content"].split("\nevidence:\n", 1)[1].rsplit("\noutput_schema:", 1)[0]
        identifiers = re.findall(r"(?m)^\[(E\d+)\]", evidence)
        details.append({
            "q": r["q"], "error": r.get("error"),
            "input_unchanged": all(m["content"] in r.get("native_prompt", "") for m in messages),
            "evidence20": identifiers == [f"E{i}" for i in range(1, 21)],
            "input_token_matches": response.get("usage", {}).get("prompt_tokens") == r.get("input_tokens"),
            "finish_reason": r.get("finish_reason"),
            "input_tokens": r.get("input_tokens"), "usage": response.get("usage"),
            "reasoning_chars": len(output.get("reasoning_content") or ""),
        })
    return {
        "kind": "reader_only", "count": len(rows), "details": details,
        "all_passed": len(rows) == 10 and all(
            not d["error"] and d["input_unchanged"] and d["evidence20"]
            and d["input_token_matches"] and d["finish_reason"] == "stop"
            for d in details),
        "results_sha256": sha(path / "results.jsonl"),
    }


def audit_pipeline(path):
    rows, events, calls = [read(path / name) for name in ("results.jsonl", "audit.jsonl", "calls.jsonl")]
    prompts = [r for r in events if r["event"] == "prompt_evidence"]
    checks = [r for r in events if r["event"] == "hypothesis_schema"]
    responses = [r for r in calls if r["event"] == "response"]
    counts = Counter(r["event"] for r in events)
    passed = (
        len(rows) == 5 and not any(r.get("error") for r in rows)
        and len(checks) == 5 and all(r["valid"] for r in checks)
        and len(prompts) == 5 and all(
            len(r["documents"]) == 20 and all(c["full"] for c in r["checks"]) for r in prompts)
        and counts["answer_validation_bypassed"] == 5
        and counts["unexpected_answer_validator"] == 0
        and len(responses) == 10 and all(r["finish_reason"] == "stop" for r in responses)
    )
    return {
        "kind": "live_pipeline", "count": len(rows), "all_passed": passed,
        "audit_events": dict(counts), "generation_calls": len(responses),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in responses)),
        "evidence_documents": sum(len(r["documents"]) for r in prompts),
        "all_full_evidence": all(all(c["full"] for c in r["checks"]) for r in prompts),
        "hypothesis_schema_success": sum(r["valid"] for r in checks),
        "results_sha256": sha(path / "results.jsonl"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    reports = {}
    for name in ("reader10_nothink", "reader10_think"):
        path = args.stage / name
        if (path / "completion.json").exists():
            reports[name] = audit_reader(path)
    path = args.stage / "pilot5"
    if (path / "completion.json").exists():
        reports["pilot5"] = audit_pipeline(path)
    report = {"runs": reports, "all_completed_runs_passed": all(r["all_passed"] for r in reports.values())}
    (args.stage / "mechanical_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
