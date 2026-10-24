"""Retry only failed claim audits, preserving the full seven-claim denominator."""
import hashlib
import json
import os
from pathlib import Path

from probe_claim_entailment_v1 import call, messages, validate


run = Path(__file__).resolve().parent
source = run / "cited_claim_entailment_probe7_v1.json"
output = run / "cited_claim_entailment_probe7_retry1.json"
journal_path = output.with_suffix(".progress.jsonl")
if output.exists() or journal_path.exists():
    raise ValueError("Existing retry artifacts")
prior = json.loads(source.read_text())
old = {r["id"]: r for r in prior["results"]}
cases = prior["cases"]
assert len(cases) == len(old) == 7 and {c["id"] for c in cases} == set(old)
key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
if not key:
    raise ValueError("provider key unavailable")
results, requested = [], []
with journal_path.open("x") as journal:
    for case in cases:
        previous = old[case["id"]]
        assert previous["request_without_credentials"]["messages"] == messages(case)
        if previous["status"] == "ok":
            validate(previous["judgement"], case)
            result = {**previous, "reused_original_success": True}
        else:
            result = call(case, key)
            assert result["request_sha256"] == previous["request_sha256"]
            requested.append(case["id"])
        results.append(result)
        journal.write(json.dumps(result, ensure_ascii=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())
        print(json.dumps({"id": result["id"], "status": result["status"],
                          "reused": result.get("reused_original_success", False),
                          "error": result.get("error")}), flush=True)
report = {
    **prior, "results": results, "retry_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "retry_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "retried_ids": requested, "api_requests": len(requested), "training_ready": False,
}
with output.open("x") as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
