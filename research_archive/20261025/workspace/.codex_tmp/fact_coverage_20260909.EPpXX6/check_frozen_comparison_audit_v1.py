"""Independent request/schema/source verification for either new comparison."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha
from verify_empty_quote_candidate_v1 import save
from audit_native8b_v1 import audit_messages, audit_validate, prepare, summarize, transport

PREFIXES = {
    "native8b": ("qwen3_8b", "qwen3_8b_validation_v1.json"),
    "context": ("without_reasoning_hints58", "without_reasoning_hints58_validation_v1.json"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=PREFIXES)
    args = parser.parse_args()
    prefix, analysisname = PREFIXES[args.variant]
    reportpath = HERE / f"{prefix}_blind_audits_v1.json"
    prepath = HERE / f"{prefix}_blind_audits_preflight_v1.json"
    journalpath = HERE / f"{prefix}_blind_audits_v1.progress.jsonl"
    destination = HERE / f"{prefix}_blind_audits_postflight_v1.json"
    assert not destination.exists()
    report, pre = load(reportpath), load(prepath)
    assert report["preflight_sha256"] == sha(prepath)
    assert all(sha(Path(p)) == h for p, h in {
        **pre["source_artifacts_sha256"], **pre["dependencies_sha256"],
    }.items())
    original = [r for r in load(HERE / "token_rejection_new116_replay_v1.json")["results"]
                if r["renderer"] == "original_prompt_v1"]
    cases, states = prepare({"results": original + load(HERE / analysisname)["results"]})
    assert cases == pre["cases"] == report["cases"] and states == pre["states"] == report["states"]
    by_id = {r["id"]: r for r in report["results"]}
    old_by_id = {r["id"]: r for r in load(HERE / "blind_audits_v1.json")["results"]}
    assert len(by_id) == len(report["results"]) == len(cases)
    assert set(pre["reused_result_ids"]).isdisjoint(pre["new_request_ids"])
    assert set(by_id) == set(pre["reused_result_ids"]) | set(pre["new_request_ids"])
    assert all(by_id[rid] == old_by_id[rid] for rid in pre["reused_result_ids"])
    journal = [json.loads(line) for line in journalpath.read_text().splitlines()]
    assert len(journal) == len(pre["new_request_ids"]) == len({r["id"] for r in journal})
    assert set(r["id"] for r in journal) == set(pre["new_request_ids"])
    assert all(r == by_id[r["id"]] for r in journal)
    failures = []
    for case in cases:
        result = by_id[case["id"]]
        expected = {"model": "glm-5.3", "temperature": 0, "max_tokens": 16384,
                    "messages": audit_messages(case)}
        assert result["request_without_credentials"] == expected
        assert result["request_sha256"] == transport.digest(expected)
        if result["status"] == "ok":
            assert result["response_model"] == "glm-5.3" and result["finish_reason"] == "stop"
            assert audit_validate(json.loads(result["raw_response"]), case) == result["judgement"]
        else:
            reproduced = False
            if result.get("response_model") == "glm-5.3" and result.get("finish_reason") == "stop":
                try:
                    audit_validate(json.loads(result["raw_response"]), case)
                except Exception as exc:
                    assert result["error"] == f"{type(exc).__name__}: {exc}"
                    reproduced = True
                else:
                    raise AssertionError("stored failure did not reproduce")
            failures.append({
                "id": case["id"], "new": case["id"] in pre["new_request_ids"],
                "error": result["error"], "response_model": result.get("response_model"),
                "finish_reason": result.get("finish_reason"), "usage": result.get("usage"),
                "validation_failure_reproduced": reproduced,
            })
    labeled, summary = summarize(states, report["results"])
    assert labeled == report["labeled_states"] and summary == report["summary"]
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_verified": True,
        "evaluation_only": True, "training_ready": False, "audit_report_sha256": sha(reportpath),
        "all116_states_retained": True, "old_results_reused_exactly_including_errors": True,
        "new_calls": len(journal), "new_statuses": dict(Counter(r["status"] for r in journal)),
        "summary": summary, "failures": failures, "checker_sha256": sha(Path(__file__)),
        "certified_answer_accuracy": None,
    })
    print(json.dumps({"variant": args.variant, "protocol_verified": True,
                      "new_calls": len(journal), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
