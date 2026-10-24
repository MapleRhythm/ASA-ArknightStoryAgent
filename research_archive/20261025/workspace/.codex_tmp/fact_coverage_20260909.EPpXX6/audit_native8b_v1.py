"""Same frozen blind rubrics; reuse identical old calls, including old errors."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha
from verify_empty_quote_candidate_v1 import counts, save
from audit_fact_coverage_pairs_v1 import (
    audit_messages, audit_validate, prepare, summarize, transport,
)

PREFLIGHT = HERE / "qwen3_8b_blind_audits_preflight_v1.json"
JOURNAL = HERE / "qwen3_8b_blind_audits_v1.progress.jsonl"
DESTINATION = HERE / "qwen3_8b_blind_audits_v1.json"


def split_requests(cases, old_cases, old_results, message_builder=audit_messages):
    assert len(cases) == len({c["id"] for c in cases})
    old_by_id = {c["id"]: c for c in old_cases}
    results_by_id = {r["id"]: r for r in old_results}
    assert len(old_by_id) == len(old_cases) and len(results_by_id) == len(old_results)
    assert set(results_by_id) == set(old_by_id)
    reused, fresh = [], []
    for case in cases:
        if case["id"] in old_by_id:
            assert message_builder(case) == message_builder(old_by_id[case["id"]]), "different provider input"
            reused.append(results_by_id[case["id"]])
        else:
            fresh.append(case)
    return reused, fresh


def main():
    assert not any(p.exists() for p in (PREFLIGHT, JOURNAL, DESTINATION))
    assert counts(HERE / "native8b_audit_tests_v1.xml") == {"tests": 5, "failures": 0, "errors": 0, "skipped": 0}
    post = load(HERE / "qwen3_8b_generation_postflight_v1.json")
    assert post["protocol_verified"] and post["states"] == 58
    assert all(sha(Path(p)) == h for p, h in post["artifacts_sha256"].items())
    native = load(HERE / "qwen3_8b_validation_v1.json")
    old_analysis = load(HERE / "token_rejection_new116_replay_v1.json")
    originals = [r for r in old_analysis["results"] if r["renderer"] == "original_prompt_v1"]
    assert len(originals) == len(native["results"]) == 58
    combined = {"results": originals + native["results"]}
    cases, states = prepare(combined)
    old_report = load(HERE / "blind_audits_v1.json")
    old_post = load(HERE / "blind_audits_postflight_v1.json")
    assert old_post["protocol_verified"]
    assert old_post["audit_report_sha256"] == sha(HERE / "blind_audits_v1.json")
    reused, fresh = split_requests(cases, old_report["cases"], old_report["results"])
    assert len(fresh) <= 116
    for result in reused:
        case = next(c for c in cases if c["id"] == result["id"])
        expected = {"model": "glm-5.3", "temperature": 0, "max_tokens": 16384, "messages": audit_messages(case)}
        assert result["request_without_credentials"] == expected
        assert result["request_sha256"] == transport.digest(expected)
    dependencies = dict(old_report["dependencies_sha256"])
    dependencies.update({str(p): sha(p) for p in [
        Path(__file__), HERE / "audit_fact_coverage_pairs_v1.py", HERE / "test_native8b_audit_v1.py",
    ]})
    assert all(sha(Path(p)) == h for p, h in dependencies.items())
    frozen = {str(p): sha(p) for p in [
        HERE / "qwen3_8b_generation_postflight_v1.json", HERE / "qwen3_8b_validation_v1.json",
        HERE / "token_rejection_new116_replay_v1.json", HERE / "blind_audits_v1.json",
        HERE / "blind_audits_postflight_v1.json",
    ]}
    preflight = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "judge_type": "model_not_human", "cases": cases, "states": states,
        "source_artifacts_sha256": frozen, "dependencies_sha256": dependencies,
        "reused_result_ids": [r["id"] for r in reused], "new_request_ids": [c["id"] for c in fresh],
        "reused_errors": sum(r["status"] != "ok" for r in reused),
        "new_request_count": len(fresh),
        "protocol": [
            "Compare original4B58 and native8B58; every non-direct and pipeline failure remains in its58 denominator.",
            "Exact same frozen coverage-v3/content-v2 rubrics and glm-5.3/max_tokens16384 as paired4B.",
            "Reuse only identical provider-visible messages; retain old errors, never cherry-pick retries.",
            "New calls never see other answers, model/variant identity, prior verdicts or source-review notes.",
            "One call per unseen input with two concurrent requests; no retry or malformed-output repair.",
            "Model judges have documented causal/identity/consistency failures; not human gold or promotion evidence alone.",
        ],
    }
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    assert key or not fresh, "Provider credential unavailable"
    save(PREFLIGHT, preflight)
    print(json.dumps({"states": len(states), "unique_cases": len(cases),
                      "reused": len(reused), "new_requests": len(fresh)}), flush=True)
    transport.messages, transport.validate = audit_messages, audit_validate
    results = list(reused)
    with JOURNAL.open("x", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(transport.call, case, key)
                       for case in sorted(fresh, key=lambda c: transport.digest(audit_messages(c)))]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({"completed_new": len(results) - len(reused), "total_new": len(fresh),
                                  "status": result["status"], "error": result.get("error")}), flush=True)
    assert all(sha(Path(p)) == h for p, h in {**frozen, **dependencies}.items())
    by_id = {r["id"]: r for r in results}
    assert len(by_id) == len(results) == len(cases)
    assert set(by_id) == {c["id"] for c in cases}
    labeled, summary = summarize(states, results)
    old_labeled = {r["id"]: r for r in old_report["labeled_states"]}
    assert all(r == old_labeled[r["id"]] for r in labeled if r["renderer"] == "original_prompt_v1")
    report = {**preflight, "preflight_sha256": sha(PREFLIGHT),
              "results": [by_id[c["id"]] for c in cases], "labeled_states": labeled, "summary": summary,
              "unique_statuses": dict(Counter(r["status"] for r in results)),
              "new_statuses": dict(Counter(by_id[c["id"]]["status"] for c in fresh))}
    save(DESTINATION, report)
    print(json.dumps({"summary": summary, "new_statuses": report["new_statuses"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
