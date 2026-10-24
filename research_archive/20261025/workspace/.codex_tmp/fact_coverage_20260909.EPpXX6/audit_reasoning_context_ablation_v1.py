"""Frozen original4B versus no-hints4B blind audit; no retries or relabels."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha
from verify_empty_quote_candidate_v1 import save, counts
from audit_native8b_v1 import split_requests, audit_messages, audit_validate, prepare, summarize, transport

PREFLIGHT = HERE / "without_reasoning_hints58_blind_audits_preflight_v1.json"
JOURNAL = HERE / "without_reasoning_hints58_blind_audits_v1.progress.jsonl"
DESTINATION = HERE / "without_reasoning_hints58_blind_audits_v1.json"


def main():
    assert not any(p.exists() for p in (PREFLIGHT, JOURNAL, DESTINATION))
    assert counts(HERE / "native8b_audit_tests_v1.xml") == {"tests": 5, "failures": 0, "errors": 0, "skipped": 0}
    postpath = HERE / "without_reasoning_hints58_generation_postflight_v1.json"
    analysispath = HERE / "without_reasoning_hints58_validation_v1.json"
    post = load(postpath)
    assert post["protocol_verified"] and post["states"] == 58
    assert all(sha(Path(p)) == h for p, h in post["artifacts_sha256"].items())
    native = load(analysispath)
    baselinepath = HERE / "token_rejection_new116_replay_v1.json"
    originals = [r for r in load(baselinepath)["results"] if r["renderer"] == "original_prompt_v1"]
    assert len(originals) == len(native["results"]) == 58
    cases, states = prepare({"results": originals + native["results"]})
    oldpath = HERE / "blind_audits_v1.json"
    old_report = load(oldpath)
    old_post = load(HERE / "blind_audits_postflight_v1.json")
    assert old_post["protocol_verified"] and old_post["audit_report_sha256"] == sha(oldpath)
    reused, fresh = split_requests(cases, old_report["cases"], old_report["results"])
    assert len(fresh) <= 116
    for result in reused:
        case = next(c for c in cases if c["id"] == result["id"])
        expected = {"model": "glm-5.3", "temperature": 0, "max_tokens": 16384, "messages": audit_messages(case)}
        assert result["request_without_credentials"] == expected
        assert result["request_sha256"] == transport.digest(expected)
    dependencies = dict(old_report["dependencies_sha256"])
    dependencies.update({str(p): sha(p) for p in [
        Path(__file__), HERE / "audit_fact_coverage_pairs_v1.py", HERE / "audit_native8b_v1.py",
    ]})
    assert all(sha(Path(p)) == h for p, h in dependencies.items())
    frozen = {str(p): sha(p) for p in [
        postpath, analysispath, baselinepath, oldpath, HERE / "blind_audits_postflight_v1.json",
    ]}
    preflight = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "judge_type": "model_not_human", "cases": cases, "states": states,
        "source_artifacts_sha256": frozen, "dependencies_sha256": dependencies,
        "reused_result_ids": [r["id"] for r in reused], "new_request_ids": [c["id"] for c in fresh],
        "new_request_count": len(fresh), "reused_errors": sum(r["status"] != "ok" for r in reused),
        "protocol": [
            "Original4B58 versus no-reasoning-context4B58; all nonanswers/errors retained.",
            "Identical frozen coverage-v3/content-v2 rubrics and glm-5.3/16384 budget.",
            "Reuse identical provider-visible inputs including errors; no selective retries or output repair.",
            "No alternate answers/variant identity/prior verdicts/source-review notes are exposed to the grader.",
            "Two concurrent calls only; model grades are not human gold or a production promotion gate.",
        ],
    }
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    assert key or not fresh
    save(PREFLIGHT, preflight)
    print(json.dumps({"states": len(states), "unique_cases": len(cases), "reused": len(reused),
                      "new_requests": len(fresh)}), flush=True)
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
    save(DESTINATION, {
        **preflight, "preflight_sha256": sha(PREFLIGHT),
        "results": [by_id[c["id"]] for c in cases], "labeled_states": labeled, "summary": summary,
        "unique_statuses": dict(Counter(r["status"] for r in results)),
        "new_statuses": dict(Counter(by_id[c["id"]]["status"] for c in fresh)),
    })
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
