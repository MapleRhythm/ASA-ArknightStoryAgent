"""Frozen three-arm audit; no new request for any identical historical input."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot, counts
from audit_native8b_v1 import split_requests
from audit_fact_coverage_pairs_v1 import (
    audit_messages, audit_validate, prepare, summarize, transport,
)

PREFIX = "quote_pointer8b58"
PREFLIGHT = HERE / f"{PREFIX}_blind_audits_preflight_v1.json"
JOURNAL = HERE / f"{PREFIX}_blind_audits_v1.progress.jsonl"
DESTINATION = HERE / f"{PREFIX}_blind_audits_v1.json"
CONTROL = HERE / "qwen3_8b_blind_audits_v1.json"


def combine_triplet(original, native, pointer, size=58):
    arms = [original, native, pointer]
    assert all(len(arm) == size for arm in arms), "missing evaluation state"
    for rows in zip(*arms):
        for field in ("source_id", "question", "evidence_brief",
                      "current_round", "max_retrieval_rounds"):
            assert all(row[field] == rows[0][field] for row in rows), field
    results = [row for arm in arms for row in arm]
    assert len({row["id"] for row in results}) == size * 3, "duplicate state id"
    return {"results": results}


def analysis():
    original = [r for r in load(HERE / "token_rejection_new116_replay_v1.json")["results"]
                if r["renderer"] == "original_prompt_v1"]
    native = load(HERE / "qwen3_8b_validation_v1.json")["results"]
    pointer = load(HERE / "quote_pointer8b58_validation_v1.json")["results"]
    return combine_triplet(original, native, pointer)


def expected_request(case):
    return {"model": "glm-5.3", "temperature": 0, "max_tokens": 16384,
            "messages": audit_messages(case)}


def assert_request(result, case):
    expected = expected_request(case)
    assert result["request_without_credentials"] == expected
    assert result["request_sha256"] == transport.digest(expected)
    payload = json.loads(expected["messages"][1]["content"])
    assert set(payload) == (
        {"question", "shown_answer"} if case["audit_kind"] == "coverage" else
        {"question", "round", "max_rounds", "evidence", "candidate"}
    ), "unexpected provider-visible fields"


def main():
    assert not any(p.exists() for p in (PREFLIGHT, JOURNAL, DESTINATION))
    assert counts(HERE / "quote_pointer_audit_tests_v1.xml") == {
        "tests": 13, "failures": 0, "errors": 0, "skipped": 0}
    post = load(HERE / "quote_pointer8b58_generation_postflight_v1.json")
    assert post["protocol_verified"] and post["states"] == 58
    assert all(sha(Path(p)) == h for p, h in post["artifacts_sha256"].items())
    current_source = snapshot(SOURCE)
    assert current_source == load(HERE / "checkpoint_v20_state.json")["runtime_source_sha256"]
    cases, states = prepare(analysis())
    old_report = load(CONTROL)
    old_post = load(HERE / "qwen3_8b_blind_audits_postflight_v1.json")
    assert old_post["protocol_verified"] and old_post["audit_report_sha256"] == sha(CONTROL)
    reused, fresh = split_requests(cases, old_report["cases"], old_report["results"])
    assert len(reused) == len(old_report["results"])
    assert len(fresh) <= 116
    by_case = {c["id"]: c for c in cases}
    for result in reused:
        assert_request(result, by_case[result["id"]])
    dependencies = dict(old_report["dependencies_sha256"])
    dependencies.update({str(p): sha(p) for p in [
        Path(__file__), HERE / "audit_native8b_v1.py",
        HERE / "audit_fact_coverage_pairs_v1.py",
        HERE / "test_quote_pointer_audit_v1.py",
    ]})
    assert all(sha(Path(p)) == h for p, h in dependencies.items())
    frozen = {str(p): sha(p) for p in [
        CONTROL, HERE / "qwen3_8b_blind_audits_postflight_v1.json",
        HERE / "qwen3_8b_validation_v1.json", HERE / "token_rejection_new116_replay_v1.json",
        HERE / "quote_pointer8b58_generation_postflight_v1.json",
        HERE / "quote_pointer8b58_validation_v1.json", HERE / "quote_pointer_audit_tests_v1.xml",
        HERE / "checkpoint_v20_state.json",
    ]}
    preflight = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "judge_type": "model_not_human",
        "cases": cases, "states": states, "source_artifacts_sha256": frozen,
        "dependencies_sha256": dependencies, "runtime_source_sha256": current_source,
        "reused_result_ids": [r["id"] for r in reused],
        "new_request_ids": [c["id"] for c in fresh],
        "reused_errors": sum(r["status"] != "ok" for r in reused),
        "new_request_count": len(fresh),
        "protocol": [
            "Original4B58/native8B58/pointer8B58: retain all174 states, no-answer actions and all errors.",
            "Frozen coverage-v3/content-v2, glm-5.3, temperature0, max_tokens16384; two concurrent calls.",
            "Same provider messages reuse old results INCLUDING failures; no retry or output repair.",
            "Audit the displayed grounded answer, not model final_answer or pointer payload.",
            "Judges never receive variant/model identity, other answers, prior verdicts, gold or source reviews.",
            "Pointer raw output remains intact; the expanded quotes were bound by code, not generated.",
            "These model opinions are not human gold, E2E accuracy or sufficient promotion evidence.",
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
            futures = [pool.submit(transport.call, c, key)
                       for c in sorted(fresh, key=lambda c: transport.digest(audit_messages(c)))]
            for future in as_completed(futures):
                result = future.result()
                assert_request(result, by_case[result["id"]])
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({"completed_new": len(results) - len(reused), "total_new": len(fresh),
                                  "status": result["status"], "error": result.get("error")}), flush=True)
    assert all(sha(Path(p)) == h for p, h in {**frozen, **dependencies}.items())
    assert snapshot(SOURCE) == current_source
    by_id = {r["id"]: r for r in results}
    assert len(by_id) == len(results) == len(cases) and set(by_id) == set(by_case)
    labeled, summary = summarize(states, results)
    controls = [r for r in labeled if r["renderer"] != "quote_pointer8b_v1"]
    assert controls == old_report["labeled_states"]
    report = {**preflight, "preflight_sha256": sha(PREFLIGHT),
              "results": [by_id[c["id"]] for c in cases], "labeled_states": labeled,
              "summary": summary, "unique_statuses": dict(Counter(r["status"] for r in results)),
              "new_statuses": dict(Counter(by_id[c["id"]]["status"] for c in fresh))}
    save(DESTINATION, report)
    print(json.dumps({"summary": summary, "new_statuses": report["new_statuses"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
