"""Independent postflight: all174 states and actual frozen provider requests."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot
from audit_quote_pointer8b_v1 import (
    PREFLIGHT, JOURNAL, DESTINATION, CONTROL, analysis, assert_request,
    audit_messages, audit_validate, prepare, summarize,
)


def main():
    destination = HERE / "quote_pointer8b58_blind_audits_postflight_v1.json"
    assert not destination.exists()
    report, pre, control = load(DESTINATION), load(PREFLIGHT), load(CONTROL)
    assert report["preflight_sha256"] == sha(PREFLIGHT)
    assert all(sha(Path(p)) == h for p, h in {
        **pre["source_artifacts_sha256"], **pre["dependencies_sha256"],
    }.items())
    assert snapshot(SOURCE) == pre["runtime_source_sha256"]
    cases, states = prepare(analysis())
    assert cases == pre["cases"] == report["cases"]
    assert states == pre["states"] == report["states"] and len(states) == 174
    by_id = {r["id"]: r for r in report["results"]}
    old_by_id = {r["id"]: r for r in control["results"]}
    old_cases = {c["id"]: c for c in control["cases"]}
    assert len(by_id) == len(report["results"]) == len(cases)
    reused, fresh = set(pre["reused_result_ids"]), set(pre["new_request_ids"])
    assert reused == set(old_by_id) and reused.isdisjoint(fresh)
    assert set(by_id) == reused | fresh
    assert all(by_id[rid] == old_by_id[rid] for rid in reused)
    journal = [json.loads(line) for line in JOURNAL.read_text().splitlines()]
    assert len(journal) == len(fresh) == len({r["id"] for r in journal})
    assert {r["id"] for r in journal} == fresh and all(r == by_id[r["id"]] for r in journal)
    failures = []
    for case in cases:
        result = by_id[case["id"]]
        assert_request(result, case)
        if case["id"] in reused:
            assert audit_messages(case) == audit_messages(old_cases[case["id"]])
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
                    raise AssertionError("stored audit failure did not reproduce")
            failures.append({
                "id": case["id"], "new": case["id"] in fresh, "error": result["error"],
                "response_model": result.get("response_model"), "finish_reason": result.get("finish_reason"),
                "usage": result.get("usage"), "validation_failure_reproduced": reproduced,
            })
    labeled, summary = summarize(states, report["results"])
    assert labeled == report["labeled_states"] and summary == report["summary"]
    assert [r for r in labeled if r["renderer"] != "quote_pointer8b_v1"] == control["labeled_states"]
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_verified": True,
        "evaluation_only": True, "training_ready": False, "audit_report_sha256": sha(DESTINATION),
        "all174_states_retained": True, "controls_unchanged_including_errors": True,
        "new_calls": len(journal), "new_statuses": dict(Counter(r["status"] for r in journal)),
        "summary": summary, "failures": failures, "checker_sha256": sha(Path(__file__)),
        "certified_answer_accuracy": None, "source_review_sent": False,
    })
    print(json.dumps({"protocol_verified": True, "new_calls": len(journal), "summary": summary},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
