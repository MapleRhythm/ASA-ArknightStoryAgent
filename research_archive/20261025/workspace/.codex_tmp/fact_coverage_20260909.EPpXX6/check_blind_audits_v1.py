"""Verify all126 frozen requests and failures; derive paired automatic opinions."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha
from verify_empty_quote_candidate_v1 import save
from audit_fact_coverage_pairs_v1 import (
    audit_messages, audit_validate, prepare, summarize, transport,
)


def main():
    path = HERE / "blind_audits_v1.json"
    destination = HERE / "blind_audits_postflight_v1.json"
    assert not destination.exists()
    report = load(path)
    assert sha(HERE / "validation_v1.json") == report["source_sha256"]
    assert sha(HERE / "fact_coverage_pairs116_v1.matrix.json") == report["matrix_sha256"]
    assert sha(HERE / "audit_fact_coverage_pairs_v1.py") == report["driver_sha256"]
    assert sha(HERE / "blind_audits_preflight_v1.json") == report["preflight_sha256"]
    assert all(sha(Path(p)) == digest for p, digest in report["dependencies_sha256"].items())
    cases, states = prepare(load(HERE / "validation_v1.json"))
    assert len(cases) == 126 and len(states) == 116
    assert cases == report["cases"] and states == report["states"]
    # Both newly applied guards leave all direct-answer audit inputs unchanged.
    current_cases, current_states = prepare(load(HERE / "token_rejection_new116_replay_v1.json"))
    assert current_cases == cases and current_states == states
    results = report["results"]
    by_id = {r["id"]: r for r in results}
    assert len(results) == len(by_id) == 126 and set(by_id) == {c["id"] for c in cases}
    failures = []
    for case in cases:
        result = by_id[case["id"]]
        expected = {
            "model": "glm-5.3", "temperature": 0, "max_tokens": 16384,
            "messages": audit_messages(case),
        }
        assert result["request_without_credentials"] == expected
        assert result["request_sha256"] == transport.digest(expected)
        user_payload = json.loads(expected["messages"][1]["content"])
        assert set(user_payload) == (
            {"question", "shown_answer"} if case["audit_kind"] == "coverage" else
            {"question", "round", "max_rounds", "evidence", "candidate"}
        )
        if result["status"] == "ok":
            assert result["response_model"] == "glm-5.3" and result["finish_reason"] == "stop"
            assert audit_validate(json.loads(result["raw_response"]), case) == result["judgement"]
        else:
            failure = {"id": case["id"], "error": result["error"], "reproduced_validation_failure": False}
            if result.get("response_model") == "glm-5.3" and result.get("finish_reason") == "stop":
                try:
                    audit_validate(json.loads(result["raw_response"]), case)
                except Exception as exc:
                    assert result["error"] == f"{type(exc).__name__}: {exc}"
                    failure["reproduced_validation_failure"] = True
                else:
                    raise AssertionError("Stored validation error did not reproduce")
            failures.append(failure)
    labeled, summary = summarize(states, results)
    assert labeled == report["labeled_states"] and summary == report["summary"]
    matrix = load(HERE / "fact_coverage_pairs116_v1.matrix.json")
    labeled_by_id = {r["id"]: r for r in labeled}
    pairs = []
    for index, selected in enumerate(matrix["selection"]):
        variants = {
            row["renderer"]: labeled_by_id[row["id"]]
            for row in matrix["rows"][index * 2:index * 2 + 2]
        }
        pair = {
            "parent_id": selected["parent_record_id"], "question": selected["question"],
            "prior_six_question_development_set": selected["prior_six_question_development_set"],
        }
        for side, variant in (("original", "original_prompt_v1"), ("candidate", "fact_coverage_prompt_v1")):
            for kind in ("coverage", "semantics"):
                pair[side + "_" + kind] = variants[variant][kind + "_label"]
        pairs.append(pair)
    assert len(pairs) == 58
    transitions = {
        kind: dict(Counter(p["original_" + kind] + "->" + p["candidate_" + kind] for p in pairs))
        for kind in ("coverage", "semantics")
    }
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "protocol_verified": True,
        "audit_report_sha256": sha(path), "all126_actual_requests_verified": True,
        "current_guarded_direct_inputs_identical": True, "all116_states_retained": True,
        "no_retry_or_failure_repair": True, "failures": failures,
        "unique_statuses": dict(Counter(r["status"] for r in results)),
        "summary": summary, "paired_results": pairs, "paired_automatic_transitions": transitions,
        "limitations": [
            "Automatic judgments, not human gold or certified answer accuracy.",
            "Matching response.model does not independently attest provider backend weights.",
            "Coverage does not prove factual support; source-review notes remain independent and do not rewrite these labels.",
            "The abstention rendering repair changes one non-direct display; those texts were never sent to either audit.",
            "No fresh retrieval or full production orchestration was executed.",
        ],
    })
    print(json.dumps({
        "protocol_verified": True, "unique_statuses": dict(Counter(r["status"] for r in results)),
        "summary": summary, "transitions": transitions,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
