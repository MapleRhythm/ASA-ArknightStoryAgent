"""Postflight: exact requests, unchanged fixtures, synthetic—not story—labels."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_answer_slots_v1 as owner
import probe_answer_slots_v2 as v2
import probe_answer_slots_v3 as v3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    source = HERE / "slot_cause_scope40_v3.json"
    report = load(source)
    destination = HERE / "slot_cause_scope40_postflight_v3.json"
    assert not destination.exists()
    fixture_path = HERE / "slot_cause_scope_synthetic20_v1.json"
    fixture = load(fixture_path)
    assert sha(fixture_path) == report["fixture_sha256"]
    assert sha(HERE / "run_slot_cause_scope40_v3.py") == report["driver_sha256"]
    assert sha(HERE / "probe_answer_slots_v2.py") == report["rubric_script_sha256"]["schema_v2"]
    assert sha(HERE / "probe_answer_slots_v3.py") == report["rubric_script_sha256"]["cause_scope_v3"]
    assert sha(HERE.parent / "probe_claim_entailment_v1.py") == report["transport_sha256"]
    assert sha(HERE / "slot_cause_scope40_preflight_v3.json") == report["preflight_sha256"]
    assert v3.RUBRIC.replace(v3.CLARIFICATION, "") == v2.RUBRIC
    originals = {c["id"]: c for c in fixture["cases"]}
    results = {r["id"]: r for r in report["results"]}
    assert len(originals) == 20 and len(results) == len(report["cases"]) == 40
    pairs = defaultdict(list)
    matches = Counter()
    for case in report["cases"]:
        original = originals[case["parent_id"]]
        assert case["question"] == original["question"] and case["shown_answer"] == original["shown_answer"]
        module = v2 if case["condition"] == "schema_v2" else v3
        expected = {"model": "glm-5.3", "temperature": 0, "max_tokens": 16384, "messages": module.messages(case)}
        assert set(json.loads(expected["messages"][1]["content"])) == {"question", "shown_answer"}
        result = results[case["id"]]
        assert result["request_without_credentials"] == expected
        assert result["request_sha256"] == owner.transport.digest(expected)
        assert result["status"] == "ok" and result["response_model"] == "glm-5.3" and result["finish_reason"] == "stop"
        assert v2.validate(json.loads(result["raw_response"]), case) == result["judgement"]
        matches[case["condition"]] += result["judgement"]["coverage"] == original["expected_coverage"]
        pairs[case["parent_id"]].append(case)
    assert len(pairs) == 20 and all(len(pair) == 2 for pair in pairs.values())
    assert sum(pair[0]["condition"] == "schema_v2" for pair in pairs.values()) == 10
    assert matches == {"schema_v2": 19, "cause_scope_v3": 20}
    differing = [
        parent for parent in pairs
        if results[parent + "::schema_v2"]["judgement"]["coverage"] != results[parent + "::cause_scope_v3"]["judgement"]["coverage"]
    ]
    assert differing == ["scope08"]
    saved = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "protocol_verified": True,
        "all40_requests_and_verbatim_spans_verified": True, "fixture_unchanged": True,
        "expected_labels_not_sent": True, "balanced_order_verified": True,
        "report_sha256": sha(source), "fixture_sha256": sha(fixture_path),
        "summary": report["summary"], "different_coverage_cases": differing,
        "independent_assistant_review": [
            "All 40 structured judgments were read; the comparison uses predeclared synthetic coverage labels, not story facts.",
            "scope08 v2 calls a dispatcher's specific stop instruction mere repetition; v3 accepts the immediate instruction as the courier's reason.",
            "Both versions correctly reject the control asking why the dispatcher issued that instruction when only the instruction is repeated.",
            "The two explicit two-part questions remain partial, and unspecified/absent identities remain none under v3.",
            "Only one case differs; this is limited evidence for a definition clarification, not broad judge-accuracy certification.",
        ],
        "limitations": [
            "Assistant-authored synthetic fixtures, no independent human-gold validation.",
            "Passing a coverage contract cannot establish story factuality, attribution or causal truth.",
            "The old39 outputs and story-answer labels remain unchanged; no production gate is installed.",
        ],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(saved, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"postflight": str(destination), "synthetic_matches": dict(matches)}))


if __name__ == "__main__":
    main()
