"""Keep paired development counts and source-review limitations explicit."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


def sha(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def label(result):
    return (result.get("judgement") or {}).get("verdict", "audit_error") if result["status"] == "ok" else "audit_error"


def main():
    baseline = load("semantic79_combined_v1.json")
    audited = load("semantic_filter7_audit_v1.json")
    candidates = load("semantic_filter7_candidate_v1.json")
    assert audited["source_analysis_sha256"] == sha("semantic_filter7_candidate_v1.json")
    assert audited["protocol"] == baseline["protocol"]
    old_by_id = {r["id"]: r for r in baseline["results"]}
    new_by_id = {r["id"]: r for r in audited["results"]}
    candidate_by_id = {r["id"]: r for r in candidates["results"]}
    traces = load("semantic_filter7_trace_v1.json")["traces"]
    assert len(new_by_id) == len(traces) == 7
    pairs = []
    for trace in traces:
        parent, candidate_id = trace["parent_id"], trace["candidate_id"]
        assert candidate_id == parent + "::semantic_filter_v1"
        old, new = old_by_id[parent], new_by_id[candidate_id]
        candidate = candidate_by_id[candidate_id]
        pairs.append({
            "parent_id": parent, "candidate_id": candidate_id, "question": candidate["question"],
            "baseline_auto_verdict": label(old), "candidate_auto_verdict": label(new),
            "candidate_audit_status": new["status"], "candidate_audit_error": new.get("error"),
            "candidate_action": candidate["grounded"]["next_action"],
            "candidate_answer": candidate["grounded"]["answer"],
            "candidate_reason": (new.get("judgement") or {}).get("reason"),
            "kept_fact_count": trace["kept_fact_count"], "audit_failures": trace["audit_failures"],
        })
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "promotion_approved": False,
        "states": 7, "selection": "deliberate_development_diagnostics_not_holdout",
        "source_artifacts_sha256": {name: sha(name) for name in (
            "semantic79_combined_v1.json", "semantic_filter_plan_v1.json",
            "semantic_filter7_candidate_v1.json", "semantic_filter7_audit_v1.json",
            "cited_claim_probe13_v2.json", "semantic_filter7_trace_v1.json",
        )},
        "baseline_auto_counts": dict(Counter(p["baseline_auto_verdict"] for p in pairs)),
        "candidate_auto_counts": dict(Counter(p["candidate_auto_verdict"] for p in pairs)),
        "candidate_actions": dict(Counter(p["candidate_action"] for p in pairs)),
        "pairs": pairs,
        "limitations": [
            "All counts are imperfect model judgments; baseline includes source-confirmed false positives.",
            "A before/after automatic count delta is not a real accuracy improvement estimate.",
            "One claim-level quote-format failure forces a whole diagnostic abstention by the frozen conservative rule.",
            "Atomic fact filtering can remove supported subclaims bundled with an unsupported subclaim.",
            "Yato's generic crisis description survives entailment and still does not identify the requested crisis.",
            "No independent holdout, runtime semantic filter, target rewrite, training or deployment.",
        ],
    }
    with (HERE / "semantic_filter7_comparison_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
