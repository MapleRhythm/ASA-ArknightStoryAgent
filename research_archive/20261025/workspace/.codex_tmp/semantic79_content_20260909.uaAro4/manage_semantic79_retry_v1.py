"""One explicit failed-item retry; original judgments and denominator remain."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "canonical_fixed79_validation_v1.json"
ORIGINAL = HERE / "semantic79_content_v2.json"
RETRY_ANALYSIS = HERE / "semantic79_retry1_analysis.json"
RETRY = HERE / "semantic79_retry1.json"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "merge"])
    args = parser.parse_args()
    analysis, original = load(ANALYSIS), load(ORIGINAL)
    rows = analysis["results"]
    by_id = {r["id"]: r for r in original["results"]}
    assert len(rows) == len(by_id) == 79
    assert set(by_id) == {r["id"] for r in rows}
    assert original["source_analysis_sha256"] == sha(ANALYSIS)
    failed = {rid for rid, result in by_id.items() if result["status"] == "error"}
    assert len(failed) == 3
    if args.mode == "prepare":
        write_new(RETRY_ANALYSIS, {
            "results": [row for row in rows if row["id"] in failed],
            "source_analysis_sha256": sha(ANALYSIS), "original_audit_sha256": sha(ORIGINAL),
            "retry_reason": "Only original schema/validation failures; no successful item is rerun.",
            "evaluation_only": True, "training_ready": False,
        })
        print(json.dumps({"retry_ids": sorted(failed), "full_denominator": len(rows)}))
        return

    retry = load(RETRY)
    assert retry["source_analysis_sha256"] == sha(RETRY_ANALYSIS)
    assert retry["rows"] == len(failed)
    assert retry["requested_model"] == original["requested_model"]
    for key in ("protocol", "judge_script_sha256", "rubric_script_sha256", "transport_script_sha256"):
        assert original[key] == retry[key], key
    retry_by_id = {r["id"]: r for r in retry["results"]}
    assert len(retry_by_id) == len(retry["results"]) and set(retry_by_id) == failed
    combined = []
    for row in rows:
        rid = row["id"]
        selected = retry_by_id[rid] if rid in failed else by_id[rid]
        combined.append({
            **selected,
            "selected_attempt": 2 if rid in failed else 1,
            "original_status": by_id[rid]["status"],
            "original_error": by_id[rid].get("error"),
        })
    action_by_id = {r["id"]: r["grounded"]["next_action"] for r in rows}
    label = lambda result: (result.get("judgement") or {}).get("verdict", "audit_error") if result["status"] == "ok" else "audit_error"
    scores = dict(Counter(label(r) for r in combined))
    direct = [r for r in combined if action_by_id[r["id"]] == "answer_directly"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": original["protocol"], "requested_model": original["requested_model"],
        "judge_type": "model_not_human", "training_ready": False, "evaluation_only": True,
        "source_analysis_sha256": sha(ANALYSIS),
        "original_audit_sha256": sha(ORIGINAL), "retry_audit_sha256": sha(RETRY),
        "states": 79, "unique_questions": 58, "total_api_requests": 79 + len(failed),
        "original_statuses": original["statuses"], "retry_statuses": retry["statuses"],
        "effective_statuses": dict(Counter(r["status"] for r in combined)),
        "all79_verdict_counts": scores, "direct39_verdict_counts": dict(Counter(label(r) for r in direct)),
        "context_sufficiency_counts": dict(Counter(
            (r.get("judgement") or {}).get("context_sufficiency", "audit_error") if r["status"] == "ok" else "audit_error"
            for r in combined
        )),
        "results": combined,
        "limitations": [
            "Model-audit labels only; known source-review disagreements remain separate.",
            "79 historical evidence states from 58 questions, not fresh production E2E tests.",
            "Original failures and raw responses remain in the original artifact; failed retries remain in denominator.",
            "Context sufficiency is not an evidence retrieval recall metric.",
            "No new generation/retrieval, automatic answer correction or training approval.",
        ],
    }
    assert len(direct) == 39 and sum(scores.values()) == 79
    write_new(HERE / "semantic79_combined_v1.json", report)
    print(json.dumps({k: report[k] for k in (
        "effective_statuses", "all79_verdict_counts", "direct39_verdict_counts", "context_sufficiency_counts",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
