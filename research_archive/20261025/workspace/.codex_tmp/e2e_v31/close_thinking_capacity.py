from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reader_evidence_router_20260909.v27"))
import run_reader_router as shared
from selector_protocol import CASE_IDS, blocks


def main():
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    post = shared.load(HERE / "postflight_v1.json")
    review = shared.load(HERE / "source_review_v31.json")
    assert [r["id"] for r in review["reviews"]] == CASE_IDS
    assert post["calls"] == 16 and post["questions"] == 8
    assert post["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert post["generations_sha256"] == shared.sha(HERE / "generations.json")
    assert post["summary"]["think"]["finish"] == {"length": 6, "stop": 2}
    sources = {r["id"]: r for r in pre["rows"]}
    results = {r["id"]: r for r in gen["rows"]}
    source_checks = output_checks = failed_budget_checks = 0
    for row in review["reviews"]:
        source, result = sources[row["id"]], results[row["id"]]
        evidence = dict(blocks(source["evidence_brief"]))
        for check in row.get("evidence_checks", []):
            assert check["quote"] in evidence[check["evidence_id"]], (row["id"], check)
            source_checks += 1
        for check in row.get("output_checks", []):
            item = result["conditions"][check["condition"]]
            text = item["grounded"]["answer"] if check["field"] == "display" else item[check["field"]]
            assert check["span"] in text, (row["id"], check)
            output_checks += 1
        thought = result["conditions"]["think"]
        if row["think"] == "length_failure":
            assert thought["status"] == "error" and thought["finish_reason"] == "length"
            assert thought["generated_tokens"] == 4096
            assert "grounded" not in thought and "answer_text" not in thought
            failed_budget_checks += 1
    assert failed_budget_checks == 6
    basis = [HERE / name for name in (
        "preflight.json", "generations.json", "postflight_v1.json",
        "source_review_v31.json", "close_thinking_capacity.py",
    )]
    shared.save(HERE / "checkpoint_v31.json", {
        "version": "native9b_thinking_budget_closed_v31",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "production_changed": False,
        "promotion_decision": review["promotion_decision"],
        "unique_questions": 8, "calls": 16, "mechanical_summary": post["summary"],
        "source_excerpt_checks": source_checks, "output_span_checks": output_checks,
        "budget_failures_preserved_checks": failed_budget_checks,
        "basis_sha256": {str(p): shared.sha(p) for p in basis},
        "limitations": review["limitations"],
    })
    print({"source_checks": source_checks, "output_checks": output_checks,
           "budget_failures_preserved": failed_budget_checks,
           "promotion_decision": review["promotion_decision"]})


if __name__ == "__main__":
    main()
