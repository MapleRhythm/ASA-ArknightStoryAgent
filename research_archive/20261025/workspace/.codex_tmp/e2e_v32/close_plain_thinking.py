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
    review = shared.load(HERE / "source_review_v32.json")
    old = shared.load(HERE.parent / "reader_thinking_budget_20260910.v31/preflight.json")
    assert [r["id"] for r in review["reviews"]] == CASE_IDS
    assert post["calls"] == 16 and post["summary"]["think"]["finish"] == {"length": 4, "stop": 4}
    assert post["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert post["generations_sha256"] == shared.sha(HERE / "generations.json")
    sources, results = {r["id"]: r for r in pre["rows"]}, {r["id"]: r for r in gen["rows"]}
    for previous, current in zip(old["rows"], pre["rows"], strict=True):
        for key in ("id", "question", "hypothesis", "evidence_brief", "sampling_seed"):
            assert previous[key] == current[key]
    source_checks = output_checks = errors = 0
    for row in review["reviews"]:
        evidence = dict(blocks(sources[row["id"]]["evidence_brief"]))
        result = results[row["id"]]
        for check in row.get("evidence_checks", []):
            assert check["quote"] in evidence[check["evidence_id"]], (row["id"], check)
            source_checks += 1
        for check in row.get("output_checks", []):
            item = result["conditions"][check["condition"]]
            assert item["status"] == "ok" and check["span"] in item["display_answer"]
            output_checks += 1
        if row["think"] == "length_failure":
            item = result["conditions"]["think"]
            assert item["status"] == "error" and item["finish_reason"] == "length"
            assert item["generated_tokens"] == 4096 and "display_answer" not in item
            errors += 1
    assert errors == 4
    files = [HERE / name for name in (
        "preflight.json", "generations.json", "postflight_v1.json",
        "source_review_v32.json", "close_plain_thinking.py",
    )]
    report = {
        "version": "native9b_plain_thinking_closed_v32",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "production_changed": False,
        "promotion_decision": review["promotion_decision"],
        "questions": 8, "calls": 16, "mechanical_summary": post["summary"],
        "source_excerpt_checks": source_checks, "actual_output_span_checks": output_checks,
        "budget_errors_preserved": errors, "full_source_and_hypothesis_match_v31": True,
        "prose_NOT_production_validated": True,
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "limitations": review["limitations"],
    }
    shared.save(HERE / "checkpoint_v32.json", report)
    print({"source_checks": source_checks, "output_checks": output_checks, "errors_preserved": errors})


if __name__ == "__main__":
    main()
