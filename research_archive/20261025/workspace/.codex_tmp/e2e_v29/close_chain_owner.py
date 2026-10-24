from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "reader_evidence_router_20260909.v27"
sys.path.insert(0, str(PREVIOUS))
import run_reader_router as shared
from selector_protocol import CASE_IDS, blocks


def main():
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    post = shared.load(HERE / "postflight_v1.json")
    review = shared.load(HERE / "source_review_v29.json")
    assert post["generations_sha256"] == shared.sha(HERE / "generations.json")
    assert post["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert post["calls"] == 32 and post["pairs"] == 16
    assert [r["id"] for r in review["reviews"]] == CASE_IDS
    by_pair = {r["pair_id"]: r for r in gen["rows"]}
    original = {r["id"]: dict(blocks(r["evidence_brief"])) for r in pre["rows"]}
    excerpt_count = 0
    for row in review["reviews"]:
        for check in row.get("evidence_checks", []):
            assert check["quote"] in original[row["id"]][check["evidence_id"]], (row["id"], check)
            excerpt_count += 1
        for repeat in range(2):
            actual = by_pair[f"{row['id']}:seed{repeat}"]
            for kind in ("control", "candidate"):
                label = row[f"seed{repeat}"][kind]
                if label == "no_answer":
                    assert actual["conditions"][kind]["grounded"]["next_action"] == "abstain"
                else:
                    assert actual["conditions"][kind]["grounded"]["next_action"] == "answer_directly"
        if "display_check" in row:
            check = row["display_check"]
            actual = by_pair[check["pair_id"]]["conditions"]["candidate"]
            assert check["candidate_answer_span"] in actual["grounded"]["answer"]
            payload = shared.parse_prefilled(actual["raw_output"])[0]
            assert check["candidate_answer_span"] not in payload["final_answer"]
    report = {
        "version": "explicit_chain_owner_closed_v29",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "production_changed": False,
        "promotion_decision": "reject_this_candidate",
        "unique_questions": 8, "paired_seeds_per_question": 2,
        "calls": 32, "finish_reasons": post["finish_reasons"],
        "mechanical_summary": post["summary"],
        "source_excerpt_checks_passed": excerpt_count,
        "raw_vs_display_negative_regression_verified": True,
        "basis_sha256": {str(p): shared.sha(p) for p in (
            HERE / "preflight.json", HERE / "generations.json", HERE / "postflight_v1.json",
            HERE / "source_review_v29.json", Path(__file__),
        )},
        "limitations": review["limitations"],
    }
    shared.save(HERE / "checkpoint_v29.json", report)
    print(json.dumps({"calls": 32, "source_excerpt_checks_passed": excerpt_count,
                      "promotion_decision": report["promotion_decision"]}), flush=True)


if __name__ == "__main__":
    main()
