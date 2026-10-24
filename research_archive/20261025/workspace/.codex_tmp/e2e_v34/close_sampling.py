from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reader_evidence_router_20260909.v27"))
import run_reader_router as shared
from selector_protocol import blocks


def main():
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    post = shared.load(HERE / "postflight_v1.json")
    review = shared.load(HERE / "source_review_v34.json")
    assert [r["id"] for r in review["reviews"]] == [r["id"] for r in pre["rows"]]
    assert post["calls"] == 8 and post["questions"] == 4
    assert post["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert post["generations_sha256"] == shared.sha(HERE / "generations.json")
    assert post["summary"]["model_card"]["action"] == {"abstain": 3}
    assert post["summary"]["legacy"]["action"] == {"answer_directly": 1}
    source_checks = output_checks = warning_checks = errors = 0
    for annotation, source, result in zip(review["reviews"], pre["rows"], gen["rows"], strict=True):
        assert annotation["id"] == source["id"] == result["id"]
        evidence = dict(blocks(source["evidence_brief"]))
        for check in annotation.get("evidence_checks", []):
            assert check["quote"] in evidence[check["evidence_id"]]
            source_checks += 1
        for check in annotation.get("output_checks", []):
            item = result["conditions"][check["condition"]]
            text = item["grounded"]["answer"] if check["field"] == "display" else item[check["field"]]
            assert check["span"] in text
            output_checks += 1
        for kind, warnings in annotation.get("required_warnings", {}).items():
            item = result["conditions"][kind]
            assert item["grounded"]["next_action"] == "abstain"
            assert set(warnings) <= set(item["grounded"]["grounding_warnings"])
            warning_checks += len(warnings)
        if annotation.get("both_length_failures"):
            assert all(item["finish_reason"] == "length" for item in result["conditions"].values())
        for item in result["conditions"].values():
            if item["status"] == "error":
                assert item["finish_reason"] == "length" and item["generated_tokens"] == 4096
                assert "grounded" not in item
                errors += 1
    assert errors == 4
    files = [HERE / name for name in (
        "preflight.json", "generations.json", "postflight_v1.json", "source_review_v34.json", "close_sampling.py",
    )]
    shared.save(HERE / "checkpoint_v34.json", {
        "version": "native9b_quote_sampler_closed_v34",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "production_changed": False,
        "promotion_decision": review["promotion_decision"], "calls": 8, "questions": 4,
        "mechanical_summary": post["summary"], "source_checks": source_checks,
        "output_checks": output_checks, "warning_checks": warning_checks,
        "budget_failures_preserved": errors, "basis_sha256": {str(p): shared.sha(p) for p in files},
        "limitations": review["limitations"],
    })
    print({"source_checks": source_checks, "output_checks": output_checks,
           "warning_checks": warning_checks, "errors_preserved": errors})


if __name__ == "__main__":
    main()
