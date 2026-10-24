from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "planner_entity_provenance_20260910.v37"
sys.path.insert(0, str(HERE.parent / "reader_evidence_router_20260909.v27"))
import run_reader_router as shared


def main():
    pre = shared.load(HERE / "preflight.json")
    gen = shared.load(HERE / "generations.json")
    post = shared.load(HERE / "postflight_v1.json")
    method = shared.load(HERE / "planner_method_equivalence_v1.json")
    review = shared.load(HERE / "source_review_v37.json")
    failure = shared.load(ORIGINAL / "startup_failure_v1.json")
    assert not (HERE / "checkpoint_v37.json").exists()
    assert post["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert post["generations_sha256"] == shared.sha(HERE / "generations.json")
    assert gen["actual_call_count"] == 58 and failure["actual_generation_calls"] == 0
    assert pre["attempt"] == 2
    assert pre["rows"] == shared.load(ORIGINAL / "preflight.json")["rows"]
    assert post["summary_not_accuracy"] == {
        "ok": 58, "valid_model_payload": 54, "valid_with_code_added_entities": 52,
        "valid_changed_query": 52, "added_entity_occurrences": 152, "heuristic_fallback": 4,
    }
    assert len(method["checks"]) == 58
    assert all(r["status"] == "canonical_method_matches" for r in method["checks"])
    rows = {r["id"]: r for r in gen["rows"]}
    for check in review["bindings"]:
        record = rows[check["id"]]["processing"]
        assert record["parsed_model_payload"]["entities"] == check["model_entities"]
        assert record["code_added_entities"] == check["code_added_entities"]
    for check in review["fallback_bindings"]:
        record = rows[check["id"]]["processing"]
        assert record["normalization_status"] == "heuristic_fallback"
        assert record["parsed_model_payload"]["intent"] == check["invalid_intent"]
        assert record["parsed_model_payload"]["entities"] == check["intact_model_entities"]
        assert record["control"] == record["candidate"] and record["candidate_not_applied"]
    retained = sum(
        term in row["processing"]["candidate"]["keywords"]
        for row in gen["rows"]
        for term in (row["processing"]["code_added_entities"] or [])
    )
    assert retained == 152
    log = (HERE / "run_v1.log").read_text()
    assert "Shutdown complete" in log
    assert "Engine core proc EngineCore died unexpectedly" in log
    assert "destroy_process_group() was not called" in log
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    files = [HERE / name for name in (
        "preflight.json", "generations.json", "postflight_v1.json", "planner_method_equivalence_v1.json",
        "source_review_v37.json", "run_v1.log", "close_planner_capture.py",
    )] + [ORIGINAL / "startup_failure_v1.json"]
    shared.save(HERE / "checkpoint_v37.json", {
        "version": "planner_entity_provenance_closed_v37",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "training_ready": False,
        "promotion_decision": review["promotion_decision"],
        "summary_not_accuracy": post["summary_not_accuracy"],
        "canonical_method_replays": 58, "addition_bindings": len(review["bindings"]),
        "fallback_bindings": len(review["fallback_bindings"]),
        "all_152_additions_remain_in_keywords": True,
        "startup_failure_and_shutdown_error_preserved": True,
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "limitations": review["limitations"],
    })
    print({"calls": 58, "method_replays": 58, "added_entities_still_in_keywords": retained,
           "promotion": review["promotion_decision"]})


if __name__ == "__main__":
    main()
