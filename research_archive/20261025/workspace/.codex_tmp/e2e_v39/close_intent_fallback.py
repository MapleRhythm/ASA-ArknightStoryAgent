from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reader_evidence_router_20260909.v27"))
import run_reader_router as shared


def main():
    pre = shared.load(HERE / "preflight.json")
    replay = shared.load(HERE / "replay_v1.json")
    assert replay["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    assert replay["summary_not_accuracy"] == {
        "valid_payloads_identical": 54, "normalized": 58, "invalid_intents_recovered": 4,
    }
    expected = {
        "fixed58-007": ["骸骨荒原"], "fixed58-010": ["梁洵", "宗师"],
        "fixed58-046": ["槐天裴"], "fixed58-049": [],
    }
    recovered = []
    for row in replay["rows"]:
        if row["diagnostic"] is None:
            assert row["control"] == row["candidate"] == row["routing_frozen_candidate"]
            continue
        assert row["id"] in expected
        assert row["restored_model_entity_spellings_in_question"] == expected[row["id"]]
        assert row["diagnostic"]["original_error"] == row["original_error"]
        control, candidate = row["control"], row["routing_frozen_candidate"]
        assert row["routing_frozen_diagnostic"]["cue_merge_applied"]
        assert set(control["entities"]) <= set(candidate["entities"])
        assert set(control["keywords"]) <= set(candidate["keywords"])
        assert len(candidate["entities"]) <= 12 and len(candidate["keywords"]) <= 24
        for key in control.keys() - {"entities", "keywords"}:
            assert candidate[key] == control[key]
        for term in expected[row["id"]]:
            assert term in row["question"] and term in candidate["entities"]
        assert row["candidate"]["query_type"] != control["query_type"]
        recovered.append(row["id"])
    assert set(recovered) == set(expected)
    files = [HERE / name for name in (
        "preflight.json", "replay_v1.json", "intent_field_fallback.py",
        "test_intent_field_fallback.py", "replay_intent_field_fallback.py", "close_intent_fallback.py",
    )]
    shared.save(HERE / "checkpoint_v39.json", {
        "version": "initial_planner_field_fallback_closed_v39",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "gpu_used": False, "training_ready": False,
        "valid_payloads_identical": 54, "known_enum_failures_recovered": 4,
        "restored_question_name_spellings": expected,
        "conservative_variant_all_routing_fields_and_control_cues_preserved": True,
        "basic_field_repair_changes_query_type_in_all_four_cases": True,
        "promotion_decision": "only_routing_frozen_variant_may_enter_next_isolated_retrieval_probe_no_production_change",
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "limitations": replay["limitations"],
    })
    print({"unchanged_valid": 54, "restored_failed": 4,
           "promotion": "routing-frozen retrieval probe only; no production change"})


if __name__ == "__main__":
    main()
