"""Bind assistant review to immutable source/output; not an accuracy judge."""
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
    review = shared.load(HERE / "source_review_v35.json")
    observation = shared.load(HERE / "shutdown_observation_v1.json")
    assert not (HERE / "checkpoint_v35.json").exists()
    assert post["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert post["generations_sha256"] == shared.sha(HERE / "generations.json")
    assert gen["actual_call_count"] == post["calls"] == 32
    assert post["questions"] == 8 and len(pre["rows"]) == len(gen["rows"]) == 16
    assert [r["id"] for r in review["reviews"]] == [r["id"] for r in pre["rows"][:8]]
    sources = {r["pair_id"]: r for r in pre["rows"]}
    results = {r["pair_id"]: r for r in gen["rows"]}
    noise = []
    completed = []
    for source, result in zip(pre["rows"], gen["rows"], strict=True):
        assert source["pair_id"] == result["pair_id"]
        control, candidate = (result["conditions"][k] for k in ("control", "candidate"))
        calls = {call["condition"]: call for call in result["calls"]}
        for kind, item in result["conditions"].items():
            assert item["status"] == "ok"
            assert calls[kind]["finish_reason"] == "stop"
            completed.append({
                "pair_id": source["pair_id"], "condition": kind,
                "raw_sha256": shared.digest(item["raw_output"]),
                "display_sha256": shared.digest(item["grounded"]["answer"]),
                "action": item["grounded"]["next_action"],
            })
        if not source["alias_groups"]:
            assert source["control_prompt"] == source["candidate_prompt"]
            noise.append({
                "pair_id": source["pair_id"],
                "same_raw": control["raw_output"] == candidate["raw_output"],
                "same_display": control["grounded"]["answer"] == candidate["grounded"]["answer"],
                "same_action": control["grounded"]["next_action"] == candidate["grounded"]["next_action"],
            })
    assert noise == post["byte_identical_prompt_noise_controls"]
    assert len(noise) == 8
    totals = {k: sum(r[k] for r in noise) for k in ("same_raw", "same_display", "same_action")}
    assert totals == {"same_raw": 5, "same_display": 6, "same_action": 7}
    changed_noise = results["fixed58-024:seed0"]["conditions"]
    assert changed_noise["control"]["grounded"]["next_action"] == "abstain"
    assert changed_noise["candidate"]["grounded"]["next_action"] == "answer_directly"
    counts = {"source": 0, "output": 0, "warning": 0}
    for annotation in review["reviews"]:
        evidence = dict(blocks(sources[f"{annotation['id']}:seed0"]["evidence_brief"]))
        assert sources[f"{annotation['id']}:seed0"]["evidence_brief"] == sources[f"{annotation['id']}:seed1"]["evidence_brief"]
        for check in annotation["evidence_checks"]:
            assert check["quote"] in evidence[check["evidence_id"]], check
            counts["source"] += 1
        for check in annotation["output_checks"]:
            item = results[f"{annotation['id']}:seed{check['repeat']}"]["conditions"][check["condition"]]
            text = item["grounded"]["answer"] if check["field"] == "display" else item[check["field"]]
            assert check["span"] in text, check
            counts["output"] += 1
        for check in annotation.get("warning_checks", []):
            item = results[f"{annotation['id']}:seed{check['repeat']}"]["conditions"][check["condition"]]
            assert item["grounded"]["next_action"] == "abstain"
            assert set(check["warnings"]) <= set(item["grounded"]["grounding_warnings"])
            counts["warning"] += len(check["warnings"])
    files = [HERE / name for name in (
        "preflight.json", "generations.json", "postflight_v1.json",
        "source_review_v35.json", "shutdown_observation_v1.json", "close_alias_groups.py",
    )]
    shared.save(HERE / "checkpoint_v35.json", {
        "version": "production_lora_alias_groups_closed_v35",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "production_changed": False,
        "promotion_decision": review["promotion_decision"],
        "mechanical_summary_not_accuracy": post["summary"],
        "review_bindings": counts, "completed_output_bindings": completed,
        "noise_control_totals": totals, "shutdown_error_observation": observation,
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "limitations": review["limitations"],
    })
    print({"calls": len(completed), "bindings": counts, "noise_control_totals": totals})


if __name__ == "__main__":
    main()
