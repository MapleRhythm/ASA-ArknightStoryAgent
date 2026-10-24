"""CPU replay that observes, but never changes, canonical primary samples."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import run_second_round_reader as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


def main():
    assert not (HERE / "primary_fallback_audit_v1.json").exists()
    pre = shared.load(HERE / "preflight.json")
    saved = shared.load(HERE / "generations.json")
    post = shared.load(HERE / "postflight_v1.json")
    probe.reader.verify(pre)
    assert post["round2_canonical_replays"] == 14
    assert post["generations_sha256"] == shared.sha(HERE / "generations.json")
    helpers, frozen = probe.staged.imports(), probe.sources()
    from asa_arknight_story_agent.inference.generation import answer_conclusion_generation as module
    original = module.sample_grounded_conclusions
    observed = []

    def capture(**kwargs):
        result = original(**kwargs)
        observed.append([deepcopy(asdict(conclusion)) for conclusion in result])
        return result

    module.sample_grounded_conclusions = capture
    rows, transitions = [], Counter()
    try:
        for item, result in zip(pre["records"], saved["records"], strict=True):
            key = item["pair_id"], item["condition"]
            assert key == (result["pair_id"], result["condition"])
            if item["terminal_after_round1"]:
                continue
            observed.clear()
            generator = probe.ReplayGenerator(pre["generator"], item, result)
            replay = probe.invoke(pre, helpers, item, frozen[key], generator)
            assert all(result[k] == v for k, v in replay.items())
            assert len(observed) == 1 and len(observed[0]) <= 1
            primary = observed[0][0] if observed[0] else None
            primary_action = primary["next_action"] if primary else "no_valid_sample"
            final_action = result["conclusion"]["next_action"]
            fallback = len(result["requests"]) == 2
            transitions[f"{primary_action}->{final_action}:fallback={fallback}"] += 1
            rows.append({
                "pair_id": key[0], "condition": key[1], "id": item["id"],
                "primary_conclusion": primary,
                "primary_action": primary_action, "final_action": final_action,
                "canonical_fallback_called": fallback,
                "primary_warnings": primary["grounding_warnings"] if primary else [],
                "final_warnings": result["conclusion"]["grounding_warnings"],
                "final_answer": result["final_answer"],
                "both_raw_outputs_preserved_in_generations": True,
            })
    finally:
        module.sample_grounded_conclusions = original
    assert len(rows) == 14
    shared.save(HERE / "primary_fallback_audit_v1.json", {
        "version": "read_only_primary_fallback_observation_v45",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "generations_sha256": shared.sha(HERE / "generations.json"),
        "script_sha256": shared.sha(Path(__file__)),
        "canonical_replays": 14, "transitions_not_accuracy": transitions,
        "rows": rows, "new_llm_calls": 0, "retrieval_calls": 0,
        "production_changed": False, "training_ready": False,
        "observation_only_no_selection_or_validation_change": True,
    })
    print({"canonical_replays": 14, "transitions_not_accuracy": transitions}, flush=True)


if __name__ == "__main__":
    main()
