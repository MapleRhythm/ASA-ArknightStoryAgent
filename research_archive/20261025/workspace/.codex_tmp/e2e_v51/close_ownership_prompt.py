"""Close only after full actual-output review, with no production promotion."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import run_ownership_prompt as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


def main():
    assert not (HERE / "checkpoint_v51.json").exists()
    pre, generated = shared.load(HERE / "preflight.json"), shared.load(HERE / "generations.json")
    checked, review = shared.load(HERE / "postflight_v1.json"), shared.load(HERE / "source_review_v1.json")
    probe.verify(pre)
    pre_sha = shared.sha(HERE / "preflight.json")
    assert generated["preflight_sha256"] == checked["preflight_sha256"] == pre_sha
    assert checked["canonical_replays"] == generated["actual_call_count"] == 24
    actual = {(r["pair_id"], r["condition"]): r for r in generated["attempts"]}
    inputs = {(r["id"], r["condition"]): r for r in pre["conditions"]}
    assert len(actual) == len(review["rows"]) == 24
    assert review["human_gold"] is False and review["production_promotion"] is False
    assert review["model_inputs_modified"] is False
    seen = set()
    for row in review["rows"]:
        key = row["pair_id"], row["condition"]
        assert key not in seen
        seen.add(key)
        saved = actual[key]
        assert saved["status"] == "ok"
        assert saved["calls"][0]["finish_reason"] == "stop"
        assert row["raw_sha256"] == shared.digest(saved["sanitized_raw_output"])
        assert row["answer_sha256"] == shared.digest(saved["conclusion"]["answer"])
        assert row["assistant_observation"] and row["assistant_reviewed_raw_and_display"]
    assert seen == set(actual)
    probe.reader.runtime_imports()
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
    bindings = 0
    for witness in review["source_bindings"]:
        for kind in probe.KINDS:
            item = inputs[(witness["id"], kind)]
            visible = item["visible_evidence"]
            assert visible == inputs[(witness["id"], "baseline")]["visible_evidence"]
            block = evidence_id_text_map_from_prompt(visible)[witness["evidence_id"]]
            assert witness["excerpt"] in block
            assert witness["block_sha256"] == shared.digest(block)
            bindings += 1
    assert len(review["source_bindings"]) >= 6
    idle = shared.load(HERE / "idle_samples.json")
    assert idle["elapsed_seconds"] >= 60 and len(idle["samples"]) >= 13
    assert all(not s["compute_pids"] and s["memory_mib"] < 256 and s["utilization"] < 10
               for s in idle["samples"])
    gpu = shared.gpu_idle()
    assert probe.reader.retrieval.numeric_mapping() == pre["verified_nvidia_mapping"]
    report = probe.integration.ROOT / "docs/CANONICAL_OWNERSHIP_RULE_PROMPT_20260910_v51.md"
    paths = [HERE / name for name in (
        "README.md", "run_ownership_prompt.py", "close_ownership_prompt.py",
        "preflight.json", "prepare.log", "idle_samples.json",
        "generations.progress.jsonl", "generations.json", "run_v1.log",
        "postflight_v1.json", "check_v1.log", "source_review_v1.json",
        "bind_source_review.py",
    )] + [report]
    probe.verify(pre)
    shared.save(HERE / "checkpoint_v51.json", {
        "version": "isolated_ownership_rule_prompt_closed_v51",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_first_round_model_calls": 24, "canonical_replays": 24,
        "underlying_questions": 6, "paired_seeds": 2,
        "output_review_bindings": 24, "source_condition_bindings": bindings,
        "summary_not_accuracy": checked["summary_not_accuracy"],
        "finish_counts": dict(Counter(r["calls"][0]["finish_reason"] for r in actual.values())),
        "production_changed": False, "production_prompt_promoted": False,
        "training_ready": False, "retrieval_calls": 0, "new_planner_calls": 0,
        "next_round_or_final_direct_fallback_calls": 0,
        "new_gpu_lock_and_idle_window_verified": True, "gpu_idle_at_close": gpu,
        "gpu_processes_at_close_read_only": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader",
        ], text=True),
        "human_gold": False, "semantic_accuracy_gain_claimed": False,
        "review_decision": review["decision"],
        "services_restarted": False, "default_model_changed": False, "commit_or_push": False,
        "basis_sha256": {str(p): shared.sha(p) for p in paths},
    })
    print({"calls": 24, "canonical_replays": 24, "output_bindings": 24,
           "source_bindings": bindings, "production_prompt_promoted": False}, flush=True)


if __name__ == "__main__":
    main()
