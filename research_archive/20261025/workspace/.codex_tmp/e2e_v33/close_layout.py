from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
MODULE = ROOT / "release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/grounding/grounded_fact_answers.py"
TEST = ROOT / "release/ASA-ArknightStoryAgent/tests/test_grounded_fact_layout_regression.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    replay = json.loads((HERE / "isolated_replay_v1.json").read_text())
    installed = json.loads((HERE / "installed_verification_v1.json").read_text())
    assert replay["replayed_records"] == 58 and replay["changed_answers"] == 23
    assert installed["all_passed"]
    assert installed["tests"]["release"]["counts"] == {"tests": 306, "failures": 0, "errors": 0, "skipped": 0}
    assert installed["tests"]["root"]["counts"] == {"tests": 248, "failures": 0, "errors": 0, "skipped": 0}
    assert installed["installed_source_sha256"] == replay["candidate_sha256"] == sha(MODULE)
    assert sha(TEST) == sha(HERE / "test_grounded_fact_layout_regression.py")
    source_hashes = dict(replay["source_sha256"])
    source_hashes[str(MODULE)] = sha(MODULE)
    assert all(sha(Path(p)) == h for p, h in source_hashes.items())
    counterexample = next(r for r in replay["results"]
                          if r["id"] == "v29:reader_chain_owner_20260910.v29:fixed58-028:seed1:candidate")
    assert "为了向这个老东西摇尾乞怜" in counterexample["candidate_answer"]
    assert counterexample["all_non_answer_fields_identical"]
    files = [HERE / name for name in (
        "baseline_grounded_fact_answers.py", "grounded_fact_answers.py",
        "test_grounded_fact_layout.py", "test_grounded_fact_layout_regression.py",
        "verify_layout.py", "isolated_replay_v1.json", "apply_layout.sh",
        "preexisting_source_diff.patch", "applied_source.sha256", "verify_installed.py",
        "installed_verification_v1.json", "close_layout.py",
        "release_tests_v1.xml", "release_tests_v1.log", "root_tests_v1.xml", "root_tests_v1.log",
    )]
    result = {
        "version": "canonical_grounded_fact_layout_closed_v33",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "retain_narrow_presentation_fix",
        "production_source_changed": [str(MODULE)], "new_tests": [str(TEST)],
        "no_other_existing_release_source_changed": True,
        "model_pointer_changed": False, "services_restarted": False, "training_performed": False,
        "training_ready": False, "semantic_accuracy_gain_claimed": False,
        "new_generation_calls": 0, "replayed_outputs": 58, "unique_development_questions": 8,
        "presentation_changes": 23, "non_answer_field_changes": 0,
        "budget_errors_not_replayed": 6, "known_wrong_fact_preserved_example": counterexample,
        "test_counts": {k: v["counts"] for k, v in installed["tests"].items()},
        "basis_sha256": {str(p): sha(p) for p in files},
        "limitations": [
            "Only formatting of the fact-reconstruction branch changes; unchanged raw answers are not globally reformatted.",
            "Existing statement punctuation is not stripped or repaired; original words/quotes are preserved.",
            "This does not solve semantic entailment, speaker attribution or wrong facts reconstructed into the display.",
            "58 replayed outputs are repeated conditions on eight development questions, not 58 independent questions.",
            "Existing dirty source edits are preserved and separately backed up, not attributed to this patch.",
        ],
    }
    with (HERE / "checkpoint_v33.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"decision": result["decision"], "test_counts": result["test_counts"],
                      "semantic_accuracy_gain_claimed": False}), flush=True)


if __name__ == "__main__":
    main()
