"""Isolate, test, replay and guard-apply only the explicit-question echo fix."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import counts, regression_files, save, snapshot
from apply_token_rejection_patch_v1 import scripts, CONFIG

RELATIVE = "asa_arknight_story_agent/inference/grounding/quote_validation.py"
CANDIDATE = HERE / "question_echo_candidate/src"
PATCH = HERE / "question_echo_quote_validation.py"
NEW_TEST = HERE / "test_question_echo_grounding.py"
CANONICAL_TEST = ROOT / "release/ASA-ArknightStoryAgent/tests/test_question_echo_grounding.py"
PREPARED = HERE / "question_echo_prepared_v2.json"
VERIFIED = HERE / "question_echo_candidate_verified_v1.json"
PREAPPLY = HERE / "question_echo_preapply_v1.json"
APPLIED = HERE / "question_echo_applied_verified_v1.json"


def full_tests(new_test):
    return regression_files() + [
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_empty_normalized_quote.py",
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_empty_quote_diagnostic_terms.py",
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_token_rejection_display.py",
        new_test,
    ]


def run_tests(source, xml, *, baseline=False, canonical=False):
    assert not xml.exists()
    sys.path.insert(0, str(source))
    import asa_arknight_story_agent.inference.grounding.quote_validation as module
    assert Path(module.__file__).resolve() == (source / RELATIVE).resolve()
    import pytest
    files = [NEW_TEST] if baseline else full_tests(CANONICAL_TEST if canonical else NEW_TEST)
    code = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short",
                       *(str(p) for p in files), "--junitxml=" + str(xml)])
    origins = {}
    for name, imported in sys.modules.copy().items():
        path = getattr(imported, "__file__", None)
        if path and (name == "asa_arknight_story_agent" or name.startswith("asa_arknight_story_agent.")):
            actual = Path(path).resolve()
            assert actual.is_relative_to(source.resolve())
            origins[name] = str(actual)
    expected = {"tests": 23, "failures": 16, "errors": 0, "skipped": 0} if baseline else {
        "tests": 348, "failures": 0, "errors": 0, "skipped": 0}
    assert code == (1 if baseline else 0) and counts(xml) == expected
    return origins


def replay(label, matrix, predictions, old, expected_rows):
    output = HERE / f"question_echo_{label}_replay_v1.json"
    assert not output.exists()
    subprocess.run([
        sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
        "--validator", "after", "--source-root", str(CANDIDATE),
        "--matrix", str(matrix), "--predictions", str(predictions), "--output", str(output),
    ], check=True)
    before, after = load(old), load(output)
    assert len(before["results"]) == len(after["results"]) == expected_rows
    changes = []
    for a, b in zip(before["results"], after["results"]):
        assert a["id"] == b["id"] and a["status"] == b["status"] and a.get("parsed") == b.get("parsed")
        if a.get("grounded") != b.get("grounded"):
            changes.append({"id": a["id"], "question": a["question"],
                            "before": a.get("grounded"), "after": b.get("grounded")})
    return {"states": expected_rows, "changes": changes, "old_analysis_sha256": sha(old),
            "replay_sha256": sha(output), "matrix_sha256": sha(matrix), "predictions_sha256": sha(predictions)}


def main():
    mode = sys.argv[1]
    assert mode in {"prepare", "baseline", "candidate", "apply", "canonical"}
    if mode == "prepare":
        assert not PREPARED.exists()
        before = snapshot(SOURCE)
        if CANDIDATE.exists():
            previous = load(HERE / "question_echo_prepared_v1.json")
            assert before == previous["canonical_source_hashes"]
            assert snapshot(CANDIDATE) == previous["candidate_source_hashes"]
            assert sha(PATCH) == previous["patch_sha256"]
        else:
            for relative in before:
                target = CANDIDATE / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE / relative, target)
            assert snapshot(CANDIDATE) == before
            shutil.copyfile(PATCH, CANDIDATE / RELATIVE)
        candidate = snapshot(CANDIDATE)
        assert [p for p in before if before[p] != candidate[p]] == [RELATIVE]
        save(PREPARED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_source_hashes": before, "candidate_source_hashes": candidate,
            "test_sha256": sha(NEW_TEST), "patch_sha256": sha(PATCH),
            "config_sha256": sha(CONFIG), "dataset_scripts_sha256": scripts(),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        })
    elif mode == "baseline":
        assert snapshot(SOURCE) == load(PREPARED)["canonical_source_hashes"]
        run_tests(SOURCE, HERE / "question_echo_baseline23_v2.xml", baseline=True)
    elif mode == "candidate":
        assert not VERIFIED.exists()
        pre = load(PREPARED)
        assert snapshot(SOURCE) == pre["canonical_source_hashes"]
        assert snapshot(CANDIDATE) == pre["candidate_source_hashes"]
        assert sha(PATCH) == pre["patch_sha256"] and sha(NEW_TEST) == pre["test_sha256"]
        assert counts(HERE / "question_echo_baseline23_v2.xml") == {
            "tests": 23, "failures": 16, "errors": 0, "skipped": 0}
        xml = HERE / "question_echo_candidate348_v1.xml"
        origins = run_tests(CANDIDATE, xml)
        replays = {
            "fixed79": replay("fixed79", RUN / "fixed79.current_quote.matrix.json",
                              RUN / "fixed79_configured_lora_ctx12288/predictions.json",
                              HERE / "token_rejection_fixed79_replay_v1.json", 79),
            "paired116": replay("paired116", HERE / "fact_coverage_pairs116_v1.matrix.json",
                                HERE / "paired116_ctx12288_v1/predictions.json",
                                HERE / "token_rejection_new116_replay_v1.json", 116),
            "native8b58": replay("native8b58", HERE / "qwen3_8b_native58_v1.matrix.json",
                                 HERE / "qwen3_8b_native58_ctx12288_v1/predictions.json",
                                 HERE / "qwen3_8b_validation_v1.json", 58),
            "no_hints58": replay("no_hints58", HERE / "without_reasoning_hints58_v1.matrix.json",
                                 HERE / "without_reasoning_hints58_ctx12288_v1/predictions.json",
                                 HERE / "without_reasoning_hints58_validation_v1.json", 58),
        }
        assert snapshot(SOURCE) == pre["canonical_source_hashes"]
        save(VERIFIED, {
            **pre, "prepared_sha256": sha(PREPARED), "candidate_tests": counts(xml),
            "candidate_test_xml_sha256": sha(xml), "origins": origins, "replays": replays,
            "training_ready": False, "old_outputs_and_audits_overwritten": False,
        })
        print(json.dumps({"candidate_tests": counts(xml),
                          "changed_by_replay": {k: len(v["changes"]) for k, v in replays.items()}}), flush=True)
    elif mode == "apply":
        assert not PREAPPLY.exists() and not APPLIED.exists() and not CANONICAL_TEST.exists()
        verified = load(VERIFIED)
        assert verified["prepared_sha256"] == sha(PREPARED)
        assert verified["candidate_tests"] == {"tests": 348, "failures": 0, "errors": 0, "skipped": 0}
        assert snapshot(SOURCE) == verified["canonical_source_hashes"]
        assert sha(PATCH) == verified["patch_sha256"] and sha(NEW_TEST) == verified["test_sha256"]
        assert sha(CONFIG) == verified["config_sha256"] and scripts() == verified["dataset_scripts_sha256"]
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == verified["head"]
        assert all(verified["replays"][k]["changes"] == [] for k in ("fixed79", "paired116", "native8b58"))
        changes = verified["replays"]["no_hints58"]["changes"]
        assert len(changes) == 1 and changes[0]["id"].startswith("relabel-04e53609ed1e65775ec14112::")
        assert changes[0]["before"]["answer"] == changes[0]["question"]
        assert changes[0]["before"]["next_action"] == "answer_directly"
        assert changes[0]["after"]["next_action"] == "retrieve_more" and changes[0]["after"]["answer"] == ""
        assert "supported_fact_1_repeats_question" in changes[0]["after"]["grounding_warnings"]
        backup = HERE / "question_echo_quote_validation.original.py"
        assert not backup.exists()
        shutil.copyfile(SOURCE / RELATIVE, backup)
        assert sha(backup) == verified["canonical_source_hashes"][RELATIVE]
        save(PREAPPLY, {
            **load(PREPARED), "candidate_verified_sha256": sha(VERIFIED),
            "old_target_sha256": sha(backup), "backup_path": str(backup),
            "candidate_sha256": {
                "release/ASA-ArknightStoryAgent/src/" + RELATIVE: sha(PATCH),
                "release/ASA-ArknightStoryAgent/tests/test_question_echo_grounding.py": sha(NEW_TEST),
            },
        })
        shutil.copyfile(PATCH, SOURCE / RELATIVE)
        shutil.copyfile(NEW_TEST, CANONICAL_TEST)
        assert snapshot(SOURCE) == verified["candidate_source_hashes"]
        print("Applied explicit-question echo guard; canonical verification pending.", flush=True)
    else:
        pre = load(PREAPPLY)
        assert not APPLIED.exists()
        assert snapshot(SOURCE) == pre["candidate_source_hashes"]
        assert all(sha(ROOT / p) == h for p, h in pre["candidate_sha256"].items())
        xml = HERE / "question_echo_canonical348_v1.xml"
        origins = run_tests(SOURCE, xml, canonical=True)
        assert sha(CONFIG) == pre["config_sha256"] and scripts() == pre["dataset_scripts_sha256"]
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == pre["head"]
        save(APPLIED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preapply_sha256": sha(PREAPPLY), "applied_sha256": pre["candidate_sha256"],
            "canonical_tests": counts(xml), "origins": origins, "backup_path": pre["backup_path"],
            "other_runtime_and_config_unchanged": True, "training_ready": False,
            "old_audits_overwritten": False, "context_replay_changed_actions": 1,
            "new_certified_correct_answers": 0, "models_changed": False, "services_restarted": False,
        })
    print(json.dumps({"stage": mode, "done": True, "training_ready": False}), flush=True)


if __name__ == "__main__":
    main()
