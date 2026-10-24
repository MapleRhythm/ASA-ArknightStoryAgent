"""Isolate, reproduce, replay and guard-apply score-prioritized prompt dedup."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot, counts
from verify_apply_question_echo_v1 import full_tests
from apply_token_rejection_patch_v1 import scripts, CONFIG

RELATIVE = "asa_arknight_story_agent/inference/evidence/prompt_ordering.py"
CANDIDATE = HERE / "prompt_priority_candidate/src"
PATCH = HERE / "prompt_priority_candidate.py"
TEST = HERE / "test_prompt_priority_dedup.py"
CANONICAL_TEST = ROOT / "release/ASA-ArknightStoryAgent/tests/test_prompt_priority_dedup.py"
PREPARED = HERE / "prompt_priority_prepared_v1.json"
BASELINE = HERE / "prompt_priority_baseline_v1.json"
VERIFIED = HERE / "prompt_priority_candidate_verified_v1.json"
PREAPPLY = HERE / "prompt_priority_preapply_v1.json"
APPLIED = HERE / "prompt_priority_applied_verified_v1.json"


def previous_tests():
    tests = ROOT / "release/ASA-ArknightStoryAgent/tests"
    return full_tests(tests / "test_question_echo_grounding.py") + [
        tests / "test_chain_score_text_binding.py"]


def frozen(pre, key):
    assert snapshot(SOURCE) == pre[key]
    assert sha(CONFIG) == pre["config_sha256"] and scripts() == pre["dataset_scripts_sha256"]
    assert sha(PATCH) == pre["patch_sha256"] and sha(TEST) == pre["test_sha256"]
    assert all(sha(Path(p)) == h for p, h in pre["verification_inputs_sha256"].items())
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == pre["head"]


def tests(source, xml, baseline=False, canonical=False):
    assert not xml.exists()
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.inference.evidence import prompt_ordering as module
    assert Path(module.__file__).resolve() == (source / RELATIVE).resolve()
    import pytest
    files = [TEST] if baseline else previous_tests() + [CANONICAL_TEST if canonical else TEST]
    code = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short",
                       *(str(p) for p in files), "--junitxml=" + str(xml)])
    result = counts(xml)
    assert result["errors"] == result["skipped"] == 0
    if baseline:
        assert code == 1 and 0 < result["failures"] < result["tests"]
    else:
        assert code == 0 and result["failures"] == 0
        assert result["tests"] == 362 + load(BASELINE)["counts"]["tests"]
    origins = {}
    for name, imported in sys.modules.copy().items():
        filename = getattr(imported, "__file__", None)
        if filename and (name == "asa_arknight_story_agent" or name.startswith("asa_arknight_story_agent.")):
            actual = Path(filename).resolve()
            assert actual.is_relative_to(source.resolve())
            origins[name] = str(actual)
    return {"counts": result, "origins": origins, "xml_sha256": sha(xml), "exit_code": int(code)}


def run_replay(source, mode):
    output = HERE / f"prompt_priority_handoff_{mode}_v1.json"
    assert not output.exists()
    subprocess.run([sys.executable, str(HERE / "replay_prompt_priority_handoff_v1.py"),
                    str(source), str(output), mode], check=True)
    return {"path": str(output), "sha256": sha(output)}


def main():
    mode = sys.argv[1]
    assert mode in {"prepare", "baseline", "candidate", "apply", "canonical"}
    if mode == "prepare":
        assert not PREPARED.exists() and not CANDIDATE.exists() and not CANONICAL_TEST.exists()
        before = snapshot(SOURCE)
        assert before == load(HERE / "checkpoint_v20_state.json")["runtime_source_sha256"]
        for relative in before:
            target = CANDIDATE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / relative, target)
        assert snapshot(CANDIDATE) == before
        shutil.copyfile(PATCH, CANDIDATE / RELATIVE)
        candidate = snapshot(CANDIDATE)
        assert set(candidate) == set(before)
        assert [k for k in before if before[k] != candidate[k]] == [RELATIVE]
        # Seal all already completed old-source postflights before changing code.
        sealed = [
            HERE / "quote_pointer8b58_generation_postflight_v1.json",
            HERE / "quote_pointer8b58_blind_audits_postflight_v1.json",
            HERE / "susie_handoff4_postflight_v1.json",
        ]
        assert all(p.is_file() for p in sealed)
        for p in sealed[1:]:
            assert load(p)["protocol_verified"] is True
        verification = previous_tests() + sealed + [
            Path(__file__), HERE / "replay_prompt_priority_handoff_v1.py",
            HERE / "susie_prompt_handoff_v1.json", HERE / "susie_prompt_handoff_preflight_v1.json",
            HERE / "fact_coverage_pairs116_v1.matrix.json",
            HERE / "checkpoint_v20_state.json",
        ]
        save(PREPARED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_source_hashes": before, "candidate_source_hashes": candidate,
            "patch_sha256": sha(PATCH), "test_sha256": sha(TEST),
            "config_sha256": sha(CONFIG), "dataset_scripts_sha256": scripts(),
            "verification_inputs_sha256": {str(p): sha(p) for p in verification},
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "sealed_old_source_postflights": {str(p): sha(p) for p in sealed},
        })
    elif mode == "baseline":
        assert not BASELINE.exists()
        pre = load(PREPARED)
        frozen(pre, "canonical_source_hashes")
        result = tests(SOURCE, HERE / "prompt_priority_baseline_v1.xml", baseline=True)
        replay = run_replay(SOURCE, mode)
        frozen(pre, "canonical_source_hashes")
        save(BASELINE, {**result, "prepared_sha256": sha(PREPARED), "handoff_replay": replay})
    elif mode == "candidate":
        assert not VERIFIED.exists()
        pre = load(PREPARED)
        frozen(pre, "canonical_source_hashes")
        assert snapshot(CANDIDATE) == pre["candidate_source_hashes"]
        assert load(BASELINE)["prepared_sha256"] == sha(PREPARED)
        result = tests(CANDIDATE, HERE / "prompt_priority_candidate_v1.xml")
        replay = run_replay(CANDIDATE, mode)
        frozen(pre, "canonical_source_hashes")
        assert snapshot(CANDIDATE) == pre["candidate_source_hashes"]
        save(VERIFIED, {
            **pre, "prepared_sha256": sha(PREPARED), "baseline_sha256": sha(BASELINE),
            "candidate_tests": result, "handoff_replay": replay,
            "evaluation_only": True, "training_ready": False,
            "new_answer_accuracy": None, "new_full_rag_recall": None,
        })
    elif mode == "apply":
        assert not PREAPPLY.exists() and not APPLIED.exists() and not CANONICAL_TEST.exists()
        verified = load(VERIFIED)
        frozen(verified, "canonical_source_hashes")
        assert verified["prepared_sha256"] == sha(PREPARED)
        assert verified["baseline_sha256"] == sha(BASELINE)
        assert verified["candidate_tests"]["counts"]["failures"] == 0
        assert sha(Path(verified["handoff_replay"]["path"])) == verified["handoff_replay"]["sha256"]
        replay = load(Path(verified["handoff_replay"]["path"]))
        assert replay["new_model_calls"] == 0 and replay["all_inputs_unchanged"]
        assert replay["baseline_reproduced_before_comparison"]
        backup = HERE / "prompt_ordering.original.v20.py"
        assert not backup.exists()
        shutil.copyfile(SOURCE / RELATIVE, backup)
        assert sha(backup) == verified["canonical_source_hashes"][RELATIVE]
        save(PREAPPLY, {
            **load(PREPARED), "candidate_verified_sha256": sha(VERIFIED),
            "backup_path": str(backup), "old_target_sha256": sha(backup),
            "applied_sha256": {
                "release/ASA-ArknightStoryAgent/src/" + RELATIVE: sha(PATCH),
                "release/ASA-ArknightStoryAgent/tests/test_prompt_priority_dedup.py": sha(TEST),
            },
        })
        assert not (SOURCE / RELATIVE).is_symlink()
        shutil.copyfile(PATCH, SOURCE / RELATIVE)
        shutil.copyfile(TEST, CANONICAL_TEST)
        assert snapshot(SOURCE) == verified["candidate_source_hashes"]
    else:
        assert not APPLIED.exists()
        pre = load(PREAPPLY)
        frozen(pre, "candidate_source_hashes")
        assert all(sha(ROOT / p) == h for p, h in pre["applied_sha256"].items())
        result = tests(SOURCE, HERE / "prompt_priority_canonical_v1.xml", canonical=True)
        replay = run_replay(SOURCE, mode)
        candidate = load(HERE / "prompt_priority_handoff_candidate_v1.json")
        assert load(Path(replay["path"]))["conditions"] == candidate["conditions"]
        frozen(pre, "candidate_source_hashes")
        save(APPLIED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preapply_sha256": sha(PREAPPLY), "applied_sha256": pre["applied_sha256"],
            "canonical_tests": result, "backup_path": pre["backup_path"],
            "canonical_handoff_replay": replay, "candidate_replay_identical": True,
            "only_prompt_dedup_priority_changed": True, "training_ready": False,
            "new_certified_correct_answers": 0, "models_changed": False,
            "services_restarted": False, "old_generation_or_audit_rewritten": False,
        })
    print(json.dumps({"stage": mode, "done": True, "training_ready": False}), flush=True)


if __name__ == "__main__":
    main()
