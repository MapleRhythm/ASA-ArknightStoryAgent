"""Disable MiniRAG retrieval and prompt hints with isolated, guarded application."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot, counts
from verify_apply_prompt_priority_v1 import previous_tests
from apply_token_rejection_patch_v1 import scripts

PATCH_DIR = HERE / "no_minirag_candidate_files_v1"
CANDIDATE = HERE / "no_minirag_candidate_v1/src"
CONFIG = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
RUNNER = ROOT / "scripts/run_cpu_inference.py"
TEST = HERE / "test_no_minirag_runtime_v1.py"
CANONICAL_TEST = ROOT / "release/ASA-ArknightStoryAgent/tests/test_no_minirag_runtime.py"
REPLAY = HERE / "replay_no_minirag_prompts_v1.py"
PRE = HERE / "no_minirag_prepared_v1.json"
BASELINE = HERE / "no_minirag_baseline_v1.json"
VERIFIED = HERE / "no_minirag_candidate_verified_v1.json"
PREAPPLY = HERE / "no_minirag_preapply_v1.json"
APPLIED = HERE / "no_minirag_applied_verified_v1.json"

SOURCE_TARGETS = {
    "asa_arknight_story_agent/config.py": "config.py",
    "asa_arknight_story_agent/inference/pipeline/retrieval_round.py": "pipeline_retrieval_round.py",
    "asa_arknight_story_agent/inference/retrieval/pipeline.py": "retrieval_pipeline.py",
    "asa_arknight_story_agent/inference/retrieval/round.py": "retrieval_round_function.py",
    "asa_arknight_story_agent/inference/retrieval/search.py": "search.py",
    "asa_arknight_story_agent/inference/generation/minimal_conclusion_prompt.py": "minimal_conclusion_prompt.py",
    "asa_arknight_story_agent/inference/generation/direct_answer_prompt.py": "direct_answer_prompt.py",
}
NON_SOURCE_TARGETS = {
    CONFIG: PATCH_DIR / "runtime_gpu_reranker_qwen35_4b.json",
    RUNNER: PATCH_DIR / "run_cpu_inference.py",
}


def all_tests(new_test):
    return previous_tests() + [
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_hint_metadata_isolation.py",
        new_test,
    ]


def frozen(pre, source_key, config_key, runner_key):
    assert snapshot(SOURCE) == pre[source_key]
    assert sha(CONFIG) == pre[config_key] and sha(RUNNER) == pre[runner_key]
    assert scripts() == pre["dataset_scripts_sha256"]
    assert all(sha(Path(p)) == h for p, h in pre["verification_inputs_sha256"].items())
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == pre["head"]


def tests(source, config, runner, mode):
    xml = HERE / f"no_minirag_{mode}_v1.xml"
    assert not xml.exists()
    env = dict(os.environ, ASA_NO_MINIRAG_CONFIG=str(config),
               ASA_NO_MINIRAG_RUNNER=str(runner), ASA_ROOT_SRC=str(ROOT / "src"))
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.inference.retrieval import search
    assert Path(search.__file__).resolve() == (
        source / "asa_arknight_story_agent/inference/retrieval/search.py").resolve()
    import pytest
    files = [TEST] if mode == "baseline" else all_tests(CANONICAL_TEST if mode == "canonical" else TEST)
    old_env = {k: os.environ.get(k) for k in ("ASA_NO_MINIRAG_CONFIG", "ASA_NO_MINIRAG_RUNNER", "ASA_ROOT_SRC")}
    os.environ.update({k: env[k] for k in old_env})
    try:
        code = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short",
                           *(str(p) for p in files), "--junitxml=" + str(xml)])
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    result = counts(xml)
    assert result["errors"] == result["skipped"] == 0
    if mode == "baseline":
        assert code == 1 and result["tests"] == 12 and 0 < result["failures"] < 12
    else:
        assert code == 0 and result == {"tests": 387, "failures": 0, "errors": 0, "skipped": 0}
    origins = {}
    for name, module in sys.modules.copy().items():
        path = getattr(module, "__file__", None)
        if path and (name == "asa_arknight_story_agent" or name.startswith("asa_arknight_story_agent.")):
            assert Path(path).resolve().is_relative_to(source.resolve())
            origins[name] = path
    return {"counts": result, "origins": origins, "xml_sha256": sha(xml)}


def replay(source, config, mode):
    output = HERE / f"no_minirag_prompt_replay_{mode}_v1.json"
    assert not output.exists()
    subprocess.run([
        sys.executable, str(REPLAY), str(source), str(config), str(output),
    ], check=True)
    result = load(output)
    assert result["states"] == 62 and result["all_evidence_and_packing_unchanged"]
    assert result["only_minirag_hint_line_removed"] and result["all_fit_configured_output_budget"]
    return {"path": str(output), "sha256": sha(output)}


def main():
    mode = sys.argv[1]
    assert mode in {"prepare", "baseline", "candidate", "apply", "canonical"}
    if mode == "prepare":
        assert not PRE.exists() and not CANDIDATE.exists() and not CANONICAL_TEST.exists()
        before = snapshot(SOURCE)
        hint = load(HERE / "hint_metadata_preapply_v2.json")
        applied_hint = load(HERE / "hint_metadata_applied_verified_v2.json")
        assert applied_hint["preapply_sha256"] == sha(HERE / "hint_metadata_preapply_v2.json")
        assert applied_hint["canonical_tests"]["counts"] == {
            "tests": 375, "failures": 0, "errors": 0, "skipped": 0}
        assert before == hint["candidate_source_hashes"]
        assert sha(CONFIG) == hint["config_sha256"]
        for relative in before:
            target = CANDIDATE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / relative, target)
        assert snapshot(CANDIDATE) == before
        for relative, name in SOURCE_TARGETS.items():
            shutil.copyfile(PATCH_DIR / name, CANDIDATE / relative)
        after = snapshot(CANDIDATE)
        assert set(after) == set(before)
        assert sorted(p for p in before if before[p] != after[p]) == sorted(SOURCE_TARGETS)
        assert set(NON_SOURCE_TARGETS) == {CONFIG, RUNNER}
        sealed = [
            HERE / "checkpoint_v21_state.json", HERE / "hint_metadata_applied_verified_v2.json",
            HERE / "single_pointer_handoff4_postflight_v1.json",
            HERE / "single_pointer_handoff4_source_review_v1.json",
        ]
        assert all(p.is_file() for p in sealed)
        inputs = all_tests(ROOT / "release/ASA-ArknightStoryAgent/tests/test_hint_metadata_isolation.py") + [
            Path(__file__), TEST, REPLAY, *PATCH_DIR.iterdir(), *sealed,
            HERE / "fact_coverage_pairs116_v1.matrix.json", HERE / "susie_handoff4_v1.matrix.json",
        ]
        save(PRE, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_source_hashes": before, "candidate_source_hashes": after,
            "canonical_config_sha256": sha(CONFIG), "candidate_config_sha256": sha(NON_SOURCE_TARGETS[CONFIG]),
            "canonical_runner_sha256": sha(RUNNER), "candidate_runner_sha256": sha(NON_SOURCE_TARGETS[RUNNER]),
            "candidate_patch_sha256": {relative: sha(PATCH_DIR / name)
                                       for relative, name in SOURCE_TARGETS.items()},
            "test_sha256": sha(TEST), "replay_sha256": sha(REPLAY),
            "dataset_scripts_sha256": scripts(),
            "verification_inputs_sha256": {str(p): sha(p) for p in inputs},
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "sealed_prechange_artifacts": {str(p): sha(p) for p in sealed},
        })
    elif mode == "baseline":
        assert not BASELINE.exists()
        pre = load(PRE)
        frozen(pre, "canonical_source_hashes", "canonical_config_sha256", "canonical_runner_sha256")
        result = tests(SOURCE, CONFIG, RUNNER, mode)
        frozen(pre, "canonical_source_hashes", "canonical_config_sha256", "canonical_runner_sha256")
        save(BASELINE, {**result, "prepared_sha256": sha(PRE)})
    elif mode == "candidate":
        assert not VERIFIED.exists()
        pre = load(PRE)
        frozen(pre, "canonical_source_hashes", "canonical_config_sha256", "canonical_runner_sha256")
        assert snapshot(CANDIDATE) == pre["candidate_source_hashes"]
        assert sha(NON_SOURCE_TARGETS[CONFIG]) == pre["candidate_config_sha256"]
        assert sha(NON_SOURCE_TARGETS[RUNNER]) == pre["candidate_runner_sha256"]
        result = tests(CANDIDATE, NON_SOURCE_TARGETS[CONFIG], NON_SOURCE_TARGETS[RUNNER], mode)
        prompt_replay = replay(CANDIDATE, NON_SOURCE_TARGETS[CONFIG], mode)
        frozen(pre, "canonical_source_hashes", "canonical_config_sha256", "canonical_runner_sha256")
        save(VERIFIED, {
            **pre, "prepared_sha256": sha(PRE), "baseline_sha256": sha(BASELINE),
            "candidate_tests": result, "prompt_replay": prompt_replay,
            "minirag_index_loaded": False, "minirag_search_calls": 0,
            "training_ready": False, "new_answer_accuracy": None, "new_full_rag_recall": None,
        })
    elif mode == "apply":
        assert not PREAPPLY.exists() and not APPLIED.exists() and not CANONICAL_TEST.exists()
        verified = load(VERIFIED)
        frozen(verified, "canonical_source_hashes", "canonical_config_sha256", "canonical_runner_sha256")
        assert verified["candidate_tests"]["counts"] == {
            "tests": 387, "failures": 0, "errors": 0, "skipped": 0}
        assert load(Path(verified["prompt_replay"]["path"]))["only_minirag_hint_line_removed"]
        backups = HERE / "no_minirag_original_v1"
        assert not backups.exists()
        backups.mkdir()
        applied = {}
        for relative, name in SOURCE_TARGETS.items():
            target, backup = SOURCE / relative, backups / name
            assert not target.is_symlink()
            shutil.copyfile(target, backup)
            assert sha(backup) == verified["canonical_source_hashes"][relative]
            applied["release/ASA-ArknightStoryAgent/src/" + relative] = sha(PATCH_DIR / name)
        for target, patch in NON_SOURCE_TARGETS.items():
            backup = backups / target.name
            shutil.copyfile(target, backup)
            expected = (verified["canonical_config_sha256"] if target == CONFIG
                        else verified["canonical_runner_sha256"])
            assert sha(backup) == expected
            applied[str(target.relative_to(ROOT))] = sha(patch)
        applied["release/ASA-ArknightStoryAgent/tests/test_no_minirag_runtime.py"] = sha(TEST)
        save(PREAPPLY, {
            **load(PRE), "candidate_verified_sha256": sha(VERIFIED),
            "backup_dir": str(backups), "applied_sha256": applied,
        })
        for relative, name in SOURCE_TARGETS.items():
            shutil.copyfile(PATCH_DIR / name, SOURCE / relative)
        for target, patch in NON_SOURCE_TARGETS.items():
            shutil.copyfile(patch, target)
        shutil.copyfile(TEST, CANONICAL_TEST)
        assert snapshot(SOURCE) == verified["candidate_source_hashes"]
        assert sha(CONFIG) == verified["candidate_config_sha256"]
        assert sha(RUNNER) == verified["candidate_runner_sha256"]
    else:
        assert not APPLIED.exists()
        pre = load(PREAPPLY)
        frozen(pre, "candidate_source_hashes", "candidate_config_sha256", "candidate_runner_sha256")
        assert all(sha(ROOT / p) == h for p, h in pre["applied_sha256"].items())
        result = tests(SOURCE, CONFIG, RUNNER, mode)
        prompt_replay = replay(SOURCE, CONFIG, mode)
        candidate_replay = load(HERE / "no_minirag_prompt_replay_candidate_v1.json")
        assert load(Path(prompt_replay["path"]))["results"] == candidate_replay["results"]
        frozen(pre, "candidate_source_hashes", "candidate_config_sha256", "candidate_runner_sha256")
        save(APPLIED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preapply_sha256": sha(PREAPPLY), "applied_sha256": pre["applied_sha256"],
            "canonical_tests": result, "canonical_prompt_replay": prompt_replay,
            "candidate_replay_identical": True, "minirag_enabled": False,
            "minirag_index_path": None, "minirag_search_calls_in_tests": 0,
            "minirag_prompt_hints_removed": True, "historical_minirag_code_and_index_deleted": False,
            "dense_and_sparse_retrieval_retained": True, "training_ready": False,
            "models_changed": False, "services_restarted": False, "new_training_runs": 0,
            "new_certified_answer_accuracy": None, "new_full_rag_recall": None,
        })
    print(json.dumps({"stage": mode, "done": True, "minirag_enabled": False,
                      "training_ready": False}), flush=True)


if __name__ == "__main__":
    main()
