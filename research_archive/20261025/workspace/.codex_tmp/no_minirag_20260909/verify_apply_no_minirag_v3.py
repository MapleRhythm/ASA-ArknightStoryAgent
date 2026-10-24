"""Guard the final MiniRAG shutdown at retriever-load and config layers."""
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


PATCH_DIR = HERE / "no_minirag_central_candidate_v3"
GPU_CONFIG = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
RUNNER = ROOT / "scripts/run_cpu_inference.py"
TEST = HERE / "test_no_minirag_runtime_v3.py"
CANONICAL_TEST = ROOT / "release/ASA-ArknightStoryAgent/tests/test_no_minirag_runtime.py"
RELEASE_HYBRID_RELATIVE = "asa_arknight_story_agent/retrieval/hybrid.py"

TARGETS = {
    ROOT / "src/goldenglow/retrieval/hybrid.py":
        PATCH_DIR / "src_goldenglow_retrieval_hybrid.py",
    SOURCE / RELEASE_HYBRID_RELATIVE:
        PATCH_DIR / "release_ASA-ArknightStoryAgent_src_asa_arknight_story_agent_retrieval_hybrid.py",
    ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_cpu_api_no_reranker.json":
        PATCH_DIR / "runtime_cpu_api_no_reranker.json",
    ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_cpu_qwen35_4b_no_reranker.json":
        PATCH_DIR / "runtime_cpu_qwen35_4b_no_reranker.json",
}
CPU_CONFIGS = [
    ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_cpu_api_no_reranker.json",
    ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_cpu_qwen35_4b_no_reranker.json",
]
RETRIEVERS = [
    ROOT / "src/goldenglow/retrieval/hybrid.py",
    SOURCE / RELEASE_HYBRID_RELATIVE,
]

PRE = HERE / "no_minirag_central_prepared_v3.json"
BASELINE = HERE / "no_minirag_central_baseline_v3.json"
VERIFIED = HERE / "no_minirag_central_candidate_verified_v3.json"
PREAPPLY = HERE / "no_minirag_central_preapply_v3.json"
APPLIED = HERE / "no_minirag_central_applied_verified_v3.json"


def all_tests(new_test: Path) -> list[Path]:
    return previous_tests() + [
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_hint_metadata_isolation.py",
        new_test,
    ]


def target_hashes(paths) -> dict[str, str]:
    return {str(path): sha(path) for path in paths}


def expected_candidate_source(before: dict[str, str]) -> dict[str, str]:
    result = dict(before)
    result[RELEASE_HYBRID_RELATIVE] = sha(TARGETS[SOURCE / RELEASE_HYBRID_RELATIVE])
    return result


def active_enabled_configs() -> list[str]:
    enabled = []
    roots = [
        ROOT / "configs",
        ROOT / "release/ASA-ArknightStoryAgent/configs",
    ]
    for directory in roots:
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            retrieval = payload.get("retrieval") if isinstance(payload, dict) else None
            if isinstance(retrieval, dict) and retrieval.get("enable_minirag") is True:
                enabled.append(str(path.relative_to(ROOT)))
    return enabled


def frozen(pre: dict, source_key: str, target_key: str, test_key: str) -> None:
    assert snapshot(SOURCE) == pre[source_key]
    assert target_hashes(TARGETS) == pre[target_key]
    assert sha(GPU_CONFIG) == pre["gpu_config_sha256"]
    assert sha(RUNNER) == pre["runner_sha256"]
    assert sha(CANONICAL_TEST) == pre[test_key]
    assert scripts() == pre["dataset_scripts_sha256"]
    assert all(sha(Path(path)) == digest
               for path, digest in pre["verification_inputs_sha256"].items())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == pre["head"]


def run_tests(
    *,
    configs: list[Path],
    retrievers: list[Path],
    test: Path,
    mode: str,
) -> dict:
    xml = HERE / f"no_minirag_central_{mode}_v3.xml"
    assert not xml.exists()
    env_values = {
        "ASA_NO_MINIRAG_CONFIG": str(GPU_CONFIG),
        "ASA_NO_MINIRAG_CONFIGS": os.pathsep.join(str(path) for path in configs),
        "ASA_NO_MINIRAG_RETRIEVERS": os.pathsep.join(str(path) for path in retrievers),
        "ASA_NO_MINIRAG_RUNNER": str(RUNNER),
        "ASA_ROOT_SRC": str(ROOT / "src"),
    }
    old_env = {key: os.environ.get(key) for key in env_values}
    os.environ.update(env_values)
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.retrieval import search
    assert Path(search.__file__).resolve() == (
        SOURCE / "asa_arknight_story_agent/inference/retrieval/search.py").resolve()
    import pytest
    files = [test] if mode == "baseline" else all_tests(test)
    try:
        code = pytest.main([
            "-q", "-p", "no:cacheprovider", "--tb=short",
            *(str(path) for path in files),
            "--junitxml=" + str(xml),
        ])
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    result = counts(xml)
    assert result["errors"] == result["skipped"] == 0
    if mode == "baseline":
        assert code == 1
        assert result == {"tests": 15, "failures": 3, "errors": 0, "skipped": 0}
    else:
        assert code == 0
        assert result == {"tests": 390, "failures": 0, "errors": 0, "skipped": 0}
    origins = {}
    for name, module in sys.modules.copy().items():
        filename = getattr(module, "__file__", None)
        if filename and (
            name == "asa_arknight_story_agent"
            or name.startswith("asa_arknight_story_agent.")
        ):
            actual = Path(filename).resolve()
            assert actual.is_relative_to(SOURCE.resolve())
            origins[name] = str(actual)
    return {
        "counts": result,
        "exit_code": int(code),
        "origins": origins,
        "xml_sha256": sha(xml),
    }


def main() -> None:
    mode = sys.argv[1]
    assert mode in {"prepare", "baseline", "candidate", "apply", "canonical"}
    if mode == "prepare":
        assert not any(path.exists() for path in (PRE, BASELINE, VERIFIED, PREAPPLY, APPLIED))
        assert PATCH_DIR.is_dir()
        assert CANONICAL_TEST.is_file()
        v2_preapply = load(HERE / "no_minirag_preapply_v2.json")
        v2_applied = load(HERE / "no_minirag_applied_verified_v2.json")
        assert v2_applied["canonical_tests"]["counts"] == {
            "tests": 387, "failures": 0, "errors": 0, "skipped": 0}
        before = snapshot(SOURCE)
        assert before == v2_preapply["candidate_source_hashes"]
        assert sha(GPU_CONFIG) == v2_preapply["candidate_config_sha256"]
        assert sha(RUNNER) == v2_preapply["candidate_runner_sha256"]
        before_targets = target_hashes(TARGETS)
        candidate_targets = {
            str(target): sha(candidate) for target, candidate in TARGETS.items()
        }
        assert set(before_targets) == set(candidate_targets)
        assert all(before_targets[path] != candidate_targets[path] for path in before_targets)
        candidate_source = expected_candidate_source(before)
        assert sorted(
            key for key in before if before[key] != candidate_source[key]
        ) == [RELEASE_HYBRID_RELATIVE]
        assert active_enabled_configs() == [
            "release/ASA-ArknightStoryAgent/configs/runtime_cpu_api_no_reranker.json",
            "release/ASA-ArknightStoryAgent/configs/runtime_cpu_qwen35_4b_no_reranker.json",
        ]
        subprocess.run(
            [sys.executable, "-m", "py_compile",
             str(TARGETS[ROOT / "src/goldenglow/retrieval/hybrid.py"]),
             str(TARGETS[SOURCE / RELEASE_HYBRID_RELATIVE]),
             str(TEST)],
            check=True,
        )
        for path in CPU_CONFIGS:
            json.loads(TARGETS[path].read_text())
        verification_inputs = [
            Path(__file__), TEST,
            HERE / "no_minirag_applied_verified_v2.json",
            HERE / "no_minirag_preapply_v2.json",
            *all_tests(TEST),
            *TARGETS.values(),
        ]
        save(PRE, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "canonical_source_hashes": before,
            "candidate_source_hashes": candidate_source,
            "canonical_target_hashes": before_targets,
            "candidate_target_hashes": candidate_targets,
            "canonical_test_sha256": sha(CANONICAL_TEST),
            "candidate_test_sha256": sha(TEST),
            "gpu_config_sha256": sha(GPU_CONFIG),
            "runner_sha256": sha(RUNNER),
            "dataset_scripts_sha256": scripts(),
            "verification_inputs_sha256": {
                str(path): sha(path) for path in verification_inputs
            },
            "v2_applied_sha256": sha(HERE / "no_minirag_applied_verified_v2.json"),
            "active_enabled_configs_before": active_enabled_configs(),
        })
    elif mode == "baseline":
        pre = load(PRE)
        frozen(pre, "canonical_source_hashes", "canonical_target_hashes",
               "canonical_test_sha256")
        result = run_tests(
            configs=[GPU_CONFIG, *CPU_CONFIGS],
            retrievers=RETRIEVERS,
            test=TEST,
            mode=mode,
        )
        frozen(pre, "canonical_source_hashes", "canonical_target_hashes",
               "canonical_test_sha256")
        save(BASELINE, {**result, "prepared_sha256": sha(PRE)})
    elif mode == "candidate":
        pre = load(PRE)
        frozen(pre, "canonical_source_hashes", "canonical_target_hashes",
               "canonical_test_sha256")
        assert target_hashes(TARGETS.values()) == {
            str(path): digest
            for path, digest in zip(TARGETS.values(), pre["candidate_target_hashes"].values())
        }
        candidate_configs = [
            GPU_CONFIG,
            *(TARGETS[path] for path in CPU_CONFIGS),
        ]
        candidate_retrievers = [
            TARGETS[path] for path in RETRIEVERS
        ]
        result = run_tests(
            configs=candidate_configs,
            retrievers=candidate_retrievers,
            test=TEST,
            mode=mode,
        )
        frozen(pre, "canonical_source_hashes", "canonical_target_hashes",
               "canonical_test_sha256")
        save(VERIFIED, {
            **pre,
            "prepared_sha256": sha(PRE),
            "baseline_sha256": sha(BASELINE),
            "candidate_tests": result,
            "central_minirag_load_calls": 0,
            "active_enabled_configs_after_candidate": [],
        })
    elif mode == "apply":
        verified = load(VERIFIED)
        frozen(verified, "canonical_source_hashes", "canonical_target_hashes",
               "canonical_test_sha256")
        assert verified["candidate_tests"]["counts"] == {
            "tests": 390, "failures": 0, "errors": 0, "skipped": 0}
        backups = HERE / "no_minirag_central_original_v3"
        assert not backups.exists()
        backups.mkdir()
        applied = {}
        for index, (target, candidate) in enumerate(TARGETS.items()):
            backup = backups / f"{index:02d}_{target.name}"
            shutil.copyfile(target, backup)
            assert sha(backup) == verified["canonical_target_hashes"][str(target)]
            applied[str(target.relative_to(ROOT))] = sha(candidate)
        test_backup = backups / "test_no_minirag_runtime.v2.py"
        shutil.copyfile(CANONICAL_TEST, test_backup)
        assert sha(test_backup) == verified["canonical_test_sha256"]
        applied[str(CANONICAL_TEST.relative_to(ROOT))] = sha(TEST)
        save(PREAPPLY, {
            **load(PRE),
            "candidate_verified_sha256": sha(VERIFIED),
            "backup_dir": str(backups),
            "applied_sha256": applied,
        })
        for target, candidate in TARGETS.items():
            shutil.copyfile(candidate, target)
        shutil.copyfile(TEST, CANONICAL_TEST)
        assert snapshot(SOURCE) == verified["candidate_source_hashes"]
        assert target_hashes(TARGETS) == verified["candidate_target_hashes"]
        assert sha(CANONICAL_TEST) == verified["candidate_test_sha256"]
    else:
        pre = load(PREAPPLY)
        frozen(pre, "candidate_source_hashes", "candidate_target_hashes",
               "candidate_test_sha256")
        assert all(sha(ROOT / path) == digest
                   for path, digest in pre["applied_sha256"].items())
        result = run_tests(
            configs=[GPU_CONFIG, *CPU_CONFIGS],
            retrievers=RETRIEVERS,
            test=CANONICAL_TEST,
            mode=mode,
        )
        assert active_enabled_configs() == []
        frozen(pre, "candidate_source_hashes", "candidate_target_hashes",
               "candidate_test_sha256")
        save(APPLIED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preapply_sha256": sha(PREAPPLY),
            "applied_sha256": pre["applied_sha256"],
            "canonical_tests": result,
            "active_enabled_configs": [],
            "central_minirag_load_calls": 0,
            "minirag_search_calls": 0,
            "minirag_prompt_hints_removed": True,
            "historical_minirag_code_and_indexes_deleted": False,
            "dense_sparse_reranker_and_neighbor_expansion_retained": True,
            "services_restarted": False,
            "new_training_runs": 0,
            "new_certified_answer_accuracy": None,
            "new_full_rag_recall": None,
        })
    print(json.dumps({
        "stage": mode,
        "done": True,
        "minirag_enabled": False,
        "training_ready": False,
    }), flush=True)


if __name__ == "__main__":
    main()
