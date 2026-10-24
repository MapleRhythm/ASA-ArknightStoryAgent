"""Guarded synchronization of the verified no-MiniRAG runtime to host 248."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SOURCE = ROOT / "release/ASA-ArknightStoryAgent/src"
DEPS = ROOT / ".canonical_sync_20260908.lxnhWO/cpu_test_deps"
CANDIDATE_SOURCE = HERE / "candidate_src"
TEST = HERE / "test_no_minirag_runtime_v3.py"
CANONICAL_TEST = ROOT / "release/ASA-ArknightStoryAgent/tests/test_no_minirag_runtime.py"

V2_PREAPPLY = HERE / "no_minirag_preapply_v2.json"
V2_APPLIED = HERE / "no_minirag_applied_verified_v2.json"
V3_PREAPPLY = HERE / "no_minirag_central_preapply_v3.json"
V3_APPLIED = HERE / "no_minirag_central_applied_verified_v3.json"

SOURCE_PATCHES = {
    "asa_arknight_story_agent/config.py": "config.py",
    "asa_arknight_story_agent/inference/pipeline/retrieval_round.py":
        "pipeline_retrieval_round.py",
    "asa_arknight_story_agent/inference/retrieval/pipeline.py":
        "retrieval_pipeline.py",
    "asa_arknight_story_agent/inference/retrieval/round.py":
        "retrieval_round_function.py",
    "asa_arknight_story_agent/inference/retrieval/search.py": "search.py",
    "asa_arknight_story_agent/inference/generation/minimal_conclusion_prompt.py":
        "minimal_conclusion_prompt.py",
    "asa_arknight_story_agent/inference/generation/direct_answer_prompt.py":
        "direct_answer_prompt.py",
    "asa_arknight_story_agent/retrieval/hybrid.py":
        "release_retriever_hybrid.py",
}
EXTERNAL_PATCHES = {
    "src/goldenglow/retrieval/hybrid.py": "root_goldenglow_hybrid.py",
    "scripts/run_cpu_inference.py": "run_cpu_inference.py",
    "configs/runtime_inference.json": "configs_runtime_inference.json",
    "configs/runtime_inference_gpu.json": "configs_runtime_inference_gpu.json",
    "configs/runtime_inference_gpu_ctx10000_eval.json":
        "configs_runtime_inference_gpu_ctx10000_eval.json",
    "configs/runtime_inference_gpu_ctx10000_quote_refine.json":
        "configs_runtime_inference_gpu_ctx10000_quote_refine.json",
}
CONFIG_TARGETS = [
    ROOT / "configs/runtime_inference.json",
    ROOT / "configs/runtime_inference_gpu.json",
    ROOT / "configs/runtime_inference_gpu_ctx10000_eval.json",
    ROOT / "configs/runtime_inference_gpu_ctx10000_quote_refine.json",
]
RETRIEVER_TARGETS = [
    ROOT / "src/goldenglow/retrieval/hybrid.py",
    SOURCE / "asa_arknight_story_agent/retrieval/hybrid.py",
]
CURRENT_EXTERNAL_SHA256 = {
    "src/goldenglow/retrieval/hybrid.py":
        "4db4cea9a473749d057cda2a9bb972be820598650e1e0528e73f5e6ee1756af3",
    "scripts/run_cpu_inference.py":
        "ddbdcd2d63f220d23d300a86af4b1fcba7f7d2bd762ed98b0a92262e410f5945",
    "configs/runtime_inference.json":
        "49110e1260fc3220217c3d908d660af3c85e2ac1a55d4db6523bfe52eaa1b6d2",
    "configs/runtime_inference_gpu.json":
        "88ae770a9a9c90e869207aa9fe6120d9aee7353ab1e06581fa77c424bca14ed0",
    "configs/runtime_inference_gpu_ctx10000_eval.json":
        "e287fa4cfec17626d1668b4c4ee61fc5eddd7cf167c6e64a73b653bb2bf97592",
    "configs/runtime_inference_gpu_ctx10000_quote_refine.json":
        "88ae770a9a9c90e869207aa9fe6120d9aee7353ab1e06581fa77c424bca14ed0",
}

PRE = HERE / "mirror_prepared_v1.json"
BASELINE = HERE / "mirror_baseline_v1.json"
VERIFIED = HERE / "mirror_candidate_verified_v1.json"
PREAPPLY = HERE / "mirror_preapply_v1.json"
APPLIED = HERE / "mirror_applied_verified_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def source_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.rglob("*.py"))
    }


def external_hashes() -> dict[str, str]:
    return {relative: sha(ROOT / relative) for relative in EXTERNAL_PATCHES}


def candidate_external_hashes() -> dict[str, str]:
    return {
        relative: sha(HERE / patch)
        for relative, patch in EXTERNAL_PATCHES.items()
    }


def active_enabled_configs() -> list[str]:
    enabled = []
    for directory in (
        ROOT / "configs",
        ROOT / "release/ASA-ArknightStoryAgent/configs",
    ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            retrieval = payload.get("retrieval") if isinstance(payload, dict) else None
            if isinstance(retrieval, dict) and retrieval.get("enable_minirag") is True:
                enabled.append(str(path.relative_to(ROOT)))
    return enabled


def existing_regression_tests() -> list[Path]:
    root_tests = [
        ROOT / "tests/test_soda_cleaner_integrity.py",
        ROOT / "tests/test_soda_followup_integrity.py",
        ROOT / "tests/test_soda_verifier_contract.py",
        ROOT / "tests/test_soda_verifier_merge_integrity.py",
    ]
    release_tests = [
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_chain_score_text_binding.py",
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_empty_normalized_quote.py",
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_empty_quote_diagnostic_terms.py",
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_question_echo_grounding.py",
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_token_rejection_display.py",
    ]
    tests = root_tests + release_tests
    assert all(path.is_file() for path in tests)
    return tests


def frozen(pre: dict, source_key: str, external_key: str, test_present: bool) -> None:
    assert source_snapshot(SOURCE) == pre[source_key]
    assert external_hashes() == pre[external_key]
    assert CANONICAL_TEST.exists() is test_present
    if test_present:
        assert sha(CANONICAL_TEST) == pre["candidate_test_sha256"]
    assert all(sha(Path(path)) == digest
               for path, digest in pre["verification_inputs_sha256"].items())


def test_counts(xml: Path) -> dict[str, int]:
    suites = ET.parse(xml).getroot().findall("testsuite")
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def run_tests(
    *,
    source: Path,
    configs: list[Path],
    retrievers: list[Path],
    runner: Path,
    test: Path,
    mode: str,
) -> dict:
    xml = HERE / f"mirror_{mode}_v1.xml"
    assert not xml.exists()
    sys.path[:0] = [str(DEPS), str(source), str(ROOT / "src")]
    env_values = {
        "ASA_NO_MINIRAG_CONFIG": str(configs[1]),
        "ASA_NO_MINIRAG_CONFIGS": os.pathsep.join(str(path) for path in configs),
        "ASA_NO_MINIRAG_RETRIEVERS": os.pathsep.join(
            str(path) for path in retrievers),
        "ASA_NO_MINIRAG_RUNNER": str(runner),
        "ASA_ROOT_SRC": str(ROOT / "src"),
    }
    old_env = {key: os.environ.get(key) for key in env_values}
    os.environ.update(env_values)
    from asa_arknight_story_agent.inference.retrieval import search
    assert Path(search.__file__).resolve() == (
        source / "asa_arknight_story_agent/inference/retrieval/search.py").resolve()
    import pytest
    files = [test] if mode == "baseline" else existing_regression_tests() + [test]
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
    counts = test_counts(xml)
    assert counts["errors"] == counts["skipped"] == 0
    if mode == "baseline":
        assert code == 1
        assert counts == {"tests": 15, "failures": 12, "errors": 0, "skipped": 0}
    else:
        assert code == 0
        assert counts == {"tests": 221, "failures": 0, "errors": 0, "skipped": 0}
    return {
        "counts": counts,
        "exit_code": int(code),
        "xml_sha256": sha(xml),
        "python_version": sys.version,
        "pytest_version": pytest.__version__,
        "cpu_only": True,
    }


def main() -> None:
    mode = sys.argv[1]
    assert mode in {"prepare", "baseline", "candidate", "apply", "canonical"}
    v2_pre = load(V2_PREAPPLY)
    v2_applied = load(V2_APPLIED)
    v3_pre = load(V3_PREAPPLY)
    v3_applied = load(V3_APPLIED)
    assert v2_applied["canonical_tests"]["counts"] == {
        "tests": 387, "failures": 0, "errors": 0, "skipped": 0}
    assert v3_applied["canonical_tests"]["counts"] == {
        "tests": 390, "failures": 0, "errors": 0, "skipped": 0}
    if mode == "prepare":
        assert not any(path.exists() for path in (PRE, BASELINE, VERIFIED, PREAPPLY, APPLIED))
        assert not CANDIDATE_SOURCE.exists()
        assert not CANONICAL_TEST.exists()
        before = source_snapshot(SOURCE)
        assert before == v2_pre["canonical_source_hashes"]
        assert external_hashes() == CURRENT_EXTERNAL_SHA256
        assert active_enabled_configs() == [
            "configs/runtime_inference.json",
            "configs/runtime_inference_gpu.json",
            "configs/runtime_inference_gpu_ctx10000_eval.json",
            "configs/runtime_inference_gpu_ctx10000_quote_refine.json",
        ]
        shutil.copytree(SOURCE, CANDIDATE_SOURCE)
        for relative, patch in SOURCE_PATCHES.items():
            shutil.copyfile(HERE / patch, CANDIDATE_SOURCE / relative)
        candidate_source = source_snapshot(CANDIDATE_SOURCE)
        assert candidate_source == v3_pre["candidate_source_hashes"]
        subprocess.run([
            sys.executable, "-m", "py_compile",
            *(str(HERE / patch) for patch in SOURCE_PATCHES.values()),
            *(str(HERE / patch) for relative, patch in EXTERNAL_PATCHES.items()
              if relative.endswith(".py")),
            str(TEST),
        ], check=True)
        for path in CONFIG_TARGETS:
            json.loads((HERE / EXTERNAL_PATCHES[str(path.relative_to(ROOT))]).read_text())
        verification_inputs = [
            Path(__file__), TEST, V2_PREAPPLY, V2_APPLIED, V3_PREAPPLY, V3_APPLIED,
            *existing_regression_tests(),
            *(HERE / patch for patch in SOURCE_PATCHES.values()),
            *(HERE / patch for patch in EXTERNAL_PATCHES.values()),
        ]
        save(PRE, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_source_hashes": before,
            "candidate_source_hashes": candidate_source,
            "canonical_external_hashes": external_hashes(),
            "candidate_external_hashes": candidate_external_hashes(),
            "candidate_test_sha256": sha(TEST),
            "verification_inputs_sha256": {
                str(path): sha(path) for path in verification_inputs
            },
            "canonical_receipts_sha256": {
                str(path.name): sha(path)
                for path in (V2_PREAPPLY, V2_APPLIED, V3_PREAPPLY, V3_APPLIED)
            },
            "active_enabled_configs_before": active_enabled_configs(),
        })
    elif mode == "baseline":
        pre = load(PRE)
        frozen(pre, "canonical_source_hashes", "canonical_external_hashes", False)
        result = run_tests(
            source=SOURCE,
            configs=CONFIG_TARGETS,
            retrievers=RETRIEVER_TARGETS,
            runner=ROOT / "scripts/run_cpu_inference.py",
            test=TEST,
            mode=mode,
        )
        frozen(pre, "canonical_source_hashes", "canonical_external_hashes", False)
        save(BASELINE, {**result, "prepared_sha256": sha(PRE)})
    elif mode == "candidate":
        pre = load(PRE)
        frozen(pre, "canonical_source_hashes", "canonical_external_hashes", False)
        assert source_snapshot(CANDIDATE_SOURCE) == pre["candidate_source_hashes"]
        assert candidate_external_hashes() == pre["candidate_external_hashes"]
        candidate_configs = [
            HERE / EXTERNAL_PATCHES[str(path.relative_to(ROOT))]
            for path in CONFIG_TARGETS
        ]
        candidate_retrievers = [
            HERE / EXTERNAL_PATCHES["src/goldenglow/retrieval/hybrid.py"],
            CANDIDATE_SOURCE / "asa_arknight_story_agent/retrieval/hybrid.py",
        ]
        result = run_tests(
            source=CANDIDATE_SOURCE,
            configs=candidate_configs,
            retrievers=candidate_retrievers,
            runner=HERE / EXTERNAL_PATCHES["scripts/run_cpu_inference.py"],
            test=TEST,
            mode=mode,
        )
        frozen(pre, "canonical_source_hashes", "canonical_external_hashes", False)
        save(VERIFIED, {
            **pre,
            "prepared_sha256": sha(PRE),
            "baseline_sha256": sha(BASELINE),
            "candidate_tests": result,
            "active_enabled_configs_after_candidate": [],
            "minirag_index_load_calls": 0,
            "minirag_search_calls": 0,
        })
    elif mode == "apply":
        pre = load(VERIFIED)
        frozen(pre, "canonical_source_hashes", "canonical_external_hashes", False)
        assert pre["candidate_tests"]["counts"] == {
            "tests": 221, "failures": 0, "errors": 0, "skipped": 0}
        backup = HERE / "original_v1"
        assert not backup.exists()
        backup.mkdir()
        applied = {}
        for index, (relative, patch) in enumerate(SOURCE_PATCHES.items()):
            target = SOURCE / relative
            destination = backup / f"source_{index:02d}_{target.name}"
            shutil.copyfile(target, destination)
            applied[str(target.relative_to(ROOT))] = sha(HERE / patch)
        for index, (relative, patch) in enumerate(EXTERNAL_PATCHES.items()):
            target = ROOT / relative
            destination = backup / f"external_{index:02d}_{target.name}"
            shutil.copyfile(target, destination)
            assert sha(destination) == pre["canonical_external_hashes"][relative]
            applied[relative] = sha(HERE / patch)
        save(PREAPPLY, {
            **load(PRE),
            "candidate_verified_sha256": sha(VERIFIED),
            "backup_dir": str(backup),
            "applied_sha256": applied,
            "canonical_test_preexisting": False,
        })
        for relative, patch in SOURCE_PATCHES.items():
            shutil.copyfile(HERE / patch, SOURCE / relative)
        for relative, patch in EXTERNAL_PATCHES.items():
            shutil.copyfile(HERE / patch, ROOT / relative)
        shutil.copyfile(TEST, CANONICAL_TEST)
        assert source_snapshot(SOURCE) == pre["candidate_source_hashes"]
        assert external_hashes() == pre["candidate_external_hashes"]
        assert sha(CANONICAL_TEST) == pre["candidate_test_sha256"]
    else:
        pre = load(PREAPPLY)
        frozen(pre, "candidate_source_hashes", "candidate_external_hashes", True)
        result = run_tests(
            source=SOURCE,
            configs=CONFIG_TARGETS,
            retrievers=RETRIEVER_TARGETS,
            runner=ROOT / "scripts/run_cpu_inference.py",
            test=CANONICAL_TEST,
            mode=mode,
        )
        assert active_enabled_configs() == []
        frozen(pre, "candidate_source_hashes", "candidate_external_hashes", True)
        save(APPLIED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_241_v2_receipt_sha256": sha(V2_APPLIED),
            "canonical_241_v3_receipt_sha256": sha(V3_APPLIED),
            "preapply_sha256": sha(PREAPPLY),
            "applied_sha256": pre["applied_sha256"],
            "postapply_tests": result,
            "full_release_runtime_python_matches_241": True,
            "active_enabled_configs": [],
            "minirag_index_load_calls": 0,
            "minirag_search_calls": 0,
            "minirag_prompt_hints_removed": True,
            "historical_minirag_code_and_indexes_deleted": False,
            "dense_sparse_reranker_and_neighbor_expansion_retained": True,
            "models_or_training_data_modified": False,
            "services_restarted": False,
            "gpu_used": False,
            "training_ready": False,
            "limitations": [
                "221 CPU tests validate the mirror code/config contract; "
                "they are not a new answer-accuracy or full-RAG recall measurement."
            ],
        })
    print(json.dumps({
        "stage": mode,
        "completed": True,
        "minirag_enabled": False,
        "training_ready": False,
    }), flush=True)


if __name__ == "__main__":
    main()
