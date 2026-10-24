"""Apply the narrow tested quote fix after the paired experiment is frozen."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import CANDIDATE, RELATIVE, counts, save, snapshot

TARGETS = {
    "release/ASA-ArknightStoryAgent/src/" + RELATIVE: "quote_validation.py",
    "release/ASA-ArknightStoryAgent/tests/test_empty_normalized_quote.py": "test_empty_normalized_quote.py",
    "release/ASA-ArknightStoryAgent/tests/test_empty_quote_diagnostic_terms.py": "test_empty_quote_followup_v1.py",
}


def validate_new_replay():
    destination = HERE / "empty_quote_new116_comparison_v1.json"
    assert not destination.exists()
    post = load(HERE / "generation_postflight_v1.json")
    assert post["protocol_verified"] and post["source_config_base_and_adapter_unchanged"]
    original = HERE / "validation_v1.json"
    assert sha(original) == post["analysis_sha256"]
    replay = HERE / "empty_quote_new116_replay_v1.json"
    assert not replay.exists()
    subprocess.run([
        sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
        "--validator", "after", "--source-root", str(CANDIDATE),
        "--matrix", str(HERE / "fact_coverage_pairs116_v1.matrix.json"),
        "--predictions", str(HERE / "paired116_ctx12288_v1/predictions.json"),
        "--output", str(replay),
    ], check=True)
    before, after = load(original), load(replay)
    assert len(before["results"]) == len(after["results"]) == 116
    assert [r["id"] for r in before["results"]] == [r["id"] for r in after["results"]]
    changes = [
        {"id": a["id"], "before": a, "after": b}
        for a, b in zip(before["results"], after["results"])
        if a["status"] != b["status"] or a.get("grounded") != b.get("grounded")
    ]
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "old_analysis_sha256": sha(original), "candidate_analysis_sha256": sha(replay),
        "states": 116, "changes": changes, "all_grounded_results_unchanged": not changes,
    })
    assert not changes, "New-data changes need source review before apply"
    print("All116 new outcomes unchanged under the isolated quote fix.")


def stable_sources_without_target():
    runtime = snapshot(SOURCE)
    runtime.pop(RELATIVE)
    scripts = {
        name: sha(ROOT / "scripts" / name)
        for name in (
            "build_soda_api_verifier_dataset.py", "clean_soda_api_verifier_dataset.py",
            "merge_soda_api_verifier_shards.py", "soda_dataset_integrity.py",
        )
    }
    return {"runtime": runtime, "dataset_scripts": scripts}


def apply():
    manifest_path = HERE / "empty_quote_preapply_v1.json"
    assert not manifest_path.exists()
    verified = load(HERE / "empty_quote_candidate_verified_v1.json")
    postflight = load(HERE / "generation_postflight_v1.json")
    assert load(HERE / "empty_quote_new116_comparison_v1.json")["all_grounded_results_unchanged"]
    assert verified["all79_grounded_results_unchanged"]
    assert verified["candidate_tests"] == {"tests": 312, "failures": 0, "errors": 0, "skipped": 0}
    assert snapshot(SOURCE) == verified["canonical_source_hashes"]
    assert postflight["source_config_base_and_adapter_unchanged"]
    old_path = SOURCE / RELATIVE
    assert sha(old_path) == verified["canonical_source_hashes"][RELATIVE]
    assert sha(HERE / "quote_validation.py") == verified["candidate_source_hashes"][RELATIVE]
    for name, digest in verified["new_tests_sha256"].items():
        assert sha(HERE / name) == digest
    assert all(not (ROOT / target).exists() for target in TARGETS if "/tests/" in target)
    backup = HERE / "empty_quote_original_backup"
    assert not backup.exists()
    backup.mkdir()
    shutil.copyfile(old_path, backup / "quote_validation.py")
    assert sha(backup / "quote_validation.py") == sha(old_path)
    config = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status_before": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "old_target_sha256": sha(old_path), "backup_path": str(backup / "quote_validation.py"),
        "candidate_sha256": {target: sha(HERE / source) for target, source in TARGETS.items()},
        "unrelated_sources": stable_sources_without_target(), "config_sha256": sha(config),
        "candidate_verification_sha256": sha(HERE / "empty_quote_candidate_verified_v1.json"),
        "new116_comparison_sha256": sha(HERE / "empty_quote_new116_comparison_v1.json"),
        "training_ready": False, "service_restart": False, "models_changed": False,
    }
    save(manifest_path, manifest)
    for target, source in TARGETS.items():
        path = ROOT / target
        if "/src/" in target:
            assert sha(path) == manifest["old_target_sha256"]
        else:
            assert not path.exists()
        shutil.copyfile(HERE / source, path)
        assert sha(path) == manifest["candidate_sha256"][target]
    print("Applied one runtime file and36 regression tests; canonical tests still required.")


def verify():
    manifest = load(HERE / "empty_quote_preapply_v1.json")
    assert all(sha(ROOT / path) == digest for path, digest in manifest["candidate_sha256"].items())
    assert stable_sources_without_target() == manifest["unrelated_sources"]
    config = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    assert sha(config) == manifest["config_sha256"]
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == manifest["head"]
    assert sha(Path(manifest["backup_path"])) == manifest["old_target_sha256"]
    tests = counts(HERE / "empty_quote_canonical_tests312_v1.xml")
    assert tests == {"tests": 312, "failures": 0, "errors": 0, "skipped": 0}
    origins = load(HERE / "empty_quote_canonical_test_origins_v1.json")
    assert origins["asa_arknight_story_agent.inference.grounding.quote_validation"] == str(SOURCE / RELATIVE)
    save(HERE / "empty_quote_applied_verified_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preapply_sha256": sha(HERE / "empty_quote_preapply_v1.json"),
        "applied_sha256": manifest["candidate_sha256"], "canonical_tests": tests,
        "old79_and_new116_grounded_results_unchanged": True,
        "unrelated_sources_and_config_unchanged": True,
        "backup_path": manifest["backup_path"], "training_ready": False,
        "models_changed": False, "services_restarted": False, "committed": False,
    })
    print("Canonical312 tests passed; unrelated source/config/HEAD unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["replay", "apply", "verify"])
    args = parser.parse_args()
    {"replay": validate_new_replay, "apply": apply, "verify": verify}[args.mode]()
