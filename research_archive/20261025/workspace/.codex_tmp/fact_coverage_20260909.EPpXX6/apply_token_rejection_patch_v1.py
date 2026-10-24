"""Apply/verify the tested token-rejection display repair, preserving dirty work."""
from datetime import datetime, timezone
import shutil
import subprocess
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import counts, save, snapshot
from verify_token_rejection_candidate_v1 import RELATIVE

TARGETS = {
    "release/ASA-ArknightStoryAgent/src/" + RELATIVE: "token_checks.py",
    "release/ASA-ArknightStoryAgent/tests/test_token_rejection_display.py": "test_token_rejection_display.py",
}
CONFIG = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"


def scripts():
    return {
        name: sha(ROOT / "scripts" / name)
        for name in (
            "build_soda_api_verifier_dataset.py", "clean_soda_api_verifier_dataset.py",
            "merge_soda_api_verifier_shards.py", "soda_dataset_integrity.py",
        )
    }


def main():
    mode = sys.argv[1]
    assert mode in {"apply", "verify"}
    verified = load(HERE / "token_rejection_candidate_verified_v1.json")
    preapply = HERE / "token_rejection_preapply_v1.json"
    if mode == "apply":
        assert not preapply.exists()
        assert verified["candidate_tests"] == {"tests": 325, "failures": 0, "errors": 0, "skipped": 0}
        assert verified["fixed79"]["changes"] == []
        assert len(verified["new116"]["changes"]) == 1
        changed = verified["new116"]["changes"][0]
        assert changed["id"].startswith("e9722207725f6eb8__")
        assert changed["id"].endswith("fact_coverage_prompt_v1") and changed["action_unchanged"]
        assert snapshot(SOURCE) == verified["canonical_source_hashes"]
        assert sha(HERE / "token_checks.py") == verified["candidate_source_hashes"][RELATIVE]
        assert sha(HERE / "test_token_rejection_display.py") == verified["test_sha256"]
        test_target = ROOT / "release/ASA-ArknightStoryAgent/tests/test_token_rejection_display.py"
        assert not test_target.exists()
        backup = HERE / "token_checks.original.py"
        assert not backup.exists()
        shutil.copyfile(SOURCE / RELATIVE, backup)
        assert sha(backup) == verified["canonical_source_hashes"][RELATIVE]
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "old_target_sha256": sha(backup), "backup_path": str(backup),
            "candidate_sha256": {path: sha(HERE / name) for path, name in TARGETS.items()},
            "candidate_source_hashes": verified["candidate_source_hashes"],
            "dataset_scripts_sha256": scripts(), "config_sha256": sha(CONFIG),
            "candidate_verified_sha256": sha(HERE / "token_rejection_candidate_verified_v1.json"),
            "training_ready": False,
        }
        save(preapply, manifest)
        for path, name in TARGETS.items():
            target = ROOT / path
            if "/src/" in path:
                assert sha(target) == manifest["old_target_sha256"]
            else:
                assert not target.exists()
            shutil.copyfile(HERE / name, target)
            assert sha(target) == manifest["candidate_sha256"][path]
        print("Applied token rejection display and13 tests; canonical verification pending.")
    else:
        manifest = load(preapply)
        assert all(sha(ROOT / path) == digest for path, digest in manifest["candidate_sha256"].items())
        assert snapshot(SOURCE) == manifest["candidate_source_hashes"]
        assert scripts() == manifest["dataset_scripts_sha256"] and sha(CONFIG) == manifest["config_sha256"]
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == manifest["head"]
        tests = counts(HERE / "token_rejection_canonical_tests325_v1.xml")
        assert tests == {"tests": 325, "failures": 0, "errors": 0, "skipped": 0}
        save(HERE / "token_rejection_applied_verified_v1.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preapply_sha256": sha(preapply), "applied_sha256": manifest["candidate_sha256"],
            "canonical_tests": tests, "unrelated_source_and_config_unchanged": True,
            "fixed79_unchanged": True, "new116_changed_refusals": 1,
            "all_direct_answers_and_actions_unchanged": True,
            "backup_path": manifest["backup_path"], "training_ready": False,
            "committed": False, "services_restarted": False, "models_changed": False,
        })
        print("Canonical325 tests passed; only the observed unsafe refusal display changed.")


if __name__ == "__main__":
    main()
