"""Guarded canonical copy of the tested cleaner only; no datasets are written."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
TARGETS = {
    "scripts/clean_soda_api_verifier_dataset.py": "clean_soda_api_verifier_dataset.py",
    "tests/test_soda_cleaner_integrity.py": "test_soda_cleaner_integrity.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    return {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}


def source_snapshot():
    paths = []
    for base in (ROOT / "release/ASA-ArknightStoryAgent/src", ROOT / "scripts", ROOT / "tests"):
        paths.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return {str(p.relative_to(ROOT)): sha(p) for p in paths if str(p.relative_to(ROOT)) not in TARGETS}


def write_new(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["apply", "verify"])
    args = parser.parse_args()
    manifest_path = HERE / "preapply_manifest.json"
    if args.mode == "apply":
        assert not manifest_path.exists()
        assert sha(ROOT / "scripts/clean_soda_api_verifier_dataset.py") == sha(HERE / "cleaner_original.py")
        assert sha(ROOT / "scripts/soda_dataset_integrity.py") == sha(HERE / "soda_dataset_integrity.py")
        assert not (ROOT / "tests/test_soda_cleaner_integrity.py").exists()
        assert test_counts(HERE / "baseline_tests_v1.xml") == {"tests": 20, "failures": 19, "errors": 0, "skipped": 0}
        assert test_counts(HERE / "candidate_tests_v1.xml") == {"tests": 20, "failures": 0, "errors": 0, "skipped": 0}
        replay = json.loads((HERE / "stored_cleaner_replay_v1.json").read_text())
        for dataset in replay["datasets"]:
            assert all(sha(Path(path)) == digest for path, digest in dataset["source_sha256"].items())
            assert dataset["candidate_replay"]["split"]["status"] == "rejected_integrity_error"
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "status_before": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
            "original_sha256": sha(HERE / "cleaner_original.py"),
            "candidate_sha256": {target: sha(HERE / candidate) for target, candidate in TARGETS.items()},
            "unrelated_code_sha256": source_snapshot(),
            "replay_sha256": sha(HERE / "stored_cleaner_replay_v1.json"),
            "datasets_modified": False, "training_ready": False,
        }
        write_new(manifest_path, manifest)
        for target, candidate in TARGETS.items():
            shutil.copyfile(HERE / candidate, ROOT / target)
            assert sha(ROOT / target) == manifest["candidate_sha256"][target]
        print("Applied only the tested cleaner and new regression tests; verification still required.")
    else:
        manifest = json.loads(manifest_path.read_text())
        assert all(sha(ROOT / path) == digest for path, digest in manifest["candidate_sha256"].items())
        assert source_snapshot() == manifest["unrelated_code_sha256"]
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == manifest["head"]
        replay = json.loads((HERE / "stored_cleaner_replay_v1.json").read_text())
        for dataset in replay["datasets"]:
            assert all(sha(Path(path)) == digest for path, digest in dataset["source_sha256"].items())
        tests = test_counts(HERE / "canonical_combined_tests_v2.xml")
        assert tests == {"tests": 234, "failures": 0, "errors": 0, "skipped": 0}, tests
        write_new(HERE / "applied_verified.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preapply_manifest_sha256": sha(manifest_path), "applied_sha256": manifest["candidate_sha256"],
            "canonical_tests": tests, "other_python_sources_unchanged": True,
            "initial_test_invocation": "v1 collection failed because PYTHONPATH omitted release/src; preserved separately.",
            "historical_datasets_unchanged": True, "training_ready": False,
            "committed": False, "services_restarted": False, "weights_changed": False,
        })
        print(json.dumps({"canonical_tests": tests, "training_ready": False}))


if __name__ == "__main__":
    main()
