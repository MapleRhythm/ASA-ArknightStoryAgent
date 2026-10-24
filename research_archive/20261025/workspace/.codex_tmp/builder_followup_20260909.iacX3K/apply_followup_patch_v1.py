"""Apply only the tested builder patch, preserving all unrelated dirty changes."""
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
    "scripts/build_soda_api_verifier_dataset.py": "build_soda_api_verifier_dataset_v2.py",
    "tests/test_soda_followup_integrity.py": "test_soda_followup_integrity_v2.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((HERE / name).read_text())


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def counts(name):
    xml = ET.parse(HERE / name).getroot()
    suites = [xml] if xml.tag == "testsuite" else xml.findall("testsuite")
    return {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}


def snapshot():
    paths = []
    for directory in ("release/ASA-ArknightStoryAgent/src", "scripts", "tests"):
        paths.extend((ROOT / directory).rglob("*.py"))
    return {
        str(p.relative_to(ROOT)): sha(p)
        for p in paths if str(p.relative_to(ROOT)) not in TARGETS
    }


def check_data(replay):
    assert all(sha(Path(p)) == digest for p, digest in replay["source_sha256"].items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["apply", "verify"])
    args = parser.parse_args()
    replay = load("stored_followup_replay_v2.json")
    check_data(replay)
    if args.mode == "apply":
        assert not (HERE / "preapply_manifest.json").exists()
        assert not (ROOT / "tests/test_soda_followup_integrity.py").exists()
        assert sha(ROOT / "scripts/build_soda_api_verifier_dataset.py") == replay["original_sha256"] == sha(HERE / "builder_original.py")
        assert sha(HERE / "build_soda_api_verifier_dataset_v2.py") == replay["candidate_sha256"]
        assert counts("baseline_tests_v1.xml") == {"tests": 30, "failures": 18, "errors": 0, "skipped": 0}
        assert counts("candidate_tests_v2.xml") == {"tests": 92, "failures": 0, "errors": 0, "skipped": 0}
        assert replay["all_other_outputs_unchanged"]
        assert replay["dropped_retrieve_positive_count"] == 31 and replay["changed_retrieve_positive_count"] == 29
        assert all(r["retained_response_unchanged"] for r in replay["changed"])
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "status_before": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
            "original_sha256": replay["original_sha256"],
            "candidate_sha256": {target: sha(HERE / candidate) for target, candidate in TARGETS.items()},
            "unrelated_sources_sha256": snapshot(),
            "replay_sha256": sha(HERE / "stored_followup_replay_v2.json"),
            "runtime_config_sha256": sha(ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"),
            "training_ready": False, "datasets_modified": False,
            "backup_note": "cp -p copied bytes but reported unsupported permissions on experiment filesystem; bytes were then checked with cmp and SHA256 before replay.",
        }
        save("preapply_manifest.json", manifest)
        for target, candidate in TARGETS.items():
            shutil.copyfile(HERE / candidate, ROOT / target)
            assert sha(ROOT / target) == manifest["candidate_sha256"][target]
        print("Applied builder and 42 regression tests only; combined verification pending.")
    else:
        manifest = load("preapply_manifest.json")
        assert all(sha(ROOT / p) == digest for p, digest in manifest["candidate_sha256"].items())
        assert snapshot() == manifest["unrelated_sources_sha256"]
        assert sha(ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json") == manifest["runtime_config_sha256"]
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == manifest["head"]
        tests = counts("canonical_combined_tests_v2.xml")
        assert tests == {"tests": 276, "failures": 0, "errors": 0, "skipped": 0}, tests
        save("applied_verified.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preapply_manifest_sha256": sha(HERE / "preapply_manifest.json"),
            "applied_sha256": manifest["candidate_sha256"], "canonical_tests": tests,
            "unrelated_sources_unchanged": True, "runtime_config_unchanged": True,
            "historical_data_unchanged": True, "training_ready": False,
            "initial_verification": "v1 passed 252 tests but omitted 24 existing tests; v2 includes the complete prior 234 plus 42 new tests.",
            "committed": False, "services_restarted": False, "weights_changed": False,
        })
        print(json.dumps({"canonical_tests": tests, "training_ready": False}))


if __name__ == "__main__":
    main()
