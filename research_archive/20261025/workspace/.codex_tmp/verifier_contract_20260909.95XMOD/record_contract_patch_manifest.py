"""Guarded provenance capture; does not itself install source or change data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
HERE = Path(__file__).resolve().parent
FILES = {
    "scripts/build_soda_api_verifier_dataset.py": ("build_soda_api_verifier_dataset.py", "original.py"),
    "scripts/merge_soda_api_verifier_shards.py": ("merge_soda_api_verifier_shards.py", "merge_original.py"),
    "scripts/soda_dataset_integrity.py": ("soda_dataset_integrity.py", None),
    "tests/test_soda_verifier_contract.py": ("test_soda_verifier_contract.py", None),
    "tests/test_soda_verifier_merge_integrity.py": ("test_soda_verifier_merge_integrity.py", None),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode()


def tests(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    return {field: sum(int(s.attrib.get(field, 0)) for s in suites)
            for field in ("tests", "failures", "errors", "skipped")}


def stamp():
    return datetime.now(timezone.utc).isoformat()


stage = sys.argv[1]
assert git("rev-parse", "HEAD").strip() == "0d594a3c9fd641fd8d7f798c8ca6b07db6c7020b"
if stage == "before":
    counts = tests(HERE / "candidate_combined_tests_v3.xml")
    assert counts == {"tests": 58, "failures": 0, "errors": 0, "skipped": 0}
    files = {}
    for relative, (candidate, original) in FILES.items():
        path = ROOT / relative
        if original is None:
            assert not path.exists(), relative
        else:
            assert sha(path) == sha(HERE / original), f"Source drift: {relative}"
        files[relative] = {
            "candidate_sha256": sha(HERE / candidate),
            "original_sha256": sha(HERE / original) if original else None,
        }
    dirty = {
        relative: sha(ROOT / relative)
        for relative in git("diff", "--name-only").splitlines()
        if (ROOT / relative).is_file()
    }
    replay = json.loads((HERE / "stored_judgment_replay_v2.json").read_text())
    for record in replay["source_files"]:
        assert sha(Path(record["path"])) == record["sha256"]
    payload = {
        "timestamp_utc": stamp(), "head": git("rev-parse", "HEAD").strip(),
        "files": files, "prior_dirty_file_sha256": dirty,
        "git_status": git("status", "--short"), "candidate_tests": counts,
        "frozen_audit_sources": replay["source_files"],
        "replay_report_sha256": sha(HERE / "stored_judgment_replay_v2.json"),
    }
    target = HERE / "preapply_manifest.json"
elif stage == "after":
    before = json.loads((HERE / "preapply_manifest.json").read_text())
    for relative, info in before["files"].items():
        assert sha(ROOT / relative) == info["candidate_sha256"], f"Installed hash mismatch: {relative}"
    for relative, digest in before["prior_dirty_file_sha256"].items():
        if relative not in FILES:
            assert sha(ROOT / relative) == digest, f"Unrelated dirty content changed: {relative}"
    for record in before["frozen_audit_sources"]:
        assert sha(Path(record["path"])) == record["sha256"]
    counts = tests(HERE / "canonical_combined_tests_v1.xml")
    assert counts["tests"] >= 173 and counts["failures"] == counts["errors"] == counts["skipped"] == 0
    payload = {
        "timestamp_utc": stamp(), "head": git("rev-parse", "HEAD").strip(),
        "installed_files": before["files"], "canonical_tests": counts,
        "unrelated_prior_dirty_files_unchanged": True,
        "frozen_audit_sources_unchanged": True,
        "training_started": False, "model_changed": False, "services_restarted": False,
        "training_ready": False, "git_status": git("status", "--short"),
    }
    target = HERE / "applied_verified.json"
else:
    raise ValueError(stage)
with target.open("x", encoding="utf-8") as output:
    json.dump(payload, output, ensure_ascii=False, indent=2)
print(json.dumps({"manifest": str(target), "stage": stage,
                  "tests": payload.get("canonical_tests", payload.get("candidate_tests"))}))
