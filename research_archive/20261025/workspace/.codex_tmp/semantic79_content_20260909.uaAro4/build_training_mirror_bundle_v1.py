"""Freeze the canonical CPU-tested repair and its Python runtime dependency."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
PACKAGE = Path("release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent")
SCRIPTS = [
    "scripts/build_soda_api_verifier_dataset.py",
    "scripts/clean_soda_api_verifier_dataset.py",
    "scripts/merge_soda_api_verifier_shards.py",
    "scripts/soda_dataset_integrity.py",
]
TESTS = [
    "tests/test_soda_verifier_contract.py", "tests/test_soda_verifier_merge_integrity.py",
    "tests/test_soda_cleaner_integrity.py", "tests/test_soda_followup_integrity.py",
]
OLD = {
    SCRIPTS[0]: "fc426928db98380eaad6708ce6d05df23b441d86114d0e1f180ad78c96d85e4f",
    SCRIPTS[1]: "ba6ee2a97839e644c9cc0397cc1d7f0f7488066ecf0977320cef38610686dfb4",
    SCRIPTS[2]: "814a8b35bf981a38a0beaa70cce7a5b10975e7f2985c3493b94fa32b196fd5bd",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert HERE.name == "training_mirror_sync_v1.CWHPXM"
    manifest_path, bundle = HERE / "canonical_bundle_manifest_v1.json", HERE / "canonical_bundle_v1.tar.gz"
    assert not manifest_path.exists() and not bundle.exists()
    paths = [Path(p) for p in SCRIPTS + TESTS] + sorted(p.relative_to(ROOT) for p in (ROOT / PACKAGE).rglob("*.py"))
    assert len(paths) == len(set(paths)) and len(paths) > 8
    before = {}
    for rel in paths:
        path = ROOT / rel
        assert path.is_file() and not path.is_symlink()
        assert path.resolve().is_relative_to(ROOT.resolve())
        before[str(rel)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="")
    xml = HERE / "canonical_mirror_tests120_v1.xml"
    command = [sys.executable, "-m", "pytest", "-q", "-c", "/dev/null", "-p", "no:cacheprovider", "--rootdir", str(ROOT), "--junitxml", str(xml), *TESTS]
    with (HERE / "canonical_mirror_tests120_v1.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0
    suites = ET.parse(xml).getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 120
    assert all(all(int(s.attrib.get(k, 0)) == 0 for k in ("errors", "failures", "skipped")) for s in suites)
    assert all(sha(ROOT / rel) == entry["sha256"] for rel, entry in before.items())
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_host": "${REMOTE_HOST}",
        "source_root": str(ROOT), "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": before, "expected_old_files_on_248": OLD, "tests": TESTS,
        "canonical_test_count": 120, "canonical_test_xml_sha256": sha(xml),
        "scope": "Three repaired data scripts, one integrity helper, four synthetic test modules, and ASA Python package dependency only. No model/config/index/dataset transfer, no service deployment.",
    }
    with manifest_path.open("x") as stream:
        json.dump(value, stream, indent=2)
    with tarfile.open(bundle, "x:gz") as archive:
        for rel in paths:
            archive.add(ROOT / rel, arcname=str(rel), recursive=False)
        archive.add(manifest_path, arcname="canonical_bundle_manifest_v1.json", recursive=False)
    assert all(sha(ROOT / rel) == entry["sha256"] for rel, entry in before.items())
    receipt = {"bundle": str(bundle), "sha256": sha(bundle), "bytes": bundle.stat().st_size, "files": len(paths), "canonical_tests": 120, "manifest_sha256": sha(manifest_path)}
    with (HERE / "canonical_bundle_receipt_v1.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
