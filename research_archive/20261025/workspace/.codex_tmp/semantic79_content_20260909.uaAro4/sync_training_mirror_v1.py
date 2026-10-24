"""Guarded, recoverable canonical -> training mirror sync, with CPU tests."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CANDIDATE = HERE / "candidate"
MANIFEST = CANDIDATE / "canonical_bundle_manifest_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def safe_path(root, rel):
    rel = PurePosixPath(rel)
    assert not rel.is_absolute() and ".." not in rel.parts
    path = root
    assert not root.is_symlink()
    for part in rel.parts:
        path = path / part
        assert not path.is_symlink(), f"Refusing symlink: {path}"
    return path


def protected_manifest():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src/goldenglow").rglob("*.py"))}


def validate_inputs(manifest, root):
    for rel, info in manifest["files"].items():
        path = safe_path(root, rel)
        assert path.is_file() and sha(path) == info["sha256"], f"Unexpected file: {path}"


def guard_originals(manifest):
    for rel in manifest["files"]:
        destination = safe_path(ROOT, rel)
        old = manifest["expected_old_files_on_248"].get(rel)
        if old is not None:
            assert destination.is_file() and sha(destination) == old, f"Unrecognized mirror modification: {rel}"
        else:
            assert not destination.exists(), f"Refusing existing mirror file: {rel}"


def test(root, phase, manifest):
    xml = HERE / f"{phase}_tests120_v1.xml"
    log_path = HERE / f"{phase}_tests120_v1.log"
    assert not xml.exists() and not log_path.exists()
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(HERE / "cpu_test_deps"), str(ROOT / "src")]),
        PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="",
    )
    for name in ("ASA_VERIFIER_SCRIPT", "ASA_CLEANER_SCRIPT", "ASA_MERGE_SCRIPT"):
        env.pop(name, None)
    command = [
        sys.executable, "-m", "pytest", "-q", "-c", "/dev/null", "-p", "no:cacheprovider",
        "--rootdir", str(root), "--junitxml", str(xml), *manifest["tests"],
    ]
    with log_path.open("x") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, f"Tests failed; see {log_path}"
    suites = ET.parse(xml).getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 120
    assert all(all(int(s.attrib.get(k, 0)) == 0 for k in ("failures", "errors", "skipped")) for s in suites)
    return {"xml": str(xml), "sha256": sha(xml), "tests": 120}


def stage():
    assert not CANDIDATE.exists()
    bundle = HERE / "canonical_bundle_v1.tar.gz"
    receipt = json.loads((HERE / "canonical_bundle_receipt_v1.json").read_text())
    assert sha(bundle) == receipt["sha256"]
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        manifest_member = archive.getmember("canonical_bundle_manifest_v1.json")
        content = archive.extractfile(manifest_member).read()
        assert hashlib.sha256(content).hexdigest() == receipt["manifest_sha256"]
        manifest = json.loads(content)
        assert manifest["source_host"] == "${REMOTE_HOST}" and manifest["canonical_test_count"] == 120
        assert {m.name for m in members} == set(manifest["files"]) | {"canonical_bundle_manifest_v1.json"}
        guard_originals(manifest)
        protected = protected_manifest()
        assert protected
        CANDIDATE.mkdir()
        for member in members:
            assert member.isfile() and member.size <= 2_000_000
            target = safe_path(CANDIDATE, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream, archive.extractfile(member) as source:
                shutil.copyfileobj(source, stream)
            target.chmod(member.mode & 0o777)
    validate_inputs(manifest, CANDIDATE)
    tests = test(CANDIDATE, "candidate", manifest)
    validate_inputs(manifest, CANDIDATE)
    guard_originals(manifest)
    assert protected_manifest() == protected
    save(HERE / "candidate_verified_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha(MANIFEST), "bundle_sha256": sha(bundle),
        "tests": tests, "protected_goldenglow_source": protected,
        "mirror_scripts_not_yet_modified": True,
    })
    print(json.dumps({"stage": "verified", "tests": tests, "files": len(manifest["files"])}))


def apply():
    verified = json.loads((HERE / "candidate_verified_v1.json").read_text())
    assert sha(MANIFEST) == verified["manifest_sha256"]
    assert sha(Path(verified["tests"]["xml"])) == verified["tests"]["sha256"]
    manifest = json.loads(MANIFEST.read_text())
    validate_inputs(manifest, CANDIDATE)
    guard_originals(manifest)
    assert protected_manifest() == verified["protected_goldenglow_source"]
    backup = HERE / "original_backup"
    assert not backup.exists()
    backup.mkdir()
    for rel, expected in manifest["expected_old_files_on_248"].items():
        original = safe_path(ROOT, rel)
        saved = safe_path(backup, rel)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, saved)
        assert sha(saved) == expected
    save(HERE / "preapply_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha(MANIFEST), "backup": str(backup),
        "old_files": manifest["expected_old_files_on_248"],
        "new_files": [p for p in manifest["files"] if p not in manifest["expected_old_files_on_248"]],
        "rollback": "Restore the three byte-verified backups only if current files still match candidate SHA; added paths are explicitly listed. No old data or weights were removed.",
    })
    # Add dependencies/tests before replacing entry scripts.
    ordered = sorted(manifest["files"], key=lambda r: r in manifest["expected_old_files_on_248"])
    with (HERE / "apply_journal_v1.jsonl").open("x", buffering=1) as journal:
        for rel in ordered:
            destination = safe_path(ROOT, rel)
            original_sha = manifest["expected_old_files_on_248"].get(rel)
            assert sha(destination) == original_sha if original_sha else not destination.exists()
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=".asa_canonical_sync_", dir=destination.parent)
            with os.fdopen(fd, "wb") as stream, (CANDIDATE / rel).open("rb") as source:
                shutil.copyfileobj(source, stream)
            temp = Path(temp_name)
            temp.chmod((CANDIDATE / rel).stat().st_mode & 0o777)
            assert sha(temp) == manifest["files"][rel]["sha256"]
            os.replace(temp, destination)
            journal.write(json.dumps({"path": rel, "sha256": sha(destination), "old_sha256": original_sha}) + "\n")
    validate_inputs(manifest, ROOT)
    tests = test(ROOT, "mirror_postapply", manifest)
    validate_inputs(manifest, ROOT)
    assert protected_manifest() == verified["protected_goldenglow_source"]
    save(HERE / "mirror_applied_verified_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_host": "${REMOTE_HOST}",
        "target_host": "${REMOTE_HOST}", "manifest_sha256": sha(MANIFEST),
        "files": len(manifest["files"]), "replaced": len(manifest["expected_old_files_on_248"]),
        "tests": tests, "backup": str(backup), "goldenglow_source_unchanged": True,
        "datasets_weights_configs_training_environment_or_services_modified": False,
    })
    print(json.dumps({"sync": "applied_and_verified", "files": len(manifest["files"]), "replaced": 3, "tests": tests}))


if __name__ == "__main__":
    assert HERE == Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/.canonical_sync_20260908.lxnhWO")
    assert sys.argv[1:] in (["stage"], ["apply"])
    {"stage": stage, "apply": apply}[sys.argv[1]]()
