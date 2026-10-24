"""Build a sibling release with immutable parent and unchanged model/config."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
PARENT = ROOT / "releases/exx-20260915-v1"
DEST = ROOT / "releases/exx-20260915-v4"
STAGE = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    assert (ROOT / "current-release").resolve() == PARENT
    assert not DEST.exists(), "Never overwrite an existing release"
    parent_lock = json.loads((PARENT / "RELEASE_LOCK.json").read_text())
    for relative, expected in parent_lock["files"].items():
        assert sha(PARENT / relative) == expected, relative
    DEST.mkdir()
    for name in ("src", "scripts", "configs", "tests"):
        shutil.copytree(PARENT / name, DEST / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Only promote tested files; unrelated historical changes stay untouched.
    changed = [
        "src/asa_arknight_story_agent/inference/grounding/evidence_id_validation.py",
        "src/asa_arknight_story_agent/inference/grounding/evidence_id_checks.py",
        "src/asa_arknight_story_agent/inference/grounding/partial_support.py",
        "src/asa_arknight_story_agent/inference/grounding/discourse_support.py",
        "src/asa_arknight_story_agent/inference/generation/evidence_id_repair.py",
        "src/asa_arknight_story_agent/inference/generation/answer_conclusion_generation.py",
        "tests/test_evidence_id_repair.py",
        "tests/test_discourse_support.py",
    ]
    for relative in changed:
        shutil.copy2(STAGE / relative, DEST / relative)
    for name in ("model", "indexes"):
        (DEST / name).symlink_to(PARENT / name, target_is_directory=True)
    shutil.copy2(STAGE / "FIX_PROTOCOL.md", DEST / "FIX_PROTOCOL.md")
    lock = dict(parent_lock)
    lock.update(release=DEST.name, parent_release=PARENT.name,
                quality_gate="not_certified_60_percent", change_protocol="FIX_PROTOCOL.md")
    lock["files"] = {
        str(path.relative_to(DEST)): sha(path)
        for directory in ("src", "scripts", "configs")
        for path in sorted((DEST / directory).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    lock["files"]["FIX_PROTOCOL.md"] = sha(DEST / "FIX_PROTOCOL.md")
    lock["files"]["tests/test_evidence_id_repair.py"] = sha(DEST / "tests/test_evidence_id_repair.py")
    lock["files"]["tests/test_discourse_support.py"] = sha(DEST / "tests/test_discourse_support.py")
    (DEST / "RELEASE_LOCK.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"release": str(DEST), "default_changed": False, "changed": changed}))


if __name__ == "__main__":
    main()
