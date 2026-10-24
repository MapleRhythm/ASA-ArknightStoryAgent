"""Restore exact pre-v4 docs and v1 pointer; refuse subsequent modifications."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
BACKUP = ROOT / "outputs/release_backups/exx-20260915-v4"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    record = json.loads((BACKUP / "publication.json").read_text())
    assert record["old"] == "releases/exx-20260915-v1"
    assert record["new"] == "releases/exx-20260915-v4"
    assert os.readlink(ROOT / "current-release") == record["new"]
    for change in record["changes"]:
        assert change["path"] in {"README.md", "docs/OFFICIAL_EXX_RELEASE_20260915.md"}
        assert sha(ROOT / change["path"]) == change["after"], "Later edits detected"
        assert sha(BACKUP / change["path"]) == change["before"], "Backup changed"
    for change in record["changes"]:
        path = ROOT / change["path"]
        temporary = path.with_name(path.name + ".exx-v4-rollback")
        assert not temporary.exists()
        shutil.copy2(BACKUP / change["path"], temporary)
        os.replace(temporary, path)
    temporary_pointer = ROOT / "current-release.exx-v4-rollback"
    temporary_pointer.symlink_to(record["old"])
    os.replace(temporary_pointer, ROOT / "current-release")
    print(json.dumps({"restored": record["old"], "experiments_and_models_preserved": True}))


if __name__ == "__main__":
    main()
