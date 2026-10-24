#!/usr/bin/env python3
"""Restore exact pre-publication files; refuse to overwrite subsequent changes."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    backup = ROOT / "outputs/release_backups/exx-20260915-v1"
    record = json.loads((backup / "publication.json").read_text())
    for item in record["changes"]:
        if sha(ROOT / item["path"]) != item["after"]:
            raise SystemExit(f"Changed since publication; refusing rollback: {item['path']}")
    saved = backup / "deactivated"
    saved.mkdir(exist_ok=False)
    for item in record["changes"]:
        target = ROOT / item["path"]
        if item["before"] is None:
            destination = saved / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(destination))
        else:
            original = backup / item["path"]
            assert sha(original) == item["before"]
            temporary = target.with_name(target.name + ".rollback-tmp")
            assert not temporary.exists()
            shutil.copy2(original, temporary)
            os.replace(temporary, target)
    os.replace(ROOT / "current-release", saved / "current-release")
    print("Restored pre-release defaults. New release, evaluation and backups are retained.")


if __name__ == "__main__":
    main()
