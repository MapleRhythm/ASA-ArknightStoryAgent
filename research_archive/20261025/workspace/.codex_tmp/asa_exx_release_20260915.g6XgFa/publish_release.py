"""Atomic official-pointer publication with exact preimages and rollback files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    root, stage = args.canonical.resolve(), args.staging.resolve()
    release = root / "releases/exx-20260915-v1"
    python = "${REMOTE_PYTHON}"
    subprocess.run([python, str(release / "scripts/run_exx_official.py"), "--verify-only"], check=True)
    assert sha(root / "scripts/run_cpu_inference.py") == sha(stage / "canonical_cli.original.py")
    backup = root / "outputs/release_backups/exx-20260915-v1"
    backup.mkdir(parents=True, exist_ok=False)
    files = {
        "scripts/run_cpu_inference.py": stage / "canonical_cli.updated.py",
        "scripts/exx_official_dispatch.py": stage / "exx_official_dispatch.py",
        "scripts/run_exx_official.py": stage / "canonical_official_entry.py",
        "scripts/run_exx_official.sh": stage / "canonical_launcher.sh",
        "scripts/run_latest_kto_interactive.sh": stage / "canonical_launcher.sh",
        "scripts/rollback_exx_20260915.py": stage / "rollback_release.py",
        "docs/OFFICIAL_EXX_RELEASE_20260915.md": stage / "OFFICIAL_EXX_RELEASE.md",
        "configs/runtime_inference.json": release / "configs/runtime_exx_official.json",
        "configs/runtime_inference_gpu.json": release / "configs/runtime_exx_official.json",
        "configs/runtime_exx_official.json": release / "configs/runtime_exx_official.json",
    }
    readme = stage / "canonical_README.updated.md"
    assert sha(root / "README.md") == sha(stage / "canonical_README.original.md")
    files["README.md"] = readme
    record = {"published_utc": datetime.now(timezone.utc).isoformat(), "release": str(release),
              "backup": str(backup), "changes": [], "public_git_or_hf_push": False}
    for relative in files:
        destination = root / relative
        if destination.exists():
            old = backup / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, old)
        record["changes"].append({"path": relative,
                                  "before": sha(destination) if destination.exists() else None,
                                  "after": sha(files[relative])})
    pointer = root / "current-release"
    assert not pointer.exists() and not pointer.is_symlink(), "Existing release pointer requires explicit migration."
    for relative, source in files.items():
        destination = root / relative
        temporary = destination.with_name(destination.name + ".exx-publish-tmp")
        assert not temporary.exists()
        shutil.copy2(source, temporary)
        if destination.suffix == ".sh":
            temporary.chmod(0o755)
        os.replace(temporary, destination)
    # Publish pointer last: prior partial installs fail closed instead of loading the wrong model.
    temporary_pointer = root / ".current-release-exx-20260915-v1"
    temporary_pointer.symlink_to("releases/exx-20260915-v1", target_is_directory=True)
    os.replace(temporary_pointer, pointer)
    (backup / "publication.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    subprocess.run([python, str(root / "scripts/run_cpu_inference.py"), "--verify-only"], check=True)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
