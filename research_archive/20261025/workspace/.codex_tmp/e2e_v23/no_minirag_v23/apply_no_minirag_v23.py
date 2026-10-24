#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


BUNDLE = Path(__file__).resolve().parent
REPO = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
BACKUP = BUNDLE / "applied_backup"

RUNNER_MAPPINGS = [
    (
        BUNDLE / "originals" / "release_run_cpu_inference.py",
        BUNDLE / "release_run_cpu_inference.py",
        REPO / "release/ASA-ArknightStoryAgent/scripts/run_cpu_inference.py",
    ),
    (
        BUNDLE / "originals" / "root_run_api_inference.py",
        BUNDLE / "root_run_api_inference.py",
        REPO / "api-mode/run_api_inference.py",
    ),
    (
        BUNDLE / "originals" / "release_run_api_inference.py",
        BUNDLE / "release_run_api_inference.py",
        REPO / "release/ASA-ArknightStoryAgent/api-mode/run_api_inference.py",
    ),
]

CONFIG_ROOTS = [
    (
        BUNDLE / "original_configs" / "root",
        BUNDLE / "configs" / "root",
        REPO / "configs",
    ),
    (
        BUNDLE / "original_configs" / "root_api",
        BUNDLE / "configs" / "root_api",
        REPO / "api-mode",
    ),
    (
        BUNDLE / "original_configs" / "release",
        BUNDLE / "configs" / "release",
        REPO / "release/ASA-ArknightStoryAgent/configs",
    ),
    (
        BUNDLE / "original_configs" / "release_api",
        BUNDLE / "configs" / "release_api",
        REPO / "release/ASA-ArknightStoryAgent/api-mode",
    ),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    mappings = list(RUNNER_MAPPINGS)
    for original_root, candidate_root, target_root in CONFIG_ROOTS:
        for candidate in sorted(candidate_root.glob("*.json")):
            original = original_root / candidate.name
            if original.exists() and original.read_bytes() != candidate.read_bytes():
                mappings.append((original, candidate, target_root / candidate.name))

    preflight = []
    for original, candidate, target in mappings:
        if not original.exists() or not candidate.exists() or not target.exists():
            raise SystemExit(f"missing apply input: {original} {candidate} {target}")
        expected = sha256(original)
        actual = sha256(target)
        if actual != expected:
            raise SystemExit(
                f"canonical drift for {target}: expected {expected}, found {actual}"
            )
        if candidate.suffix == ".py":
            compile(candidate.read_text(encoding="utf-8"), str(candidate), "exec")
        preflight.append(
            {
                "target": str(target),
                "before_sha256": actual,
                "candidate_sha256": sha256(candidate),
            }
        )

    if BACKUP.exists():
        raise SystemExit(f"backup directory already exists: {BACKUP}")
    BACKUP.mkdir(parents=True)

    applied = []
    for original, candidate, target in mappings:
        relative = target.relative_to(REPO)
        backup = BACKUP / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        atomic_copy(candidate, target)
        applied.append(
            {
                "target": str(target),
                "backup": str(backup),
                "before_sha256": sha256(original),
                "after_sha256": sha256(target),
            }
        )

    receipt = {
        "version": "no_minirag_v23",
        "repo": str(REPO),
        "preflight": preflight,
        "applied": applied,
        "count": len(applied),
    }
    receipt_path = BUNDLE / "applied_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
