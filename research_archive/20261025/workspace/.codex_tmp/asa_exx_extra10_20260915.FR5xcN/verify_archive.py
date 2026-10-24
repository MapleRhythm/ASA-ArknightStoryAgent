"""Verify local/remote copies against the frozen artifact manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
hashes = json.loads((root / "artifact_hashes.json").read_text())
for name, expected in hashes.items():
    path = root / name
    assert path.resolve().is_relative_to(root.resolve())
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    assert actual == expected, f"SHA mismatch: {name}"
print(json.dumps({"archive": str(root), "verified_files": len(hashes), "all_match": True}))
