"""Import a verified final archive to a fresh canonical incoming directory."""
import json
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone
from export_final import sha, save


def main():
    root = Path(sys.argv[1])
    receipt = json.loads((root / "export_receipt.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert sha(root / "manifest.json") == receipt["manifest_sha256"]
    archive = root / "final_artifacts.tgz"
    assert sha(archive) == receipt["archive_sha256"]
    dest = root / "verified"
    dest.mkdir(exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        assert len(members) == len(manifest["files"])
        assert {m.name for m in members} == set(manifest["files"])
        for m in members:
            p = Path(m.name)
            assert m.isfile() and not p.is_absolute() and ".." not in p.parts
            assert m.size == manifest["files"][m.name]["bytes"]
        for m in members:
            tar.extract(m, dest)
    for rel, info in manifest["files"].items():
        assert sha(dest / rel) == info["sha256"], rel
    record = {
        "at_utc": datetime.now(timezone.utc).isoformat(), "canonical_host": "${REMOTE_HOST}",
        "verified_root": str(dest), "archive_sha256": receipt["archive_sha256"],
        "verified_files": len(manifest["files"]),
        "adapter_sha256": sha(dest / "model/adapter_model.safetensors"),
        "original_complete_json_preserved": True,
        "canonical_return_complete": True, "production_promoted": False,
    }
    save(root / "canonical_import.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
