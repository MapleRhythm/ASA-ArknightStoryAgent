"""Verify and export final inference assets without changing training artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def main():
    source, dest = map(Path, sys.argv[1:])
    assert source.name == "asa_v55_a03_20260911"
    assert not (source / "failure.json").exists()
    complete = json.loads((source / "complete.json").read_text())
    assert complete["exit_code"] == 0 and not complete["production_promoted"]
    assert sha(source / "model/adapter_model.safetensors") == complete["adapter_sha256"]
    snapshot = json.loads((source / "snapshot.json").read_text())
    for rel, expected in snapshot["sha256"].items():
        assert sha(source / rel) == expected, rel
    steps = [json.loads(line) for line in (source / "steps.jsonl").read_text().splitlines()]
    assert [s["step"] for s in steps] == list(range(1, 115))
    results = json.loads((source / "model/all_results.json").read_text())
    state = json.loads((source / "model/trainer_state.json").read_text())
    assert results["epoch"] == 1.0 and state["global_step"] == 114
    import torch
    from safetensors import safe_open
    tensors = {}
    with safe_open(source / "model/adapter_model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            assert torch.isfinite(tensor).all().item(), key
            tensors[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    paths = sorted(
        p for directory in (source, source / "model", source / "data")
        for p in directory.iterdir()
        if p.is_file() and not p.is_symlink()
        and p.name not in {"startup_receipt.tgz", "training.lock"}
    )
    dest.mkdir(parents=True, exist_ok=False)
    manifest = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_host": "${REMOTE_HOST}", "source_run": str(source),
        "training_complete": complete, "results": results, "steps": len(steps),
        "tensor_count": len(tensors), "tensors_all_finite": True,
        "adapter_tensors": tensors,
        "checkpoints": "Retained on 248; this export contains final root-level inference assets.",
        "files": {str(p.relative_to(source)): {"sha256": sha(p), "bytes": p.stat().st_size}
                  for p in paths},
        "production_promoted": False,
    }
    save(dest / "manifest.json", manifest)
    archive = dest / "final_artifacts.tgz"
    with tarfile.open(archive, "x:gz", compresslevel=1) as tar:
        for p in paths:
            tar.add(p, arcname=str(p.relative_to(source)), recursive=False)
    receipt = {"archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
               "manifest_sha256": sha(dest / "manifest.json"),
               "file_count": len(paths), "tensor_count": len(tensors), "results": results}
    save(dest / "export_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
