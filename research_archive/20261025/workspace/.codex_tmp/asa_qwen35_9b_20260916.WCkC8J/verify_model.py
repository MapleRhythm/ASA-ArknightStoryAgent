"""Verify the existing official model in place; never alter model assets."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert not args.out.exists()
    session = requests.Session()
    session.trust_env = False
    url = "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3.5-9B/repo/files"
    response = session.get(url, params={"Revision": "master", "Recursive": "true"}, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    assert metadata["Success"]
    files = {r["Path"]: r for r in metadata["Data"]["Files"]}
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) == 4
    targets = shards + [
        "config.json", "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja",
    ]
    checks = []
    for name in targets:
        path = args.model / name
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        check = {
            "name": name, "sha256": digest, "size": path.stat().st_size,
            "official_sha256": files[name]["Sha256"], "official_size": files[name]["Size"],
            "source_revision": files[name]["Revision"],
            "passed": digest == files[name]["Sha256"] and path.stat().st_size == files[name]["Size"],
        }
        checks.append(check)
        print(json.dumps(check), flush=True)
    record = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()), "source": url, "source_model": "Qwen/Qwen3.5-9B",
        "checks": checks, "all_passed": all(c["passed"] for c in checks),
        "model_config": json.loads((args.model / "config.json").read_text()),
        "model_files_modified": False,
    }
    with args.out.open("x") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    raise SystemExit(0 if record["all_passed"] else 1)


if __name__ == "__main__":
    main()
