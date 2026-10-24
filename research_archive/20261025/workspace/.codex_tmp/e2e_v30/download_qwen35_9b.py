"""Download the official Qwen ModelScope snapshot, checking per-file SHA/size."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent
STORE = Path("${REMOTE_ASA_ROOT}/rank_v8_coverage/models")
DEST = STORE / "Qwen3.5-9B_modelscope_20260910"
API = "https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-9B/repo/files"
BASE = "https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-9B/repo"
MANIFEST = HERE / "official_download_manifest.json"
ALLOWED = {
    ".gitattributes", "chat_template.jinja", "config.json", "configuration.json",
    "LICENSE", "merges.txt", "model.safetensors.index.json", "preprocessor_config.json",
    "README.md", "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json",
    "vocab.json",
    *{f"model.safetensors-{i:05d}-of-00004.safetensors" for i in range(1, 5)},
}


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(mode):
    assert mode in {"metadata", "weights"}
    if not MANIFEST.exists():
        opener = build_opener(ProxyHandler({}))
        url = API + "?" + urlencode({"Revision": "master", "Recursive": "true"})
        with opener.open(url, timeout=30) as response:
            raw = json.load(response)
        assert raw["Code"] == 200
        files = raw["Data"]["Files"]
        assert {f["Path"] for f in files} == ALLOWED
        assert all(len(f["Sha256"]) == 64 and len(f["Revision"]) == 40 for f in files)
        save(MANIFEST, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "official_repository": "Qwen/Qwen3.5-9B", "api_url": url,
            "per_file_revision_pinned": True, "raw_listing": raw,
            "destination": str(DEST), "total_bytes": sum(f["Size"] for f in files),
        })
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["destination"] == str(DEST)
    files = manifest["raw_listing"]["Data"]["Files"]
    assert {f["Path"] for f in files} == ALLOWED
    STORE.mkdir(parents=True, exist_ok=True)
    assert STORE.resolve() == STORE, "unexpected destination symlink"
    assert shutil.disk_usage(STORE).free > manifest["total_bytes"] + 10 * 1024**3
    DEST.mkdir(exist_ok=True)
    assert DEST.resolve() == DEST
    environment = {
        key: value for key, value in os.environ.items()
        if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }

    def fetch(item):
        path = DEST / item["Path"]
        assert path.parent == DEST and item["Path"] in ALLOWED and not path.is_symlink()
        if path.exists():
            assert path.stat().st_size == item["Size"] and sha(path) == item["Sha256"]
            return {"path": item["Path"], "sha256": item["Sha256"], "size": item["Size"],
                    "already_present_verified": True}
        partial = path.with_name(path.name + ".download")
        assert not partial.is_symlink()
        url = BASE + "?" + urlencode({"Revision": item["Revision"], "FilePath": item["Path"]})
        command = [
            "curl", "--fail", "--location", "--silent", "--show-error", "--continue-at", "-",
            "--retry", "3", "--connect-timeout", "15", "--speed-time", "60", "--speed-limit", "1024",
            "--output", str(partial), url,
        ]
        print(json.dumps({"download_start": item["Path"], "expected_bytes": item["Size"]}), flush=True)
        subprocess.run(command, env=environment, check=True)
        assert partial.stat().st_size == item["Size"], item["Path"]
        assert sha(partial) == item["Sha256"], item["Path"]
        assert not path.exists(), "refuse concurrent overwrite"
        partial.rename(path)
        print(json.dumps({"download_verified": item["Path"], "bytes": item["Size"]}), flush=True)
        return {"path": item["Path"], "sha256": item["Sha256"], "size": item["Size"],
                "already_present_verified": False}

    chosen = [f for f in files if f["Path"].endswith(".safetensors") == (mode == "weights")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(fetch, chosen))
    report = HERE / f"{mode}_download_receipt.json"
    save(report, {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": sha(MANIFEST), "files": receipts,
                  "production_pointer_changed": False})
    print(json.dumps({"mode": mode, "verified_files": len(receipts), "destination": str(DEST)}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: download_qwen35_9b.py metadata|weights")
    main(sys.argv[1])
