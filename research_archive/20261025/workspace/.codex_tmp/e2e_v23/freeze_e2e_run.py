#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path | None = None) -> dict:
    resolved = path.resolve()
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if root is not None else None,
        "resolved_path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def tree_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        records.append(file_record(path, root=root))
    return records


def aggregate(records: list[dict]) -> str:
    canonical = [
        [record["relative_path"] or record["path"], record["size"], record["sha256"]]
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "transformers", "vllm", "sentence-transformers", "faiss-cpu"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--questions-manifest", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    parser.add_argument("--reranker-model", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    retrieval = runtime["retrieval"]
    required_contract = {
        "enable_minirag": False,
        "minirag_index_path": None,
        "minirag_top_k": 0,
        "minirag_weight": 0.0,
        "minirag_mode_weights": {},
        "minirag_chapter_isolation": False,
        "minirag_auto_second_retrieval": False,
        "minirag_scope_seed_top_k": 0,
        "minirag_expansion_query_top_k": 0,
    }
    for key, expected in required_contract.items():
        actual = retrieval.get(key)
        if actual != expected:
            raise SystemExit(f"runtime MiniRAG contract mismatch: {key}={actual!r}")

    source_records = tree_records(args.release_root / "src")
    source_records.append(
        file_record(args.release_root / "scripts/run_cpu_inference.py", root=args.release_root)
    )
    index_records = tree_records(args.index_dir)
    embedding_records = tree_records(args.embedding_model)
    reranker_records = tree_records(args.reranker_model)
    base_records = tree_records(args.base_model)
    lora_records = tree_records(args.lora)

    git_head = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(args.repo), "status", "--short"],
        text=True,
    ).splitlines()

    manifest = {
        "version": "e2e_no_minirag_fixed58_baseline_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "repo": str(args.repo),
        "git_head": git_head,
        "git_dirty_entry_count": len(git_status),
        "git_dirty_entries": git_status,
        "python": sys.version,
        "packages": package_versions(),
        "questions": file_record(args.questions),
        "questions_manifest": file_record(args.questions_manifest),
        "runtime_config": file_record(args.runtime_config),
        "runtime": runtime,
        "no_minirag_contract": required_contract,
        "source": {
            "files": len(source_records),
            "aggregate_sha256": aggregate(source_records),
            "records": source_records,
        },
        "index": {
            "files": len(index_records),
            "aggregate_sha256": aggregate(index_records),
            "records": index_records,
        },
        "embedding_model": {
            "files": len(embedding_records),
            "aggregate_sha256": aggregate(embedding_records),
            "records": embedding_records,
        },
        "reranker_model": {
            "files": len(reranker_records),
            "aggregate_sha256": aggregate(reranker_records),
            "records": reranker_records,
        },
        "base_model": {
            "files": len(base_records),
            "aggregate_sha256": aggregate(base_records),
            "records": base_records,
        },
        "lora": {
            "files": len(lora_records),
            "aggregate_sha256": aggregate(lora_records),
            "records": lora_records,
        },
        "command": args.command,
        "evaluation_boundaries": [
            "58 development questions, not an unseen holdout",
            "fresh stateless E2E retrieval and generation; historical evidence states are not reused",
            "maximum two retrieval rounds",
            "MiniRAG disabled and fail-closed",
            "results require claim-level semantic review before reporting answer accuracy",
        ],
    }
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "source_sha256": manifest["source"]["aggregate_sha256"],
        "index_sha256": manifest["index"]["aggregate_sha256"],
        "base_model_sha256": manifest["base_model"]["aggregate_sha256"],
        "lora_sha256": manifest["lora"]["aggregate_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
