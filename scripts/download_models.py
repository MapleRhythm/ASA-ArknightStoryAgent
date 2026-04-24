#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

from goldenglow.config import EMBEDDING_MODEL_DIR, RERANKER_MODEL_DIR


EMBEDDING_ALLOW_PATTERNS = [
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
]

RERANKER_SMALL_FILES = [
    "config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]


def download_snapshot(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        endpoint=os.environ["HF_ENDPOINT"],
        etag_timeout=30,
        allow_patterns=allow_patterns,
    )


def download_large_file(repo_id: str, local_dir: Path, filename: str) -> None:
    aria2c = shutil.which("aria2c")
    if not aria2c:
        download_snapshot(repo_id, local_dir, allow_patterns=[filename])
        return

    url = f"{os.environ['HF_ENDPOINT'].rstrip('/')}/{repo_id}/resolve/main/{filename}"
    subprocess.run(
        [
            aria2c,
            "-c",
            "-x",
            "16",
            "-s",
            "16",
            "-k",
            "1M",
            "--max-tries=0",
            "--timeout=60",
            "--summary-interval=30",
            "--dir",
            str(local_dir),
            "--out",
            filename,
            url,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local embedding and reranker models.")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip downloading BAAI/bge-small-zh-v1.5.",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Skip downloading BAAI/bge-reranker-v2-m3.",
    )
    args = parser.parse_args()

    if not args.skip_embedding:
        download_snapshot(
            "BAAI/bge-small-zh-v1.5",
            EMBEDDING_MODEL_DIR,
            allow_patterns=EMBEDDING_ALLOW_PATTERNS,
        )
    if not args.skip_reranker:
        download_snapshot(
            "BAAI/bge-reranker-v2-m3",
            RERANKER_MODEL_DIR,
            allow_patterns=RERANKER_SMALL_FILES,
        )
        download_large_file(
            "BAAI/bge-reranker-v2-m3",
            RERANKER_MODEL_DIR,
            "model.safetensors",
        )


if __name__ == "__main__":
    main()
