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

from asa_arknight_story_agent.config import (
    BASE_RERANKER_MODEL_DIR,
    EMBEDDING_MODEL_DIR,
    EVIDENCE_CHAIN_RERANKER_MODEL_DIR,
    MODEL_ROOT,
)


HF_OWNER = os.environ.get("ASA_HF_OWNER", "MapleRhythm")
DEFAULT_LORA_REPO = os.environ.get("ASA_LORA_REPO", f"{HF_OWNER}/asa-arknightstoryagent-4b-lora")
DEFAULT_GGUF_REPO = os.environ.get("ASA_GGUF_REPO", f"{HF_OWNER}/asa-arknightstoryagent-4b-gguf")
DEFAULT_RERANKER_REPO = os.environ.get("ASA_RERANKER_REPO", f"{HF_OWNER}/asa-evidence-chain-reranker")
DEFAULT_BASE_QWEN_REPO = os.environ.get("ASA_BASE_QWEN_REPO", "Qwen/Qwen3.5-4B")
DEFAULT_GGUF_FILENAME = os.environ.get("ASA_GGUF_FILENAME", "qwen3.5-4b-lora-merged-q4_k_m.gguf")
DEFAULT_LORA_DIR = MODEL_ROOT / "lora" / "asa-arknightstoryagent-4b-lora"
DEFAULT_GGUF_DIR = MODEL_ROOT / "gguf"
DEFAULT_BASE_QWEN_DIR = MODEL_ROOT / "qwen3.5-4b"


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

BASE_QWEN_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
]


def download_snapshot(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> None:
    print(f"[download] snapshot {repo_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        endpoint=os.environ["HF_ENDPOINT"],
        etag_timeout=30,
        allow_patterns=allow_patterns,
    )


def download_large_file(repo_id: str, local_dir: Path, filename: str) -> None:
    target = local_dir / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"[download] exists {target}")
        return
    print(f"[download] file {repo_id}/{filename} -> {target}")
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
    parser = argparse.ArgumentParser(description="Download ASA inference models.")
    parser.add_argument(
        "--runtime",
        choices=["base", "cpu-local", "cpu-api", "gpu", "all"],
        default="base",
        help=(
            "Download preset: base=embedding+base reranker, cpu-local=embedding+GGUF, "
            "cpu-api=embedding only, gpu=embedding+base Qwen+LoRA+fine-tuned reranker, all=everything."
        ),
    )
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
    parser.add_argument("--skip-base-qwen", action="store_true", help="Skip Qwen3.5 4B base model.")
    parser.add_argument("--skip-lora", action="store_true", help="Skip ASA 4B LoRA.")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip ASA merged GGUF.")
    parser.add_argument("--skip-finetuned-reranker", action="store_true", help="Skip ASA fine-tuned reranker.")
    parser.add_argument("--lora-repo", default=DEFAULT_LORA_REPO)
    parser.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO)
    parser.add_argument("--reranker-repo", default=DEFAULT_RERANKER_REPO)
    parser.add_argument("--base-qwen-repo", default=DEFAULT_BASE_QWEN_REPO)
    parser.add_argument("--gguf-filename", default=DEFAULT_GGUF_FILENAME)
    args = parser.parse_args()

    need_base_reranker = args.runtime in {"base", "all"}
    need_finetuned_reranker = args.runtime in {"gpu", "all"}
    need_base_qwen = args.runtime in {"gpu", "all"}
    need_lora = args.runtime in {"gpu", "all"}
    need_gguf = args.runtime in {"cpu-local", "all"}

    if not args.skip_embedding:
        download_snapshot(
            "BAAI/bge-small-zh-v1.5",
            EMBEDDING_MODEL_DIR,
            allow_patterns=EMBEDDING_ALLOW_PATTERNS,
        )
    if need_base_reranker and not args.skip_reranker:
        download_snapshot(
            "BAAI/bge-reranker-v2-m3",
            BASE_RERANKER_MODEL_DIR,
            allow_patterns=RERANKER_SMALL_FILES,
        )
        download_large_file(
            "BAAI/bge-reranker-v2-m3",
            BASE_RERANKER_MODEL_DIR,
            "model.safetensors",
        )
    if need_base_qwen and not args.skip_base_qwen:
        download_snapshot(args.base_qwen_repo, DEFAULT_BASE_QWEN_DIR, allow_patterns=BASE_QWEN_ALLOW_PATTERNS)
    if need_lora and not args.skip_lora:
        download_snapshot(args.lora_repo, DEFAULT_LORA_DIR)
    if need_finetuned_reranker and not args.skip_finetuned_reranker:
        download_snapshot(args.reranker_repo, EVIDENCE_CHAIN_RERANKER_MODEL_DIR)
    if need_gguf and not args.skip_gguf:
        download_large_file(args.gguf_repo, DEFAULT_GGUF_DIR, args.gguf_filename)


if __name__ == "__main__":
    main()
