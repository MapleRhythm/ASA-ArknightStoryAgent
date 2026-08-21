#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"


def should_use_train_overrides() -> bool:
    override_flag = os.environ.get("GOLDENGLOW_USE_TRAIN_OVERRIDE")
    if override_flag is not None:
        return override_flag.lower() in {"1", "true", "yes", "on"}
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip().lower()
    if conda_env == "train":
        return True
    executable = Path(sys.executable).as_posix().lower()
    return "/envs/train/" in executable or executable.endswith("/envs/train/bin/python")


if should_use_train_overrides():
    if TRAIN_PYTHON_OVERLAY_DIR.exists():
        sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
    if TRAIN_OVERRIDE_DIR.exists():
        sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from goldenglow.config import (
    BM25_TOKENS_PATH,
    CORPUS_METADATA_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_DIR,
    EXCEL_ROOT,
    FAISS_INDEX_PATH,
    INDEX_ROOT,
    OPERATOR_ALIAS_MAP_PATH,
    STORY_ROOT,
    SPARSE_INDEX_PATH,
    BuildConfig,
)
from goldenglow.data.story_parser import build_corpus_documents, build_operator_alias_lookup
from goldenglow.retrieval.hybrid import (
    build_domain_terms,
    serialize_sparse_bundle,
    tokenize_for_bm25,
)


DEFAULT_QWEN3_QUERY_PROMPT = (
    "Instruct: Given a question about Arknights story and lore, retrieve passages "
    "containing the evidence needed to answer it.\nQuery: "
)


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(payload)
    return records


def load_extra_documents(paths: list[Path]) -> list[dict]:
    documents: list[dict] = []
    seen_ids: set[str] = set()
    for path in paths:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"Extra documents file not found: {resolved}")
        for document in load_jsonl(resolved):
            document_id = str(document.get("id") or "").strip()
            clean_text = str(document.get("clean_text") or "").strip()
            search_text = str(document.get("search_text") or clean_text).strip()
            if not document_id or not clean_text or not search_text:
                raise ValueError(f"Extra document in {resolved} is missing id/clean_text/search_text")
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            normalized = dict(document)
            normalized["search_text"] = search_text
            documents.append(normalized)
    return documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 retrieval index.")
    parser.add_argument("--max-chars", type=int, default=420)
    parser.add_argument("--overlap-segments", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INDEX_ROOT,
        help="Write the complete index to this directory; use an absolute mounted-disk path for sidecar builds.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Optional Matryoshka output dimension, e.g. 1024 for Qwen3-Embedding-0.6B.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum document/query token length used by the embedding model.",
    )
    parser.add_argument(
        "--dense-query-prompt",
        type=str,
        default=None,
        help="Query-only instruction persisted in index_meta.json; documents are encoded without it.",
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=EMBEDDING_MODEL_DIR,
        help="Local path to the embedding model.",
    )
    parser.add_argument(
        "--extra-documents",
        type=Path,
        action="append",
        default=[],
        help="Additional JSONL documents to append before building FAISS/BM25. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        max_chars=args.max_chars,
        overlap_segments=args.overlap_segments,
        embedding_batch_size=args.batch_size,
    )

    documents = build_corpus_documents(
        story_root=STORY_ROOT,
        excel_root=EXCEL_ROOT,
        max_chars=config.max_chars,
        overlap_segments=config.overlap_segments,
    )
    extra_documents = load_extra_documents(args.extra_documents)
    documents.extend(extra_documents)
    operator_alias_map = build_operator_alias_lookup(EXCEL_ROOT)
    if not documents:
        raise RuntimeError("No story documents were parsed from the source data.")

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    documents_path = output_dir / DOCUMENTS_PATH.name
    faiss_index_path = output_dir / FAISS_INDEX_PATH.name
    bm25_tokens_path = output_dir / BM25_TOKENS_PATH.name
    sparse_index_path = output_dir / SPARSE_INDEX_PATH.name
    metadata_path = output_dir / CORPUS_METADATA_PATH.name
    aliases_path = output_dir / OPERATOR_ALIAS_MAP_PATH.name
    chunks_debug_path = output_dir / "story_chunks_preview.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(documents_path, documents)
    save_jsonl(chunks_debug_path, documents[:200])
    aliases_path.write_text(
        json.dumps(operator_alias_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    embedding_model = SentenceTransformer(str(args.embedding_model), device=args.device)
    embedding_model.max_seq_length = args.max_seq_length
    search_texts = [document["search_text"] for document in documents]
    encode_kwargs = {
        "batch_size": config.embedding_batch_size,
        "show_progress_bar": True,
        "normalize_embeddings": config.normalize_embeddings,
        "convert_to_numpy": True,
    }
    if args.embedding_dim is not None:
        encode_kwargs["truncate_dim"] = args.embedding_dim
    try:
        embeddings = embedding_model.encode(search_texts, **encode_kwargs)
    except TypeError as exc:
        if "truncate_dim" not in str(exc):
            raise
        encode_kwargs.pop("truncate_dim", None)
        embeddings = embedding_model.encode(search_texts, **encode_kwargs)
        if args.embedding_dim is not None:
            embeddings = embeddings[:, : args.embedding_dim]
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(faiss_index_path))

    tokenized_corpus = [tokenize_for_bm25(text) for text in tqdm(search_texts, desc="Tokenizing BM25")]
    with bm25_tokens_path.open("wb") as handle:
        pickle.dump(tokenized_corpus, handle)

    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError(
            "jieba is required to build the v2 sparse index; install it in the active environment"
        ) from exc
    domain_terms = build_domain_terms(documents, operator_alias_map)
    domain_tokenizer = jieba.Tokenizer()
    for term in domain_terms:
        domain_tokenizer.add_word(term, freq=10_000_000)
    sparse_bundle = serialize_sparse_bundle(
        documents,
        alias_lookup=operator_alias_map,
        cut_for_search=domain_tokenizer.cut_for_search,
    )
    with sparse_index_path.open("wb") as handle:
        pickle.dump(sparse_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    query_prompt = args.dense_query_prompt
    if query_prompt is None and "qwen3-embedding" in args.embedding_model.name.lower():
        query_prompt = DEFAULT_QWEN3_QUERY_PROMPT

    meta = {
        "documents": len(documents),
        "embedding_dim": int(embeddings.shape[1]),
        "story_root": str(STORY_ROOT),
        "embedding_model": str(args.embedding_model),
        "embedding_truncate_dim": args.embedding_dim,
        "dense_query_prompt": query_prompt or "",
        "dense_query_max_length": args.max_seq_length,
        "max_chars": config.max_chars,
        "overlap_segments": config.overlap_segments,
        "operator_aliases": len(operator_alias_map),
        "extra_documents": len(extra_documents),
        "extra_document_files": [
            str(path if path.is_absolute() else PROJECT_ROOT / path)
            for path in args.extra_documents
        ],
        "sparse_index_version": int(sparse_bundle["version"]),
        "sparse_lanes": sorted(sparse_bundle["lanes"]),
        "sparse_lane_weights": sparse_bundle["lane_weights"],
        "domain_terms": len(sparse_bundle["domain_terms"]),
    }
    metadata_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
