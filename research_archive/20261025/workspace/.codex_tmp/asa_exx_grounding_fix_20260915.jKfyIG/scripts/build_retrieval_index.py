#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asa_arknight_story_agent.config import (
    BM25_TOKENS_PATH,
    CHUNKS_DEBUG_PATH,
    CORPUS_METADATA_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_DIR,
    EXCEL_ROOT,
    FAISS_INDEX_PATH,
    INDEX_ROOT,
    OPERATOR_ALIAS_MAP_PATH,
    STORY_ROOT,
    BuildConfig,
)
from asa_arknight_story_agent.data.story_parser import build_corpus_documents, build_operator_alias_lookup


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

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    from asa_arknight_story_agent.retrieval.hybrid import tokenize_for_bm25

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

    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNKS_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(DOCUMENTS_PATH, documents)
    save_jsonl(CHUNKS_DEBUG_PATH, documents[:200])
    OPERATOR_ALIAS_MAP_PATH.write_text(
        json.dumps(operator_alias_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    embedding_model = SentenceTransformer(str(args.embedding_model), device=args.device)
    search_texts = [document["search_text"] for document in documents]
    embeddings = embedding_model.encode(
        search_texts,
        batch_size=config.embedding_batch_size,
        show_progress_bar=True,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    tokenized_corpus = [tokenize_for_bm25(text) for text in tqdm(search_texts, desc="Tokenizing BM25")]
    with BM25_TOKENS_PATH.open("wb") as handle:
        pickle.dump(tokenized_corpus, handle)

    meta = {
        "documents": len(documents),
        "embedding_dim": int(embeddings.shape[1]),
        "story_root": str(STORY_ROOT),
        "embedding_model": str(args.embedding_model),
        "max_chars": config.max_chars,
        "overlap_segments": config.overlap_segments,
        "operator_aliases": len(operator_alias_map),
        "extra_documents": len(extra_documents),
        "extra_document_files": [
            str(path if path.is_absolute() else PROJECT_ROOT / path)
            for path in args.extra_documents
        ],
    }
    CORPUS_METADATA_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
