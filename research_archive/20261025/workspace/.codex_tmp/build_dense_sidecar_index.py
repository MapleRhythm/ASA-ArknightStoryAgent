#!/usr/bin/env python3
"""Build a dense-only FAISS sidecar over an existing frozen ASA corpus.

The documents and sparse indexes are linked from the source index so an
embedding ablation changes only dense document vectors and query encoding.
All generated model-dependent files and logs should live on the mounted disk.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


QUERY_PROMPT = (
    "Instruct: Given a question about Arknights story and lore, retrieve passages "
    "containing the evidence needed to answer it.\nQuery: "
)
SHARED_FILES = (
    "documents.jsonl",
    "bm25_tokens.pkl",
    "sparse_index.pkl",
    "operator_aliases.json",
)


def iter_search_texts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("search_text") or row.get("clean_text") or "").strip()
            if not text:
                raise ValueError(f"empty search text at {path}:{line_number}")
            yield text


def encode_in_chunks(
    model: SentenceTransformer,
    search_texts: list[str],
    *,
    batch_size: int,
    embedding_dim: int,
    staging_dir: Path,
    overwrite_dense: bool,
) -> np.ndarray:
    """Encode without keeping full-length model outputs for the whole corpus.

    Qwen3-Embedding-4B emits 2560-d vectors before Matryoshka truncation.  The
    previous single ``model.encode`` call retained all 54k full vectors and a
    second 1024-d copy at once.  Chunking bounds host RAM while preserving the
    exact normalized 1024-d index vectors.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    legacy_staging = next(staging_dir.glob("embeddings.*.f32.mmap"), None)
    output_path = legacy_staging or (staging_dir / "embeddings.f32.mmap")
    progress_path = staging_dir / "embeddings.progress.json"
    expected_bytes = len(search_texts) * embedding_dim * np.dtype(np.float32).itemsize
    start_offset = 0
    if overwrite_dense:
        output_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
    if output_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            int(progress.get("records", -1)) == len(search_texts)
            and int(progress.get("embedding_dim", -1)) == embedding_dim
            and output_path.stat().st_size == expected_bytes
        ):
            start_offset = int(progress.get("encoded", 0))
            if not 0 <= start_offset <= len(search_texts):
                start_offset = 0
    elif output_path.exists() and output_path.stat().st_size == expected_bytes:
        # Migrate the pre-resume chunked builder.  Its log prints only after a
        # whole chunk has been flushed to the mmap, so this explicit operator
        # value is required once during migration rather than inferred here.
        start_offset = safe_resume_offset(staging_dir, len(search_texts))
    if not output_path.exists() or output_path.stat().st_size != expected_bytes:
        output_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        descriptor = os.open(output_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.ftruncate(descriptor, expected_bytes)
        os.close(descriptor)
        start_offset = 0
    output = np.memmap(
        output_path,
        mode="r+",
        dtype=np.float32,
        shape=(len(search_texts), embedding_dim),
    )
    if start_offset:
        print(f"resuming at {start_offset}/{len(search_texts)}", flush=True)
    chunk_size = max(batch_size, min(2048, batch_size * 64))
    base_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "show_progress_bar": False,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "truncate_dim": embedding_dim,
    }
    truncate_supported = True
    for start in range(start_offset, len(search_texts), chunk_size):
        stop = min(start + chunk_size, len(search_texts))
        encode_kwargs = dict(base_kwargs)
        if not truncate_supported:
            encode_kwargs.pop("truncate_dim", None)
        try:
            chunk = model.encode(search_texts[start:stop], **encode_kwargs)
        except TypeError as exc:
            if "truncate_dim" not in str(exc):
                raise
            truncate_supported = False
            encode_kwargs.pop("truncate_dim", None)
            chunk = model.encode(search_texts[start:stop], **encode_kwargs)
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape[1] != embedding_dim:
            chunk = chunk[:, :embedding_dim]
            chunk /= np.linalg.norm(chunk, axis=1, keepdims=True).clip(min=1e-12)
        if chunk.shape != (stop - start, embedding_dim):
            raise ValueError(
                f"unexpected embedding chunk shape {chunk.shape}; "
                f"expected {(stop - start, embedding_dim)}"
            )
        output[start:stop] = chunk
        output.flush()
        temporary_progress = progress_path.with_suffix(".json.tmp")
        temporary_progress.write_text(
            json.dumps(
                {
                    "records": len(search_texts),
                    "embedding_dim": embedding_dim,
                    "encoded": stop,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_progress, progress_path)
        print(f"encoded {stop}/{len(search_texts)}", flush=True)
    return output


def safe_resume_offset(staging_dir: Path, records: int) -> int:
    raw = os.environ.get("ASA_DENSE_RESUME_OFFSET", "").strip()
    if not raw:
        raise RuntimeError(
            "existing embedding mmap has no progress manifest; set "
            "ASA_DENSE_RESUME_OFFSET to a previously logged completed chunk"
        )
    offset = int(raw)
    if not 0 <= offset <= records:
        raise ValueError(f"invalid ASA_DENSE_RESUME_OFFSET={offset} for records={records}")
    return offset


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing path: {path}")
    path.symlink_to(target.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--query-prompt", default=QUERY_PROMPT)
    parser.add_argument("--overwrite-dense", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.source_index = args.source_index.resolve()
    args.output_dir = args.output_dir.resolve()
    args.embedding_model = args.embedding_model.resolve()
    if args.output_dir == args.source_index:
        raise ValueError("output index must differ from source index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in SHARED_FILES:
        source = args.source_index / name
        if not source.exists():
            raise FileNotFoundError(source)
        replace_symlink(args.output_dir / name, source)

    output_index = args.output_dir / "faiss.index"
    output_meta = args.output_dir / "index_meta.json"
    if output_index.exists() and not args.overwrite_dense:
        raise FileExistsError(f"refusing to overwrite {output_index}")

    search_texts = list(iter_search_texts(args.source_index / "documents.jsonl"))
    model = SentenceTransformer(str(args.embedding_model), device=args.device)
    model.max_seq_length = args.max_seq_length
    embeddings = encode_in_chunks(
        model,
        search_texts,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        staging_dir=args.output_dir,
        overwrite_dense=args.overwrite_dense,
    )
    if embeddings.shape != (len(search_texts), args.embedding_dim):
        raise ValueError(
            f"unexpected embedding shape {embeddings.shape}; "
            f"expected {(len(search_texts), args.embedding_dim)}"
        )
    index = faiss.IndexFlatIP(args.embedding_dim)
    index.add(embeddings)
    temporary_index = args.output_dir / "faiss.index.tmp"
    faiss.write_index(index, str(temporary_index))
    os.replace(temporary_index, output_index)
    if isinstance(embeddings, np.memmap):
        embeddings.flush()
        staging_path = Path(embeddings.filename)
        del embeddings
        staging_path.unlink(missing_ok=True)
        staging_path.with_name("embeddings.progress.json").unlink(missing_ok=True)

    source_meta = json.loads((args.source_index / "index_meta.json").read_text(encoding="utf-8"))
    source_meta.update(
        {
            "embedding_model": str(args.embedding_model),
            "embedding_dim": args.embedding_dim,
            "embedding_truncate_dim": args.embedding_dim,
            "dense_query_prompt": args.query_prompt,
            "dense_query_max_length": args.max_seq_length,
            "dense_sidecar_source_index": str(args.source_index),
            "dense_only_rebuild": True,
        }
    )
    output_meta.write_text(
        json.dumps(source_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(source_meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
