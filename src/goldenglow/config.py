from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "ArknightsGameData" / "zh_CN" / "gamedata"
STORY_ROOT = DATA_ROOT / "story"
EXCEL_ROOT = DATA_ROOT / "excel"
MODEL_ROOT = PROJECT_ROOT / "model"
INDEX_ROOT = PROJECT_ROOT / "indexes" / "arknights_story"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

EMBEDDING_MODEL_DIR = MODEL_ROOT / "embeddings" / "bge-small-zh-v1.5"
RERANKER_MODEL_DIR = MODEL_ROOT / "reranker" / "bge-reranker-v2-m3"

DOCUMENTS_PATH = INDEX_ROOT / "documents.jsonl"
FAISS_INDEX_PATH = INDEX_ROOT / "faiss.index"
BM25_TOKENS_PATH = INDEX_ROOT / "bm25_tokens.pkl"
CORPUS_METADATA_PATH = INDEX_ROOT / "index_meta.json"
CHUNKS_DEBUG_PATH = OUTPUT_ROOT / "story_chunks_preview.jsonl"


@dataclass(slots=True)
class BuildConfig:
    max_chars: int = 420
    overlap_segments: int = 1
    embedding_batch_size: int = 64
    normalize_embeddings: bool = True


@dataclass(slots=True)
class QueryConfig:
    dense_top_k: int = 60
    sparse_top_k: int = 60
    fusion_top_k: int = 40
    rerank_top_k: int = 15
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 0.8
    rerank_batch_size: int = 8
