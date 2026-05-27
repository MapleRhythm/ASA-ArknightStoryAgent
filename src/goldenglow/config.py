from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "ArknightsGameData" / "zh_CN" / "gamedata"
STORY_ROOT = DATA_ROOT / "story"
EXCEL_ROOT = DATA_ROOT / "excel"
MODEL_ROOT = PROJECT_ROOT / "model"
INDEX_ROOT = PROJECT_ROOT / "indexes" / "arknights_story"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

EMBEDDING_MODEL_DIR = MODEL_ROOT / "embeddings" / "bge-small-zh-v1.5"
BASE_RERANKER_MODEL_DIR = MODEL_ROOT / "reranker" / "bge-reranker-v2-m3"
EVIDENCE_CHAIN_RERANKER_MODEL_DIR = MODEL_ROOT / "reranker" / "bge-reranker-v2-m3-evidence-chain-answerability"
RERANKER_MODEL_DIR = EVIDENCE_CHAIN_RERANKER_MODEL_DIR if EVIDENCE_CHAIN_RERANKER_MODEL_DIR.exists() else BASE_RERANKER_MODEL_DIR

DOCUMENTS_PATH = INDEX_ROOT / "documents.jsonl"
FAISS_INDEX_PATH = INDEX_ROOT / "faiss.index"
BM25_TOKENS_PATH = INDEX_ROOT / "bm25_tokens.pkl"
CORPUS_METADATA_PATH = INDEX_ROOT / "index_meta.json"
OPERATOR_ALIAS_MAP_PATH = INDEX_ROOT / "operator_aliases.json"
MINIRAG_INDEX_ROOT = PROJECT_ROOT / "indexes" / "arknights_story_minirag"
MINIRAG_GRAPH_PATH = MINIRAG_INDEX_ROOT / "graph.json"
MANUAL_OPERATOR_ALIAS_SOURCE_PATH = PROJECT_ROOT / "data" / "processed" / "operator_aliases_manual.json"
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
    minirag_top_k: int = 120
    fusion_top_k: int = 40
    rerank_top_k: int = 15
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 0.8
    minirag_weight: float = 0.35
    minirag_mode_weights: dict[str, float] = field(default_factory=dict)
    minirag_fusion_mode: str = "score"
    minirag_chapter_isolation: bool = True
    minirag_auto_second_retrieval: bool = True
    minirag_scope_seed_top_k: int = 40
    minirag_expansion_query_top_k: int = 8
    minirag_graph_scope_min_ratio: float = 2.5
    minirag_second_pass_scope_min_ratio: float = 2.5
    reranker_candidate_top_k: int = 120
    enable_neighbor_expansion: bool = False
    neighbor_max_seed_docs: int = 24
    neighbor_story_window: int = 2
    neighbor_activity_story_sort_window: int = 1
    rerank_batch_size: int = 8
