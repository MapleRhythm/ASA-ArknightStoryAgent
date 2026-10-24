from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from asa_arknight_story_agent.config import BM25_TOKENS_PATH, DOCUMENTS_PATH, FAISS_INDEX_PATH, QueryConfig
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import (  # noqa: F401
    ACTION_HINT_TERMS,
    BRANCH_SOURCE_MARKERS,
    CONCEPT_CRISIS_QUERY_RE,
    CONCEPT_CRISIS_TERMS,
    CONCEPT_DEFINITION_TERMS,
    HIGH_RERANK_QUERY_TYPES,
    LOW_RERANK_QUERY_TYPES,
    MAIN_CHAPTER_REF_RE,
    MAIN_CHAPTER_SOURCE_RE,
    MOEGIRL_SOURCE_MARKERS,
    MOTIVE_HINT_TERMS,
    OUTCOME_HINT_TERMS,
    PROFILE_SOURCE_MARKERS,
    QUERY_CHAR_STOP_CHARS,
    QUERY_TYPES,
    REVEAL_ANSWER_TERMS,
    REVEAL_DIRECT_CONTEXT_TERMS,
    REVEAL_QUERY_RE,
    REVEAL_SUPPORT_TERMS,
    STAGE_NUMBER_RE,
    STORY_SOURCE_MARKERS,
    TARGET_CONTEXT_HINT_TERMS,
)
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_base_search import HybridBaseSearchMixin
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_evidence_chains import HybridEvidenceChainsMixin
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_fusion import HybridFusionMixin
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_orchestration import HybridSearchOrchestrationMixin
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_query_analysis import HybridQueryAnalysisMixin
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_neighbors import HybridNeighborExpansionMixin
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_scoring import HybridScoringMixin
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_utils import (
    extract_main_chapter_numbers,  # noqa: F401
    load_jsonl,
    parse_main_chapter_number,  # noqa: F401
)
from asa_arknight_story_agent.retrieval.minirag import MiniRAGIndex, document_chapter_scope_key
from asa_arknight_story_agent.retrieval.reranker import CrossEncoderReranker
from asa_arknight_story_agent.retrieval.storyline import document_storyline_scopes


class ArknightsHybridRetriever(
    HybridSearchOrchestrationMixin,
    HybridFusionMixin,
    HybridBaseSearchMixin,
    HybridNeighborExpansionMixin,
    HybridEvidenceChainsMixin,
    HybridScoringMixin,
    HybridQueryAnalysisMixin,
):
    def __init__(
        self,
        documents: list[dict],
        index: faiss.Index,
        bm25: BM25Okapi,
        embedding_model: SentenceTransformer,
        reranker: CrossEncoderReranker | None = None,
        minirag_index: MiniRAGIndex | None = None,
    ) -> None:
        self.documents = documents
        self.index = index
        self.bm25 = bm25
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.minirag_index = minirag_index
        self.chapter_doc_indices: dict[str, list[int]] = {}
        self.story_doc_indices: dict[str, list[int]] = {}
        self.storyline_doc_indices: dict[str, list[int]] = {}
        self.stage_doc_indices: dict[tuple[str, str], list[int]] = {}
        self.activity_story_sort_doc_indices: dict[str, dict[int, list[int]]] = {}
        self._dense_scope_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for doc_index, document in enumerate(documents):
            chapter_scope = document_chapter_scope_key(document)
            if chapter_scope:
                self.chapter_doc_indices.setdefault(chapter_scope, []).append(doc_index)
            story_id = str(document.get("story_id") or "").strip()
            if story_id:
                self.story_doc_indices.setdefault(story_id, []).append(doc_index)
            for storyline_scope in document_storyline_scopes(document):
                self.storyline_doc_indices.setdefault(storyline_scope, []).append(doc_index)
            activity_id = str(document.get("activity_id") or "").strip()
            stage_code = str(document.get("stage_code") or "").strip()
            if activity_id and stage_code:
                self.stage_doc_indices.setdefault((activity_id, stage_code), []).append(doc_index)
            story_sort = document.get("story_sort")
            if activity_id and isinstance(story_sort, int):
                self.activity_story_sort_doc_indices.setdefault(activity_id, {}).setdefault(story_sort, []).append(doc_index)

    @classmethod
    def from_paths(
        cls,
        *,
        embedding_model_path: Path,
        reranker_model_path: Path | None = None,
        reranker_max_length: int = 1024,
        documents_path: Path = DOCUMENTS_PATH,
        faiss_index_path: Path = FAISS_INDEX_PATH,
        bm25_tokens_path: Path = BM25_TOKENS_PATH,
        minirag_index_path: Path | None = None,
        device: str = "cpu",
    ) -> "ArknightsHybridRetriever":
        started = time.time()
        print(f"[retriever-load] documents {documents_path}", file=sys.stderr, flush=True)
        documents = load_jsonl(documents_path)
        print(f"[retriever-load] documents={len(documents)} elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        print(f"[retriever-load] faiss {faiss_index_path}", file=sys.stderr, flush=True)
        index = faiss.read_index(str(faiss_index_path))
        print(f"[retriever-load] faiss loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        print(f"[retriever-load] bm25 {bm25_tokens_path}", file=sys.stderr, flush=True)
        with bm25_tokens_path.open("rb") as handle:
            tokenized_corpus = pickle.load(handle)
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"[retriever-load] bm25 loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        print(f"[retriever-load] embedding {embedding_model_path} device={device}", file=sys.stderr, flush=True)
        embedding_model = SentenceTransformer(str(embedding_model_path), device=device)
        print(f"[retriever-load] embedding loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        reranker = None
        if reranker_model_path and reranker_model_path.exists():
            print(f"[retriever-load] reranker {reranker_model_path} device={device}", file=sys.stderr, flush=True)
            reranker = CrossEncoderReranker(reranker_model_path, device=device, max_length=reranker_max_length)
            print(f"[retriever-load] reranker loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        minirag_index = None
        resolved_minirag_path = minirag_index_path or None
        if resolved_minirag_path and resolved_minirag_path.exists():
            print(f"[retriever-load] minirag {resolved_minirag_path}", file=sys.stderr, flush=True)
            minirag_index = MiniRAGIndex.load(resolved_minirag_path)
            print(f"[retriever-load] minirag loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        print(f"[retriever-load] done elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        return cls(
            documents=documents,
            index=index,
            bm25=bm25,
            embedding_model=embedding_model,
            reranker=reranker,
            minirag_index=minirag_index,
        )
