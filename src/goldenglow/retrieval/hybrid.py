from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from goldenglow.config import BM25_TOKENS_PATH, DOCUMENTS_PATH, FAISS_INDEX_PATH, QueryConfig
from goldenglow.retrieval.reranker import CrossEncoderReranker


ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize_for_bm25(text: str) -> list[str]:
    lowered = text.lower()
    ascii_tokens = ASCII_TOKEN_RE.findall(lowered)
    cjk_chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    cjk_bigrams = [f"{cjk_chars[i]}{cjk_chars[i + 1]}" for i in range(len(cjk_chars) - 1)]
    return ascii_tokens + cjk_chars + cjk_bigrams


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ArknightsHybridRetriever:
    def __init__(
        self,
        documents: list[dict],
        index: faiss.Index,
        bm25: BM25Okapi,
        embedding_model: SentenceTransformer,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.documents = documents
        self.index = index
        self.bm25 = bm25
        self.embedding_model = embedding_model
        self.reranker = reranker

    @classmethod
    def from_paths(
        cls,
        *,
        embedding_model_path: Path,
        reranker_model_path: Path | None = None,
        documents_path: Path = DOCUMENTS_PATH,
        faiss_index_path: Path = FAISS_INDEX_PATH,
        bm25_tokens_path: Path = BM25_TOKENS_PATH,
        device: str = "cpu",
    ) -> "ArknightsHybridRetriever":
        documents = load_jsonl(documents_path)
        index = faiss.read_index(str(faiss_index_path))
        with bm25_tokens_path.open("rb") as handle:
            tokenized_corpus = pickle.load(handle)
        bm25 = BM25Okapi(tokenized_corpus)
        embedding_model = SentenceTransformer(str(embedding_model_path), device=device)
        reranker = None
        if reranker_model_path and reranker_model_path.exists():
            reranker = CrossEncoderReranker(reranker_model_path, device=device)
        return cls(
            documents=documents,
            index=index,
            bm25=bm25,
            embedding_model=embedding_model,
            reranker=reranker,
        )

    def dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        vector = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores, indices = self.index.search(vector.astype(np.float32), top_k)
        hits: list[dict[str, Any]] = []
        for score, doc_index in zip(scores[0].tolist(), indices[0].tolist()):
            if doc_index < 0:
                continue
            doc = self.documents[doc_index]
            hits.append(
                {
                    "doc_index": doc_index,
                    "score": float(score),
                    "document": doc,
                }
            )
        return hits

    def sparse_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        hits: list[dict[str, Any]] = []
        for doc_index in top_indices.tolist():
            score = float(scores[doc_index])
            if score <= 0:
                continue
            hits.append(
                {
                    "doc_index": int(doc_index),
                    "score": score,
                    "document": self.documents[int(doc_index)],
                }
            )
        return hits

    def reciprocal_rank_fusion(
        self,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
        *,
        top_k: int,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[dict[str, Any]]:
        fused: dict[int, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits):
            doc_index = hit["doc_index"]
            item = fused.setdefault(
                doc_index,
                {
                    "doc_index": doc_index,
                    "document": hit["document"],
                    "dense_score": None,
                    "sparse_score": None,
                    "fusion_score": 0.0,
                },
            )
            item["dense_score"] = hit["score"]
            item["fusion_score"] += dense_weight / (rrf_k + rank + 1)

        for rank, hit in enumerate(sparse_hits):
            doc_index = hit["doc_index"]
            item = fused.setdefault(
                doc_index,
                {
                    "doc_index": doc_index,
                    "document": hit["document"],
                    "dense_score": None,
                    "sparse_score": None,
                    "fusion_score": 0.0,
                },
            )
            item["sparse_score"] = hit["score"]
            item["fusion_score"] += sparse_weight / (rrf_k + rank + 1)

        return sorted(
            fused.values(),
            key=lambda item: item["fusion_score"],
            reverse=True,
        )[:top_k]

    def search(
        self,
        query: str,
        *,
        config: QueryConfig | None = None,
    ) -> list[dict[str, Any]]:
        query_config = config or QueryConfig()
        dense_hits = self.dense_search(query, top_k=query_config.dense_top_k)
        sparse_hits = self.sparse_search(query, top_k=query_config.sparse_top_k)
        fused_hits = self.reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            top_k=query_config.fusion_top_k,
            rrf_k=query_config.rrf_k,
            dense_weight=query_config.dense_weight,
            sparse_weight=query_config.sparse_weight,
        )
        if not self.reranker:
            return fused_hits[: query_config.rerank_top_k]
        rerank_scores = self.reranker.score(
            query=query,
            documents=[item["document"]["search_text"] for item in fused_hits],
            batch_size=query_config.rerank_batch_size,
        )
        for item, rerank_score in zip(fused_hits, rerank_scores):
            item["rerank_score"] = rerank_score
        return sorted(
            fused_hits,
            key=lambda item: item.get("rerank_score", float("-inf")),
            reverse=True,
        )[: query_config.rerank_top_k]
