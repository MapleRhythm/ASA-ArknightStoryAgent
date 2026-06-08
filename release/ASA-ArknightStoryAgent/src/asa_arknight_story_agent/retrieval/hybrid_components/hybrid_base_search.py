from __future__ import annotations

from typing import Any

import numpy as np

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_utils import tokenize_for_bm25


class HybridBaseSearchMixin:
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

    def dense_search_chapter(
        self,
        query: str,
        top_k: int,
        *,
        chapter_scope: str,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not chapter_scope:
            return []
        doc_indices = self.chapter_doc_indices.get(chapter_scope, [])
        if not doc_indices:
            return []
        scoped_indices, scoped_vectors = self._dense_scope_vectors(chapter_scope, doc_indices)
        if scoped_vectors.size == 0:
            return []
        vector = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)[0]
        scores = scoped_vectors @ vector
        take = min(top_k, len(scores))
        if take <= 0:
            return []
        if take >= len(scores):
            order = np.argsort(scores)[::-1]
        else:
            order = np.argpartition(scores, -take)[-take:]
            order = order[np.argsort(scores[order])[::-1]]
        hits: list[dict[str, Any]] = []
        for offset in order.tolist():
            doc_index = int(scoped_indices[offset])
            hits.append(
                {
                    "doc_index": doc_index,
                    "score": float(scores[offset]),
                    "document": self.documents[doc_index],
                    "scoped_source": "chapter_dense",
                }
            )
        return hits

    def _dense_scope_vectors(
        self,
        chapter_scope: str,
        doc_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self._dense_scope_cache.get(chapter_scope)
        if cached is not None:
            return cached
        valid_indices: list[int] = []
        vectors: list[np.ndarray] = []
        for doc_index in doc_indices:
            try:
                vector = self.index.reconstruct(int(doc_index))
            except Exception:
                continue
            valid_indices.append(int(doc_index))
            vectors.append(np.asarray(vector, dtype=np.float32))
        if vectors:
            payload = (np.asarray(valid_indices, dtype=np.int64), np.vstack(vectors).astype(np.float32))
        else:
            payload = (np.asarray([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32))
        if len(self._dense_scope_cache) >= 32:
            self._dense_scope_cache.pop(next(iter(self._dense_scope_cache)))
        self._dense_scope_cache[chapter_scope] = payload
        return payload

    def sparse_search(
        self,
        query: str,
        top_k: int,
        *,
        storyline_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        allowed_indices: list[int] | None = None
        if storyline_scope:
            allowed_indices = self.storyline_doc_indices.get(storyline_scope, [])
            if not allowed_indices:
                return []

        if allowed_indices is not None:
            candidate_indices = np.array(allowed_indices, dtype=np.int64)
            candidate_scores = scores[candidate_indices]
            positive_mask = candidate_scores > 0
            candidate_indices = candidate_indices[positive_mask]
            candidate_scores = candidate_scores[positive_mask]
            if len(candidate_indices) <= 0:
                return []
            if top_k >= len(candidate_indices):
                order = np.argsort(candidate_scores)[::-1]
            else:
                order = np.argpartition(candidate_scores, -top_k)[-top_k:]
                order = order[np.argsort(candidate_scores[order])[::-1]]
            top_indices = candidate_indices[order]
        elif top_k >= len(scores):
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

    def sparse_search_chapter(
        self,
        query: str,
        top_k: int,
        *,
        chapter_scope: str,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not chapter_scope:
            return []
        allowed_indices = self.chapter_doc_indices.get(chapter_scope, [])
        if not allowed_indices:
            return []
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        candidate_indices = np.array(allowed_indices, dtype=np.int64)
        candidate_scores = scores[candidate_indices]
        positive_mask = candidate_scores > 0
        candidate_indices = candidate_indices[positive_mask]
        candidate_scores = candidate_scores[positive_mask]
        if len(candidate_indices) <= 0:
            return []
        if top_k >= len(candidate_indices):
            order = np.argsort(candidate_scores)[::-1]
        else:
            order = np.argpartition(candidate_scores, -top_k)[-top_k:]
            order = order[np.argsort(candidate_scores[order])[::-1]]
        hits: list[dict[str, Any]] = []
        for offset in order.tolist():
            doc_index = int(candidate_indices[offset])
            hits.append(
                {
                    "doc_index": doc_index,
                    "score": float(candidate_scores[offset]),
                    "document": self.documents[doc_index],
                    "scoped_source": "chapter_sparse",
                }
            )
        return hits

    def minirag_search(
        self,
        query: str,
        top_k: int,
        *,
        chapter_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.minirag_index is None:
            return []
        return self.minirag_index.search(
            query,
            self.documents,
            top_k=top_k,
            chapter_scope=chapter_scope,
        )
