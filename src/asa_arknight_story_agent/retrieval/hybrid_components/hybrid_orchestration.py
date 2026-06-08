from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.config import QueryConfig


class HybridSearchOrchestrationMixin:
    def search(
        self,
        query: str,
        *,
        config: QueryConfig | None = None,
    ) -> list[dict[str, Any]]:
        fused_hits = self.search_pre_rerank(query, config=config)
        query_config = config or QueryConfig()
        return self.rerank_with_evidence_chains(
            query,
            fused_hits,
            top_k=query_config.rerank_top_k,
            batch_size=query_config.rerank_batch_size,
            fallback_to_document_rerank=True,
        )

    def search_pre_rerank(
        self,
        query: str,
        *,
        config: QueryConfig | None = None,
    ) -> list[dict[str, Any]]:
        query_config = config or QueryConfig()
        minirag_weight = self.effective_minirag_weight(query, config=query_config)
        dense_hits = self.dense_search(query, top_k=query_config.dense_top_k)
        sparse_hits = self.sparse_search(query, top_k=query_config.sparse_top_k)
        minirag_hits = (
            self.minirag_search(query, top_k=query_config.minirag_top_k)
            if minirag_weight > 0
            else []
        )
        if query_config.minirag_fusion_mode == "append":
            primary_hits = self.reciprocal_rank_fusion(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                minirag_hits=[],
                top_k=query_config.fusion_top_k,
                rrf_k=query_config.rrf_k,
                dense_weight=query_config.dense_weight,
                sparse_weight=query_config.sparse_weight,
                minirag_weight=0.0,
            )
            fused_hits = self.append_supplemental_hits(
                primary_hits,
                minirag_hits,
                top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
                source_name="minirag",
            )
        else:
            fused_hits = self.reciprocal_rank_fusion(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                minirag_hits=minirag_hits,
                top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
                rrf_k=query_config.rrf_k,
                dense_weight=query_config.dense_weight,
                sparse_weight=query_config.sparse_weight,
                minirag_weight=minirag_weight,
            )
        if query_config.enable_neighbor_expansion:
            fused_hits = self.expand_hits_with_neighbors(
                fused_hits,
                max_seed_docs=query_config.neighbor_max_seed_docs,
                story_window=query_config.neighbor_story_window,
                activity_story_sort_window=query_config.neighbor_activity_story_sort_window,
                same_story_sweep=query_config.enable_same_story_sweep,
                same_story_max_seed_docs=query_config.same_story_sweep_max_seed_docs,
                same_story_max_docs_per_story=query_config.same_story_sweep_max_docs_per_story,
                top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k)
                + (
                    query_config.same_story_sweep_extra_candidates
                    if query_config.enable_same_story_sweep
                    else 0
                ),
            )
        return fused_hits

    def effective_minirag_weight(
        self,
        query: str,
        *,
        config: QueryConfig | None = None,
    ) -> float:
        query_config = config or QueryConfig()
        query_mode = self._infer_query_mode(query)
        mode_weights = query_config.minirag_mode_weights or {}
        if query_mode in mode_weights:
            return query_config.minirag_weight * float(mode_weights[query_mode])
        if query_mode in {"relation", "reveal", "causality", "fact"}:
            return query_config.minirag_weight
        return query_config.minirag_weight * 0.25
