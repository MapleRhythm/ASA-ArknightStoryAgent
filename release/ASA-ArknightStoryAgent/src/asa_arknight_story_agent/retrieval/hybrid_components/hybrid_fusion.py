from __future__ import annotations

from typing import Any


class HybridFusionMixin:
    def reciprocal_rank_fusion(
        self,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
        minirag_hits: list[dict[str, Any]] | None = None,
        *,
        top_k: int,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
        minirag_weight: float = 0.0,
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
                    "minirag_score": None,
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
                    "minirag_score": None,
                    "fusion_score": 0.0,
                },
            )
            item["sparse_score"] = hit["score"]
            if hit.get("minirag_score") is not None:
                item["minirag_score"] = hit["minirag_score"]
            item["fusion_score"] += sparse_weight / (rrf_k + rank + 1)

        for rank, hit in enumerate(minirag_hits or []):
            doc_index = hit["doc_index"]
            item = fused.setdefault(
                doc_index,
                {
                    "doc_index": doc_index,
                    "document": hit["document"],
                    "dense_score": None,
                    "sparse_score": None,
                    "minirag_score": None,
                    "fusion_score": 0.0,
                },
            )
            item["minirag_score"] = hit.get("minirag_score", hit.get("score"))
            item["fusion_score"] += minirag_weight / (rrf_k + rank + 1)

        return sorted(
            fused.values(),
            key=lambda item: item["fusion_score"],
            reverse=True,
        )[:top_k]

    @staticmethod
    def append_supplemental_hits(
        primary_hits: list[dict[str, Any]],
        supplemental_hits: list[dict[str, Any]],
        *,
        top_k: int,
        source_name: str,
    ) -> list[dict[str, Any]]:
        merged = [dict(item) for item in primary_hits]
        seen = {int(item["doc_index"]) for item in merged}
        append_rank = 0
        for hit in supplemental_hits:
            doc_index = int(hit["doc_index"])
            if doc_index in seen:
                continue
            seen.add(doc_index)
            append_rank += 1
            item = dict(hit)
            item.setdefault("fusion_score", 0.0)
            item["supplemental_source"] = source_name
            item["supplemental_rank"] = append_rank
            merged.append(item)
            if len(merged) >= top_k:
                break
        return merged[:top_k]

    def _merge_candidate_hits(
        self,
        target: dict[int, dict[str, Any]],
        hits: list[dict[str, Any]],
        *,
        score_scale: float = 1.0,
    ) -> None:
        for hit in hits:
            doc_index = int(hit["doc_index"])
            item = target.setdefault(
                doc_index,
                {
                    "doc_index": doc_index,
                    "document": hit["document"],
                    "dense_score": None,
                    "sparse_score": None,
                    "fusion_score": 0.0,
                },
            )
            dense_score = hit.get("dense_score")
            sparse_score = hit.get("sparse_score")
            if dense_score is not None:
                item["dense_score"] = max(float(dense_score), float(item.get("dense_score") or float("-inf")))
            if sparse_score is not None:
                item["sparse_score"] = max(float(sparse_score), float(item.get("sparse_score") or float("-inf")))
            item["fusion_score"] = max(
                float(item.get("fusion_score") or 0.0),
                float(hit.get("fusion_score") or 0.0) * score_scale,
            )
