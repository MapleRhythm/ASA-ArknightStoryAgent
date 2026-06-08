from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.pipeline.constants import MULTI_QUERY_MERGE_RRF_K


def hit_raw_score(item: dict[str, Any]) -> float:
    for key in ("score", "dense_score", "sparse_score", "minirag_score", "fusion_score"):
        value = item.get(key)
        if value is not None:
            return float(value)
    return 0.0


def merge_ranked_hits(*ranked_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            doc_index = int(item["doc_index"])
            raw_score = hit_raw_score(item)
            payload = merged.setdefault(
                doc_index,
                {
                    "doc_index": doc_index,
                    "document": item["document"],
                    "score": raw_score,
                    "multi_query_rank_score": 0.0,
                    "multi_query_match_count": 0,
                    "best_query_rank": rank,
                },
            )
            payload["score"] = max(float(payload.get("score") or 0.0), raw_score)
            if item.get("minirag_score") is not None:
                payload["minirag_score"] = max(
                    float(payload.get("minirag_score") or 0.0),
                    float(item["minirag_score"]),
                )
            payload["multi_query_rank_score"] = float(payload.get("multi_query_rank_score") or 0.0) + (
                1.0 / (MULTI_QUERY_MERGE_RRF_K + rank + 1)
            )
            payload["multi_query_match_count"] = int(payload.get("multi_query_match_count") or 0) + 1
            previous_best_rank = payload.get("best_query_rank")
            payload["best_query_rank"] = min(
                int(previous_best_rank) if previous_best_rank is not None else rank,
                rank,
            )

    return sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("multi_query_rank_score") or 0.0),
            int(item.get("multi_query_match_count") or 0),
            -int(item.get("best_query_rank") or 0),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )


def merge_evidence_keep_order(*evidence_lists: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for evidence in evidence_lists:
        for item in evidence:
            doc = item.get("document") or {}
            doc_id = str(doc.get("id") or item.get("doc_index") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            merged.append(item)
            if limit is not None and len(merged) >= limit:
                return merged
    return merged
