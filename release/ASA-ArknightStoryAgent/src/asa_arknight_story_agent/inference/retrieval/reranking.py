from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.pipeline.constants import QUERY_TYPES
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


def rerank_hits(
    retriever: Any,
    rerank_query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    batch_size: int,
    query_mode: str | None = None,
) -> list[dict[str, Any]]:
    if not hits:
        return []
    if hasattr(retriever, "rerank_with_evidence_chains"):
        return retriever.rerank_with_evidence_chains(
            rerank_query,
            hits,
            top_k=top_k,
            batch_size=batch_size,
            query_mode=query_mode,
            fallback_to_document_rerank=True,
        )
    if not retriever.reranker:
        return hits[:top_k]
    scores = retriever.reranker.score(
        query=rerank_query,
        documents=[item["document"]["search_text"] for item in hits],
        batch_size=batch_size,
    )
    reranked = []
    for item, score in zip(hits, scores):
        payload = dict(item)
        payload["rerank_score"] = float(score)
        reranked.append(payload)
    reranked.sort(key=lambda item: item.get("rerank_score", float("-inf")), reverse=True)
    return reranked[:top_k]


def classify_retrieval_query_mode(hypothesis: HypothesisDocument) -> str:
    if hypothesis.query_type in QUERY_TYPES:
        return hypothesis.query_type
    answer_type = hypothesis.expected_answer_type
    question = hypothesis.question
    if hypothesis.intent == "character_relation" or any(token in answer_type for token in ("身份关系", "关系")):
        return "relation"
    if any(token in question for token in ("阴谋", "真相", "秘密", "识破", "揭穿", "曝光", "暴露", "幕后", "主使", "黑幕", "骗局", "诡计")):
        return "reveal"
    if any(token in question for token in ("谜", "怎么回事", "究竟", "到底")):
        return "mystery"
    if hypothesis.intent == "plot_reasoning" or any(token in answer_type for token in ("原因", "动机", "过程", "解释")):
        return "causality" if any(token in question for token in ("为什么", "为何", "原因", "导致", "造成")) else "reasoning"
    if any(token in answer_type for token in ("概念定义/危机原因", "answerability")):
        return "answerability"
    if hypothesis.intent in {"plot_fact", "timeline", "compare"}:
        return "fact"
    if any(token in answer_type for token in ("事实", "时间线", "对比")):
        return "fact"
    return "reasoning"
