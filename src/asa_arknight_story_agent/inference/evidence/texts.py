from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, strip_internal_evidence_meta


def evidence_identity(item: dict[str, Any]) -> str:
    doc = item.get("document") or {}
    doc_id = str(doc.get("id") or "").strip()
    if doc_id:
        return "doc:" + doc_id
    doc_index = item.get("doc_index")
    if doc_index is not None:
        return "idx:" + str(doc_index)
    return "text:" + evidence_text(item)[:160]


def evidence_text(item: dict[str, Any]) -> str:
    doc = item.get("document") or {}
    parts = [
        str(item.get("evidence_chain_text") or ""),
        str(doc.get("clean_text") or ""),
        str(doc.get("search_text") or ""),
        str(doc.get("activity_name") or ""),
        str(doc.get("story_name") or ""),
        str(doc.get("stage_code") or ""),
        str(doc.get("avg_tag") or ""),
        str(doc.get("source_path") or ""),
    ]
    text_parts = [strip_internal_evidence_meta(part).strip() for part in parts if str(part).strip()]
    return "\n".join(dedupe_keep_order(text_parts))


def document_clean_text(item: dict[str, Any]) -> str:
    doc = item.get("document") or {}
    return strip_internal_evidence_meta(str(doc.get("clean_text") or doc.get("search_text") or "")).strip()


def document_chain_text(item: dict[str, Any]) -> str:
    return strip_internal_evidence_meta(str(item.get("evidence_chain_text") or "")).strip()


def best_prompt_text(item: dict[str, Any], *, prefer_direct: bool = False) -> str:
    clean_text = document_clean_text(item)
    chain_text = document_chain_text(item)
    if prefer_direct:
        # Web-context and other direct-quote items keep their own document text.
        return clean_text or chain_text
    # The evidence-chain text is a story-ordered, source-prefixed multi-chunk
    # render and carries strictly more context than a single chunk; prefer it
    # so the model sees the chain the retriever assembled.
    return chain_text or clean_text


def prefer_direct_prompt_text(item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    payload["prompt_prefer_clean_text"] = True
    return payload


def is_web_context_item(item: dict[str, Any]) -> bool:
    doc = item.get("document") or {}
    return item.get("supplemental_source") == "web_context" or str(doc.get("id") or "").startswith("web_context/")


def is_moegirl_evidence(item: dict[str, Any]) -> bool:
    doc = item.get("document") or {}
    doc_id = str(doc.get("id") or "")
    activity_name = str(doc.get("activity_name") or "")
    source_path = str(doc.get("source_path") or "")
    source_path_lower = source_path.lower()
    return (
        doc_id.startswith("moegirl/")
        or activity_name == "萌百世界观资料"
        or "/moegirl/" in source_path_lower
        or "moegirl" in source_path_lower
        or "萌百" in source_path
    )


def evidence_score(item: dict[str, Any]) -> float:
    for key in (
        "evidence_chain_score",
        "evidence_chain_model_score",
        "rerank_score",
        "fusion_score",
        "dense_score",
        "sparse_score",
        "score",
    ):
        value = item.get(key)
        if value is not None:
            return float(value)
    return 0.0


def prompt_evidence_score(item: dict[str, Any]) -> float:
    score = evidence_score(item)
    if is_moegirl_evidence(item):
        score -= 6.0
    return score
