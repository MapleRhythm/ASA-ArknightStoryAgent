from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.anchors.action_evidence import action_target_score
from asa_arknight_story_agent.inference.anchors.terms import (
    anchor_hit_count,
    extract_action_targets,
    extract_question_anchor_terms,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.strips import split_evidence_strips
from asa_arknight_story_agent.inference.reveal.detection import is_reveal_question
from asa_arknight_story_agent.inference.reveal.scoring import reveal_direct_score
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order, strip_internal_evidence_meta


def refine_evidence_strips(
    *,
    pipeline: Any,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reranker = getattr(pipeline.retriever, "reranker", None)
    if reranker is None or not evidence:
        return evidence

    query = question
    if hypothesis.keywords:
        query = question + "\n检索线索: " + " ".join(hypothesis.keywords[:10])
    anchors = extract_question_anchor_terms(question, hypothesis)
    action_targets = extract_action_targets(question + "\n" + hypothesis.question)

    refined: list[dict[str, Any]] = []
    for item in evidence:
        doc = item.get("document") or {}
        chain_text = strip_internal_evidence_meta(str(item.get("evidence_chain_text") or "")).strip()
        clean_text = strip_internal_evidence_meta(str(doc.get("clean_text") or ""))
        if not (chain_text or clean_text):
            refined.append(item)
            continue
        strips = split_evidence_strips(clean_text, max_strips=pipeline.crag_refine_max_sentences)
        if len(strips) <= pipeline.crag_refine_top_sentences:
            refined.append(item)
            continue

        scores = reranker.score(
            query=query,
            documents=strips,
            batch_size=pipeline.query_config.rerank_batch_size,
        )
        ranked = sorted(
            enumerate(zip(strips, scores)),
            key=lambda pair: float(pair[1][1]),
            reverse=True,
        )[: pipeline.crag_refine_top_sentences]
        selected_indices = {index for index, _ in ranked}
        anchor_indices = [
            index
            for index, strip in enumerate(strips)
            if anchor_hit_count(strip, anchors) >= 2
        ]
        for index in anchor_indices[:2]:
            selected_indices.add(index)
        action_indices = [
            index
            for index, strip in enumerate(strips)
            if action_target_score(strip, action_targets) >= 2
        ]
        for index in action_indices[:3]:
            selected_indices.add(index)
        reveal_indices: list[int] = []
        if is_reveal_question(question, hypothesis):
            scored_reveal_indices = sorted(
                (
                    (reveal_direct_score(strip, question, hypothesis), index)
                    for index, strip in enumerate(strips)
                ),
                reverse=True,
            )
            reveal_indices = [index for score, index in scored_reveal_indices if score > 0][:4]
            for index in reveal_indices:
                selected_indices.add(index)
        selected_indices_list = sorted(selected_indices)
        selected_strips = [strips[index] for index in selected_indices_list]
        if chain_text and not is_reveal_question(question, hypothesis) and anchor_hit_count(chain_text, anchors) >= 2:
            selected_strips.insert(0, chain_text)
            selected_strips = dedupe_keep_order(selected_strips)
        refined_doc = dict(doc)
        refined_doc["original_clean_text"] = clean_text
        refined_doc["clean_text"] = "\n".join(selected_strips)
        refined_doc["search_text"] = refined_doc["clean_text"]
        refined_item = dict(item)
        refined_item["document"] = refined_doc
        if is_reveal_question(question, hypothesis):
            refined_item["prompt_prefer_clean_text"] = True
        if chain_text:
            refined_item["evidence_chain_text"] = chain_text
        refined_item["crag_refinement"] = {
            "enabled": True,
            "original_sentence_count": len(strips),
            "kept_sentence_count": len(selected_strips),
            "kept_sentence_indices": selected_indices_list,
            "anchor_sentence_indices": anchor_indices[:2],
            "reveal_sentence_indices": reveal_indices,
            "max_sentence_score": max(float(score) for score in scores) if scores else None,
        }
        refined.append(refined_item)
    return refined
