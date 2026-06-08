from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import ACTION_HINT_TERMS
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_utils import extract_main_chapter_numbers


class HybridFocusAdjustmentsMixin:
    def _apply_main_chapter_focus_adjustment(self, query: str, candidates: list[dict[str, Any]]) -> None:
        query_chapters = extract_main_chapter_numbers(query)
        if not query_chapters:
            return
        target_chapter = query_chapters[0]
        for item in candidates:
            document = item["document"]
            doc_chapter = self._document_main_chapter_number(document)
            if doc_chapter == target_chapter:
                item["rerank_score"] = float(item.get("rerank_score") or 0.0) + 3.0
                item["main_chapter_focus"] = f"match:{target_chapter}"
            elif doc_chapter is not None:
                item["rerank_score"] = float(item.get("rerank_score") or 0.0) - 2.5
                item["main_chapter_focus"] = f"mismatch:{doc_chapter}!={target_chapter}"

    def _apply_source_type_focus_adjustment(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        query_mode: str | None,
    ) -> None:
        background_query = self._is_background_query(query)
        detail_query = self._is_story_detail_query(query, query_mode=query_mode)
        reveal_query = self._is_reveal_query(query)
        if not background_query and not detail_query and not reveal_query:
            return
        for item in candidates:
            document = item["document"]
            source_type = self._document_source_type(document)
            item["source_type"] = source_type
            score = float(item.get("rerank_score") or 0.0)
            if background_query and source_type == "moegirl_background":
                item["rerank_score"] = score + 0.8
                item["source_type_focus"] = "background_boost"
            elif detail_query and source_type == "story_text":
                item["rerank_score"] = score + 0.8
                item["source_type_focus"] = "story_detail_boost"
            elif detail_query and source_type == "moegirl_background":
                item["rerank_score"] = score - 1.1
                item["source_type_focus"] = "background_context_only"
            if reveal_query:
                answerability = self._compute_answerability_bonus(query, document)
                if answerability > 0:
                    reveal_bonus = answerability * 1.35
                    item["rerank_score"] = float(item.get("rerank_score") or 0.0) + reveal_bonus
                    item["reveal_answerability_focus"] = round(answerability, 3)
                elif source_type == "profile":
                    item["rerank_score"] = float(item.get("rerank_score") or 0.0) - 3.0
                    item["reveal_answerability_focus"] = "profile_demote"

    def _apply_causal_focus_adjustment(self, query: str, candidates: list[dict[str, Any]]) -> None:
        query_terms = self._extract_chain_query_terms(query)
        long_query_terms = [term for term in query_terms if len(term) >= 5]
        query_action_terms = [term for term in ACTION_HINT_TERMS if term in query]
        if not long_query_terms and not query_action_terms:
            return
        for item in candidates:
            text = self._document_text(item["document"])
            has_long_exact = any(term in text for term in long_query_terms)
            has_query_action = any(term in text for term in query_action_terms)
            has_chain_bonus = float(item.get("chain_bonus") or 0.0) > 0.0
            penalty = 0.0
            boost = 0.0
            if has_long_exact:
                boost += 1.05
            if has_long_exact and has_query_action:
                boost += 0.35
            if not has_chain_bonus and long_query_terms and not has_long_exact:
                penalty += 0.45
            if not has_chain_bonus and query_action_terms and not has_query_action:
                penalty += 0.2
            if has_query_action and not has_long_exact and not has_chain_bonus:
                penalty += 0.22
            if boost or penalty:
                item["rerank_score"] = float(item.get("rerank_score") or 0.0) + boost - penalty
