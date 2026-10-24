from __future__ import annotations

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import (
    ACTION_HINT_TERMS,
    BRANCH_SOURCE_MARKERS,
    CONCEPT_CRISIS_TERMS,
    CONCEPT_DEFINITION_TERMS,
    LOW_RERANK_QUERY_TYPES,
    MOEGIRL_SOURCE_MARKERS,
    PROFILE_SOURCE_MARKERS,
    REVEAL_ANSWER_TERMS,
    REVEAL_DIRECT_CONTEXT_TERMS,
    REVEAL_SUPPORT_TERMS,
)


class HybridSupportScoresMixin:
    def _compute_causality_support_bonus(self, query: str, document: dict) -> float:
        if not self._is_causal_query(query):
            return 0.0
        text = self._document_text(document)
        source_path = str(document.get("source_path") or "")
        query_terms = self._extract_chain_query_terms(query)
        long_exact_overlap = sum(1 for term in query_terms if len(term) >= 4 and term in text)
        action_overlap = sum(1 for term in ACTION_HINT_TERMS if term in query and term in text)
        causal_markers = (
            "因为",
            "为了",
            "原因",
            "动机",
            "目的",
            "导致",
            "不得不",
            "必须",
            "只要",
            "除非",
            "计划",
            "阴谋",
            "内情",
            "威胁",
            "破坏",
            "牺牲",
            "背叛",
            "堕落",
        )
        direct_causal_markers = ("因为", "为了", "原因", "动机", "导致", "不得不", "必须", "只要", "除非")
        marker_hits = sum(1 for term in causal_markers if term in text)
        direct_marker_hits = sum(1 for term in direct_causal_markers if term in text)
        score = 0.0
        if marker_hits and (long_exact_overlap or action_overlap):
            score += min(marker_hits, 4) * 0.42
            score += min(long_exact_overlap, 2) * 0.28
            score += min(action_overlap, 2) * 0.35
        elif long_exact_overlap and action_overlap:
            score += 0.35
        if marker_hits >= 2 and ("/story/" in source_path or "/memory/" in source_path or "/rogue/" in source_path):
            score += 0.55
        if any(marker in source_path for marker in PROFILE_SOURCE_MARKERS):
            if direct_marker_hits == 0 or action_overlap == 0:
                score = min(score, 0.2) - 1.2
        return score

    def _compute_concept_crisis_support_bonus(self, query: str, document: dict) -> float:
        if not self._is_concept_crisis_query(query):
            return 0.0
        text = self._document_text(document)
        if not text:
            return 0.0
        source_path = str(document.get("source_path") or "")
        subjects = self._concept_query_subjects(query)
        subject_hits = sum(1 for subject in subjects if subject and subject in text)
        definition_hits = sum(1 for term in CONCEPT_DEFINITION_TERMS if term in text)
        crisis_hits = sum(1 for term in CONCEPT_CRISIS_TERMS if term in text)
        if subject_hits == 0:
            return -1.8

        score = min(subject_hits, 2) * 0.8
        score += min(definition_hits, 4) * 0.28
        score += min(crisis_hits, 5) * 0.34
        if definition_hits >= 1 and crisis_hits >= 1:
            score += 1.0
        if definition_hits >= 2 and crisis_hits >= 2:
            score += 0.9
        if any(marker in text for marker in ("梦", "想象", "狂想", "分支")) and any(
            marker in source_path for marker in BRANCH_SOURCE_MARKERS
        ):
            score -= 0.65
        if subject_hits and crisis_hits == 0 and any(generic in text for generic in ("危机合约", "天灾", "野兽")):
            score -= 1.0
        return score

    def _adjust_chain_score_for_query_type(
        self,
        query: str,
        document: dict,
        chain_score: float,
        query_mode: str | None,
    ) -> float:
        if query_mode != "causality" or not self._is_causal_query(query):
            return chain_score
        support_bonus = self._compute_causality_support_bonus(query, document)
        source_path = str(document.get("source_path") or "")
        if support_bonus <= 0.0:
            scale = 0.18 if any(marker in source_path for marker in PROFILE_SOURCE_MARKERS) else 0.35
            return chain_score * scale - 0.8
        if support_bonus < 0.75:
            return chain_score * 0.7
        return chain_score

    def _compute_query_overlap_bonus(
        self,
        query: str,
        document: dict,
        bridge_terms: list[str],
        *,
        query_mode: str | None = None,
    ) -> float:
        bonus = 0.0
        text = str(document.get("search_text") or document.get("clean_text") or "")
        source_path = str(document.get("source_path") or "")
        source_type = self._document_source_type(document)
        query_terms = self._extract_chain_query_terms(query)
        query_overlap = sum(1 for term in query_terms if term in text)
        long_exact_overlap = sum(1 for term in query_terms if len(term) >= 4 and term in text)
        bridge_overlap = sum(1 for term in bridge_terms if term in text)
        if query_overlap:
            bonus += min(query_overlap, 5) * 0.06
        if long_exact_overlap:
            bonus += min(long_exact_overlap, 2) * 0.28
        original_match_bonus = self._compute_original_query_match_bonus(self._original_query_text(query), text)
        bonus += original_match_bonus * self._original_query_bonus_scale(query_mode)
        if query_mode in LOW_RERANK_QUERY_TYPES and any(len(term) >= 6 and term in text for term in query_terms):
            bonus += 0.45
        if bridge_overlap:
            bonus += min(bridge_overlap, 4) * 0.12
        if query_overlap >= 2 and bridge_overlap >= 2:
            bonus += 0.28
        if "[uc]info" in source_path and bridge_overlap >= 1:
            bonus += 0.2
        if self._is_causal_query(query) and query_overlap >= 2 and bridge_overlap == 0:
            bonus -= 0.1
        if query_mode in {"causality", "reasoning", "answerability"}:
            bonus += self._compute_causality_support_bonus(query, document)
        if query_mode == "answerability" or self._is_concept_crisis_query(query):
            bonus += self._compute_concept_crisis_support_bonus(query, document)
        bonus += self._compute_answerability_bonus(query, document)
        if source_type == "moegirl_background":
            if self._is_background_query(query):
                bonus += 0.75
            elif self._is_story_detail_query(query, query_mode=query_mode):
                bonus -= 1.0
                if long_exact_overlap == 0:
                    bonus -= 0.45
        elif source_type == "story_text" and self._is_story_detail_query(query, query_mode=query_mode):
            bonus += 0.55
        return bonus

    def _compute_answerability_bonus(self, query: str, document: dict) -> float:
        if not self._is_reveal_query(query):
            return 0.0
        text = self._document_text(document)
        source_path = str(document.get("source_path") or "")
        strong_hits = sum(1 for term in REVEAL_ANSWER_TERMS if term in text)
        support_hits = sum(1 for term in REVEAL_SUPPORT_TERMS if term in text)
        direct_hits = sum(1 for term in REVEAL_DIRECT_CONTEXT_TERMS if term in query and term in text)
        if strong_hits == 0 and support_hits < 2 and direct_hits < 2:
            return 0.0
        bonus = min(strong_hits, 5) * 0.85 + min(support_hits, 4) * 0.22 + min(direct_hits, 4) * 0.2
        if "阴谋" in text or "真相" in text:
            bonus += 0.45
        if "[uc]info" in source_path and ("阴谋" in text or "曝光" in text or "真相" in text):
            bonus += 1.25
        if "[uc]info" in source_path and direct_hits >= 2:
            bonus += 0.8
        if strong_hits >= 2 and support_hits >= 2:
            bonus += 1.0
        return min(bonus, 6.0)
