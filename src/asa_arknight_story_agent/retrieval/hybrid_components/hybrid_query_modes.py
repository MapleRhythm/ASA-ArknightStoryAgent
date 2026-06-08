from __future__ import annotations

import re

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import (
    CAUSAL_QUERY_RE,
    CONCEPT_CRISIS_QUERY_RE,
    REVEAL_QUERY_RE,
)


class HybridQueryModesMixin:
    def _is_causal_query(self, query: str) -> bool:
        return bool(CAUSAL_QUERY_RE.search(query))

    def _is_reveal_query(self, query: str) -> bool:
        return bool(REVEAL_QUERY_RE.search(query))

    def _is_concept_crisis_query(self, query: str) -> bool:
        detail_markers = (
            "本名",
            "分别",
            "具体",
            "问题",
            "表现",
            "看待",
            "关系",
            "为什么",
            "为何",
            "如何",
            "怎么",
            "怎样",
        )
        if any(marker in query for marker in detail_markers):
            return False
        has_concept = any(
            term in query
            for term in (
                "是什么组织",
                "是什么势力",
                "是什么国家",
                "是什么地区",
                "是什么种族",
                "是什么概念",
                "是什么设定",
                "本质",
                "来历",
            )
        )
        has_crisis = any(term in query for term in ("危机", "祸", "患", "威胁", "为什么会成为", "为何成为", "为什么成为"))
        return bool(CONCEPT_CRISIS_QUERY_RE.search(query) and (has_concept or has_crisis))

    def _is_background_query(self, query: str) -> bool:
        original_query = self._original_query_text(query)
        detail_markers = (
            "本名",
            "分别",
            "具体",
            "为什么",
            "为何",
            "如何",
            "怎么",
            "怎样",
            "关系",
            "扮演",
            "表现",
            "看待",
            "发生",
            "做了",
            "说了",
            "同意",
            "拒绝",
            "决定",
            "目的",
            "动机",
        )
        explicit_background_markers = ("世界观", "背景", "设定", "介绍", "资料")
        if any(marker in original_query for marker in detail_markers) and not any(
            marker in original_query for marker in explicit_background_markers
        ):
            return False
        background_markers = (
            "是什么组织",
            "是什么势力",
            "是什么国家",
            "是什么地区",
            "是什么种族",
            "是什么概念",
            "是什么设定",
            "介绍",
            "背景",
            "世界观",
            "设定",
            "组织",
            "国家",
            "地区",
            "种族",
            "概念",
            "资料",
            "势力",
        )
        return any(marker in original_query for marker in background_markers) or self._is_concept_crisis_query(original_query)

    def _is_story_detail_query(self, query: str, query_mode: str | None = None) -> bool:
        if self._is_background_query(query):
            return False
        mode = query_mode or self._infer_query_mode(query)
        if mode not in {"fact", "relation", "causality", "reasoning", "reveal", "mystery"}:
            return False
        detail_markers = (
            "为什么",
            "为何",
            "如何",
            "怎么",
            "怎样",
            "具体",
            "本名",
            "分别",
            "关系",
            "扮演",
            "成为",
            "表现",
            "看待",
            "发生",
            "做了",
            "说了",
            "同意",
            "拒绝",
            "决定",
            "目的",
            "动机",
        )
        original_query = self._original_query_text(query)
        return any(marker in original_query for marker in detail_markers)

    def _concept_query_subjects(self, query: str) -> list[str]:
        original_query = self._original_query_text(query)
        compact = "".join(char for char in original_query if "\u4e00" <= char <= "\u9fff")
        subjects: list[str] = []
        for marker in ("是什么", "本质", "来历", "为什么", "为何", "怎么会", "危机", "祸患", "威胁"):
            if marker in compact:
                prefix = compact.split(marker, 1)[0]
                prefix = re.sub(r"(请问|那么|所以|这个|那个|这件事|那件事)$", "", prefix)
                if len(prefix) >= 2:
                    subjects.append(prefix)
                    break
        subjects.extend(
            term
            for term in self._extract_chain_query_terms(original_query)
            if len(term) >= 2
            and not any(noise in term for noise in ("为什么", "为何", "怎么", "如何", "什么", "具体", "危机", "威胁"))
        )
        return list(dict.fromkeys(subjects))[:4]

    def _needs_evidence_chain_query(self, query: str) -> bool:
        return (
            self._is_causal_query(query)
            or self._is_reveal_query(query)
            or self._is_concept_crisis_query(query)
        )

    def _infer_query_mode(self, query: str) -> str:
        original_query = self._original_query_text(query)
        if self._is_reveal_query(original_query):
            return "reveal"
        if self._is_concept_crisis_query(original_query):
            return "answerability"
        if self._is_causal_query(original_query):
            return "causality"
        if any(term in original_query for term in ("关系", "关联")):
            return "relation"
        if any(term in original_query for term in ("是谁", "什么", "哪", "何时", "什么时候", "哪里", "是否", "有没有", "具体")):
            return "fact"
        return "reasoning"
