from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import (
    ACTION_HINT_TERMS,
    ASCII_TOKEN_RE,
    CHAIN_QUERY_STOP_WORDS,
    ENTITY_TOKEN_RE,
    EXPANSION_STOP_WORDS,
    LINE_SPLIT_RE,
    QUERY_CHAR_STOP_CHARS,
)


class HybridQueryTermsMixin:
    def _extract_query_terms(self, query: str) -> list[str]:
        terms: list[str] = []
        for token in ENTITY_TOKEN_RE.findall(query):
            normalized = token.strip()
            if (
                not normalized
                or normalized in EXPANSION_STOP_WORDS
                or len(normalized) == 1
            ):
                continue
            if normalized.isascii() and len(normalized) < 3:
                continue
            terms.append(normalized)
        return list(dict.fromkeys(terms))

    def extract_bridge_terms(self, query: str, candidates: list[dict[str, Any]], *, limit: int = 24) -> list[str]:
        terms = list(self._extract_query_terms(query))
        counts: dict[str, int] = {}
        known = set(terms)
        for item in candidates[:40]:
            text = self._document_text(item["document"])
            weight = 2 if float(item.get("rerank_score") or item.get("fusion_score") or 0.0) > 0 else 1
            for token in ENTITY_TOKEN_RE.findall(text):
                normalized = token.strip()
                if (
                    not normalized
                    or normalized in EXPANSION_STOP_WORDS
                    or normalized in known
                    or len(normalized) == 1
                    or (normalized.isascii() and len(normalized) < 3)
                ):
                    continue
                counts[normalized] = counts.get(normalized, 0) + weight
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        terms.extend(term for term, _count in ranked[: max(0, limit - len(terms))])
        return list(dict.fromkeys(terms))[:limit]

    def _original_query_text(self, query: str) -> str:
        return query.split("\n", 1)[0].strip()

    def _extract_chain_query_terms(self, query: str) -> list[str]:
        terms: list[str] = []
        lowered = query.lower()
        terms.extend(ASCII_TOKEN_RE.findall(lowered))
        cjk_chars = [char for char in query if "\u4e00" <= char <= "\u9fff"]
        cjk_text = "".join(cjk_chars)
        for term in ACTION_HINT_TERMS:
            if term in query:
                terms.append(term)
        for size in range(2, min(7, len(cjk_text) + 1)):
            for idx in range(0, len(cjk_text) - size + 1):
                token = cjk_text[idx : idx + size]
                if (
                    token in CHAIN_QUERY_STOP_WORDS
                    or any(stop in token for stop in ("为什么", "为何", "怎么", "如何", "什么"))
                    or token.endswith(("吗", "呢", "的", "了"))
                ):
                    continue
                terms.append(token)
        return list(dict.fromkeys(term for term in terms if term and len(term) > 1))

    def _extract_original_query_chars(self, query: str) -> list[str]:
        chars = [
            char
            for char in query
            if "\u4e00" <= char <= "\u9fff" and char not in QUERY_CHAR_STOP_CHARS
        ]
        return list(dict.fromkeys(chars))

    def _compute_original_query_match_bonus(self, query: str, text: str) -> float:
        query_chars = self._extract_original_query_chars(query)
        if not query_chars:
            return 0.0

        char_hits = sum(1 for char in query_chars if char in text)
        char_ratio = char_hits / len(query_chars)
        bonus = min(char_hits, 8) * 0.05 + char_ratio * 0.35

        exact_phrases = [
            term
            for term in self._extract_chain_query_terms(query)
            if len(term) >= 4
            and term in text
            and not any(noise in term for noise in ("为什么", "为何", "怎么", "如何", "什么", "具体", "哪件"))
        ]
        if exact_phrases:
            bonus += min(len(exact_phrases), 3) * 0.7
            bonus += min(max(len(term) for term in exact_phrases), 8) * 0.12

        if len(query_chars) >= 3 and char_ratio < 0.35:
            bonus -= 0.35
        return bonus

    def _extract_expansion_terms(self, query: str, candidate_docs: list[dict], *, limit: int = 8) -> list[str]:
        normalized_query = query.lower()
        query_terms = self._extract_query_terms(query)
        scores: dict[str, int] = {}
        for document in candidate_docs:
            text = str(document.get("clean_text") or document.get("search_text") or "")
            lines = [line.strip() for line in LINE_SPLIT_RE.split(text) if line.strip()]
            source_path = str(document.get("source_path") or "")
            doc_query_overlap = sum(1 for term in query_terms if term and term in text)
            doc_weight = 1 + min(doc_query_overlap, 3)
            if "[uc]info" in source_path:
                doc_weight += 1
            seen_in_doc: set[str] = set()
            for segment in document.get("segments") or []:
                if not isinstance(segment, dict):
                    continue
                speaker = str(segment.get("speaker") or "").strip()
                if speaker and speaker not in EXPANSION_STOP_WORDS and speaker.lower() not in normalized_query:
                    scores[speaker] = scores.get(speaker, 0) + 3 + doc_weight
                    seen_in_doc.add(speaker)

            relevant_lines = [
                line for line in lines
                if any(term in line for term in query_terms)
            ]
            if not relevant_lines:
                relevant_lines = lines[:2]
            for line in relevant_lines:
                for token in ENTITY_TOKEN_RE.findall(line):
                    normalized = token.strip()
                    if (
                        not normalized
                        or normalized in EXPANSION_STOP_WORDS
                        or normalized.lower() in normalized_query
                        or normalized in seen_in_doc
                        or len(normalized) == 1
                    ):
                        continue
                    if normalized.isascii() and len(normalized) < 3:
                        continue
                    scores[normalized] = scores.get(normalized, 0) + doc_weight

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [term for term, _score in ranked[:limit]]

    def _build_expansion_queries(self, query: str, candidate_docs: list[dict]) -> list[str]:
        terms = self._extract_expansion_terms(query, candidate_docs)
        if self._is_concept_crisis_query(query):
            terms = list(
                dict.fromkeys(
                    terms
                    + [
                        "本质",
                        "原本",
                        "一体",
                        "巨兽",
                        "代理人",
                        "残躯",
                        "岁陵",
                        "镇压",
                        "苏醒",
                        "消灭",
                        "身殒",
                        "消亡",
                        "代价",
                        "动乱",
                        "灭顶之灾",
                        "开战",
                        "平息",
                        "解决",
                    ]
                )
            )
        original_query = self._original_query_text(query)
        original_query_terms = [
            term
            for term in self._extract_chain_query_terms(original_query)
            if len(term) >= 3 and not any(noise in term for noise in ("为什么", "为何", "怎么", "如何", "什么", "具体", "哪件"))
        ]
        terms = list(dict.fromkeys(original_query_terms[:8] + terms))
        if not terms:
            return []
        queries: list[str] = []
        first_terms = terms[:4]
        if first_terms:
            queries.append(f"{query} {' '.join(first_terms)}")
        second_terms = terms[4:8]
        if second_terms:
            queries.append(f"{query} {' '.join(second_terms)}")
        if len(terms) >= 6:
            queries.append(f"{query} {' '.join(terms[:2])} {' '.join(terms[-2:])}")
        return list(dict.fromkeys(queries))
