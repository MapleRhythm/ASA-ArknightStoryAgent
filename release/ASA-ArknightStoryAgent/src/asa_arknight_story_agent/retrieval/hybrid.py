from __future__ import annotations

import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from asa_arknight_story_agent.config import BM25_TOKENS_PATH, DOCUMENTS_PATH, FAISS_INDEX_PATH, QueryConfig
from asa_arknight_story_agent.retrieval.minirag import MiniRAGIndex
from asa_arknight_story_agent.retrieval.reranker import CrossEncoderReranker


ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
ENTITY_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
CAUSAL_QUERY_RE = re.compile(r"为什么|为何|原因|动机|目的|怎么会|为何要|为什么要|为什么会")
REVEAL_QUERY_RE = re.compile(r"阴谋|真相|秘密|识破|揭穿|曝光|暴露|幕后|主使|黑幕|骗局|诡计|怎么回事")
CONCEPT_CRISIS_QUERY_RE = re.compile(r"是什么|本质|来历|为何成为|为什么成为|为什么会成为|危机|祸|患|威胁")
LINE_SPLIT_RE = re.compile(r"[\n\r。！？；]+")
STAGE_NUMBER_RE = re.compile(r"(?:^|[_\-/])(?:level_)?[a-z0-9]+_(\d{2})(?:[_a-z\-/]|$)", re.IGNORECASE)
QUERY_CHAR_STOP_CHARS = set("的是了嘛吗呢啊吧呀么什怎为哪件这那其一在和与及或把被让给从向对将会要请问具体指")
LOW_RERANK_QUERY_TYPES = {"fact", "relation"}
HIGH_RERANK_QUERY_TYPES = {"causality", "reasoning", "reveal", "mystery", "answerability"}
QUERY_TYPES = LOW_RERANK_QUERY_TYPES | HIGH_RERANK_QUERY_TYPES
PROFILE_SOURCE_MARKERS = ("handbook_info_table.json", "charword_table.json")
BRANCH_SOURCE_MARKERS = ("/rogue/", "/roguelike/", "/endbook/")
CONCEPT_DEFINITION_TERMS = {
    "本质",
    "原本",
    "源自",
    "一体",
    "巨兽",
    "神明",
    "代理人",
    "残躯",
    "身躯",
    "沉睡",
    "岁陵",
    "镇压",
}
CONCEPT_CRISIS_TERMS = {
    "苏醒",
    "消灭",
    "身殒",
    "消亡",
    "危害",
    "危机",
    "祸",
    "患",
    "威胁",
    "动乱",
    "开战",
    "进攻",
    "代价",
    "生灵涂炭",
    "尸横遍野",
    "十不存一",
    "灭顶之灾",
}
EXPANSION_STOP_WORDS = {
    "博士",
    "为什么",
    "为何",
    "原因",
    "动机",
    "目的",
    "关闭",
    "开启",
    "进入",
    "离开",
    "全舰",
    "防御系统",
    "系统",
    "因为",
    "所以",
    "就是",
    "然后",
    "这个",
    "那个",
    "他们",
    "我们",
    "你们",
    "自己",
    "已经",
    "还是",
    "没有",
    "不会",
    "一次",
    "本来",
    "开始",
    "完成",
    "行动",
    "战斗",
    "城市",
    "计划",
    "路线",
    "数据",
    "确认",
    "窗口",
    "安全窗口",
}
CHAIN_QUERY_STOP_WORDS = EXPANSION_STOP_WORDS | {
    "什么",
    "怎么",
    "如何",
    "是谁",
    "是怎么",
    "要",
    "会",
    "是",
    "的",
}
ACTION_HINT_TERMS = {
    "关闭",
    "解除",
    "开启",
    "启动",
    "进入",
    "离开",
    "杀死",
    "刺杀",
    "攻击",
    "摧毁",
    "背叛",
    "保护",
    "放弃",
    "加入",
    "成为",
    "形成",
    "控制",
    "识破",
    "揭穿",
    "曝光",
    "暴露",
    "苏醒",
    "消灭",
    "镇压",
    "开战",
    "围猎",
    "狩猎",
}
MOTIVE_HINT_TERMS = {
    "因为",
    "为了",
    "目的",
    "动机",
    "原因",
    "约定",
    "交易",
    "计划",
    "打算",
    "决定",
    "选择",
    "必须",
    "不得不",
    "不必",
    "意味着",
    "会让",
    "让",
    "代价",
    "威胁",
    "危机",
    "祸",
    "患",
    "存亡",
    "一体",
}
OUTCOME_HINT_TERMS = {
    "导致",
    "最终",
    "于是",
    "长驱直入",
    "刺杀",
    "死亡",
    "死",
    "解除",
    "失去",
    "失败",
    "真相",
    "证据",
    "曝光",
    "暴露",
    "识破",
    "揭穿",
    "败露",
    "苏醒",
    "消亡",
    "灭顶之灾",
    "动乱",
    "开战",
    "平息",
    "解决",
    "祸",
    "患",
    "危机",
}
TARGET_CONTEXT_HINT_TERMS = {
    "留守",
    "目标",
    "对象",
    "对付",
    "刺客",
    "暗杀",
}
REVEAL_ANSWER_TERMS = {
    "不能留下证据",
    "留下证据",
    "证据",
    "报告损失",
    "拨给我的钱",
    "钱的窟窿",
    "窟窿",
    "栽赃",
    "嫁祸",
    "被感染者",
    "炸了",
    "炸掉",
    "偷走",
    "内鬼",
    "灭口",
    "绑架",
    "解决掉",
    "同伙",
    "主使",
    "幕后",
    "曝光",
    "败露",
    "全完了",
}
REVEAL_SUPPORT_TERMS = {
    "工厂",
    "设备",
    "物流通道",
    "地下",
    "警备队",
    "议会",
    "拨款",
    "贵族",
    "钱",
    "计划",
}


def tokenize_for_bm25(text: str) -> list[str]:
    lowered = text.lower()
    ascii_tokens = ASCII_TOKEN_RE.findall(lowered)
    cjk_chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    cjk_bigrams = [f"{cjk_chars[i]}{cjk_chars[i + 1]}" for i in range(len(cjk_chars) - 1)]
    return ascii_tokens + cjk_bigrams


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ArknightsHybridRetriever:
    def __init__(
        self,
        documents: list[dict],
        index: faiss.Index,
        bm25: BM25Okapi,
        embedding_model: SentenceTransformer,
        reranker: CrossEncoderReranker | None = None,
        minirag_index: MiniRAGIndex | None = None,
    ) -> None:
        self.documents = documents
        self.index = index
        self.bm25 = bm25
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.minirag_index = minirag_index
        self.story_doc_indices: dict[str, list[int]] = {}
        self.stage_doc_indices: dict[tuple[str, str], list[int]] = {}
        self.activity_story_sort_doc_indices: dict[str, dict[int, list[int]]] = {}
        for doc_index, document in enumerate(documents):
            story_id = str(document.get("story_id") or "").strip()
            if story_id:
                self.story_doc_indices.setdefault(story_id, []).append(doc_index)
            activity_id = str(document.get("activity_id") or "").strip()
            stage_code = str(document.get("stage_code") or "").strip()
            if activity_id and stage_code:
                self.stage_doc_indices.setdefault((activity_id, stage_code), []).append(doc_index)
            story_sort = document.get("story_sort")
            if activity_id and isinstance(story_sort, int):
                self.activity_story_sort_doc_indices.setdefault(activity_id, {}).setdefault(story_sort, []).append(doc_index)

    @classmethod
    def from_paths(
        cls,
        *,
        embedding_model_path: Path,
        reranker_model_path: Path | None = None,
        reranker_max_length: int = 1024,
        documents_path: Path = DOCUMENTS_PATH,
        faiss_index_path: Path = FAISS_INDEX_PATH,
        bm25_tokens_path: Path = BM25_TOKENS_PATH,
        minirag_index_path: Path | None = None,
        device: str = "cpu",
    ) -> "ArknightsHybridRetriever":
        started = time.time()
        print(f"[retriever-load] documents {documents_path}", file=sys.stderr, flush=True)
        documents = load_jsonl(documents_path)
        print(f"[retriever-load] documents={len(documents)} elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        print(f"[retriever-load] faiss {faiss_index_path}", file=sys.stderr, flush=True)
        index = faiss.read_index(str(faiss_index_path))
        print(f"[retriever-load] faiss loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        print(f"[retriever-load] bm25 {bm25_tokens_path}", file=sys.stderr, flush=True)
        with bm25_tokens_path.open("rb") as handle:
            tokenized_corpus = pickle.load(handle)
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"[retriever-load] bm25 loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        print(f"[retriever-load] embedding {embedding_model_path} device={device}", file=sys.stderr, flush=True)
        embedding_model = SentenceTransformer(str(embedding_model_path), device=device)
        print(f"[retriever-load] embedding loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        reranker = None
        if reranker_model_path and reranker_model_path.exists():
            print(f"[retriever-load] reranker {reranker_model_path} device={device}", file=sys.stderr, flush=True)
            reranker = CrossEncoderReranker(reranker_model_path, device=device, max_length=reranker_max_length)
            print(f"[retriever-load] reranker loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        minirag_index = None
        resolved_minirag_path = minirag_index_path or None
        if resolved_minirag_path and resolved_minirag_path.exists():
            print(f"[retriever-load] minirag {resolved_minirag_path}", file=sys.stderr, flush=True)
            minirag_index = MiniRAGIndex.load(resolved_minirag_path)
            print(f"[retriever-load] minirag loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        print(f"[retriever-load] done elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        return cls(
            documents=documents,
            index=index,
            bm25=bm25,
            embedding_model=embedding_model,
            reranker=reranker,
            minirag_index=minirag_index,
        )

    def dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        vector = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores, indices = self.index.search(vector.astype(np.float32), top_k)
        hits: list[dict[str, Any]] = []
        for score, doc_index in zip(scores[0].tolist(), indices[0].tolist()):
            if doc_index < 0:
                continue
            doc = self.documents[doc_index]
            hits.append(
                {
                    "doc_index": doc_index,
                    "score": float(score),
                    "document": doc,
                }
            )
        return hits

    def sparse_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        hits: list[dict[str, Any]] = []
        for doc_index in top_indices.tolist():
            score = float(scores[doc_index])
            if score <= 0:
                continue
            hits.append(
                {
                    "doc_index": int(doc_index),
                    "score": score,
                    "document": self.documents[int(doc_index)],
                }
            )
        return hits

    def minirag_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.minirag_index is None:
            return []
        return self.minirag_index.search(query, self.documents, top_k=top_k)

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

    def expand_hits_with_neighbors(
        self,
        hits: list[dict[str, Any]],
        *,
        max_seed_docs: int = 24,
        story_window: int = 2,
        activity_story_sort_window: int = 1,
        top_k: int = 120,
    ) -> list[dict[str, Any]]:
        if not hits:
            return hits
        neighbor_doc_indices = self._collect_story_and_stage_neighbors(
            hits,
            max_seed_docs=max_seed_docs,
            story_window=story_window,
            activity_story_sort_window=activity_story_sort_window,
        )
        merged = [dict(item) for item in hits]
        seen = {int(item["doc_index"]) for item in merged}
        neighbor_rank = 0
        for doc_index in neighbor_doc_indices:
            if doc_index in seen or not (0 <= doc_index < len(self.documents)):
                continue
            seen.add(doc_index)
            neighbor_rank += 1
            merged.append(
                {
                    "doc_index": doc_index,
                    "document": self.documents[doc_index],
                    "dense_score": None,
                    "sparse_score": None,
                    "minirag_score": None,
                    "fusion_score": 0.0,
                    "supplemental_source": "neighbor",
                    "supplemental_rank": neighbor_rank,
                }
            )
            if len(merged) >= top_k:
                break
        return merged[:top_k]

    def _is_causal_query(self, query: str) -> bool:
        return bool(CAUSAL_QUERY_RE.search(query))

    def _is_reveal_query(self, query: str) -> bool:
        return bool(REVEAL_QUERY_RE.search(query))

    def _is_concept_crisis_query(self, query: str) -> bool:
        has_concept = any(term in query for term in ("是什么", "本质", "来历"))
        has_crisis = any(term in query for term in ("危机", "祸", "患", "威胁", "为什么会成为", "为何成为", "为什么成为"))
        return bool(CONCEPT_CRISIS_QUERY_RE.search(query) and (has_concept or has_crisis))

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
            item["fusion_score"] = max(float(item.get("fusion_score") or 0.0), float(hit.get("fusion_score") or 0.0) * score_scale)

    def _collect_story_and_stage_neighbors(
        self,
        seed_hits: list[dict[str, Any]],
        *,
        max_seed_docs: int = 6,
        story_window: int = 2,
        activity_story_sort_window: int = 1,
    ) -> list[int]:
        candidate_indices: list[int] = []
        for hit in seed_hits[:max_seed_docs]:
            doc = hit["document"]
            doc_index = int(hit["doc_index"])
            if doc_index not in candidate_indices:
                candidate_indices.append(doc_index)

            story_id = str(doc.get("story_id") or "").strip()
            if story_id:
                story_indices = self.story_doc_indices.get(story_id, [])
                try:
                    current_pos = story_indices.index(doc_index)
                except ValueError:
                    current_pos = -1
                if current_pos >= 0:
                    start = max(0, current_pos - story_window)
                    end = min(len(story_indices), current_pos + story_window + 1)
                    for neighbor in story_indices[start:end]:
                        if neighbor not in candidate_indices:
                            candidate_indices.append(neighbor)

            activity_id = str(doc.get("activity_id") or "").strip()
            stage_code = str(doc.get("stage_code") or "").strip()
            if activity_id and stage_code:
                for neighbor in self.stage_doc_indices.get((activity_id, stage_code), []):
                    if neighbor not in candidate_indices:
                        candidate_indices.append(neighbor)
            story_sort = doc.get("story_sort")
            if activity_id and isinstance(story_sort, int):
                for offset in range(-activity_story_sort_window, activity_story_sort_window + 1):
                    neighbor_sort = story_sort + offset
                    for neighbor in self.activity_story_sort_doc_indices.get(activity_id, {}).get(neighbor_sort, []):
                        if neighbor not in candidate_indices:
                            candidate_indices.append(neighbor)
        return candidate_indices

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

    def _original_query_bonus_scale(self, query_mode: str | None) -> float:
        if query_mode in LOW_RERANK_QUERY_TYPES:
            return 0.45
        if query_mode in HIGH_RERANK_QUERY_TYPES:
            return 0.3
        return 0.6

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
        return bonus

    def _compute_answerability_bonus(self, query: str, document: dict) -> float:
        if not self._is_reveal_query(query):
            return 0.0
        text = self._document_text(document)
        source_path = str(document.get("source_path") or "")
        strong_hits = sum(1 for term in REVEAL_ANSWER_TERMS if term in text)
        support_hits = sum(1 for term in REVEAL_SUPPORT_TERMS if term in text)
        if strong_hits == 0 and support_hits < 2:
            return 0.0
        bonus = min(strong_hits, 5) * 0.85 + min(support_hits, 4) * 0.22
        if "阴谋" in text or "真相" in text:
            bonus += 0.45
        if "[uc]info" in source_path and ("阴谋" in text or "曝光" in text or "真相" in text):
            bonus += 1.25
        if strong_hits >= 2 and support_hits >= 2:
            bonus += 1.0
        return min(bonus, 6.0)

    def _document_text(self, document: dict) -> str:
        return str(document.get("search_text") or document.get("clean_text") or "")

    def _document_stage_number(self, document: dict) -> int | None:
        stage_code = str(document.get("stage_code") or "")
        stage_match = re.search(r"(\d+)", stage_code)
        if stage_match:
            return int(stage_match.group(1))

        source = " ".join(
            str(document.get(key) or "")
            for key in ("story_id", "story_key", "source_path", "id")
        )
        source_match = STAGE_NUMBER_RE.search(source)
        if source_match:
            return int(source_match.group(1))
        return None

    def _document_chain_terms(self, document: dict, bridge_terms: list[str]) -> set[str]:
        text = self._document_text(document)
        terms = {term for term in bridge_terms if term and term in text}
        for segment in document.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            speaker = str(segment.get("speaker") or "").strip()
            if not speaker:
                continue
            terms.add(speaker)
            if len(speaker) > 2 and speaker.endswith(("刺客", "主教", "成员", "干员")):
                terms.add(speaker[-2:])
        return terms

    def _classify_evidence_roles(
        self,
        query: str,
        document: dict,
        *,
        query_terms: list[str],
        bridge_terms: list[str],
    ) -> set[str]:
        text = self._document_text(document)
        roles: set[str] = set()
        query_overlap = sum(1 for term in query_terms if term in text)
        bridge_overlap = sum(1 for term in bridge_terms if term in text)
        query_action_terms = [term for term in ACTION_HINT_TERMS if term in query]

        if query_action_terms and any(term in text for term in query_action_terms):
            roles.add("action")
        if query_overlap >= 3:
            roles.add("action")
        if bridge_overlap >= 2:
            roles.add("context")
        if any(term in text for term in MOTIVE_HINT_TERMS) and (bridge_overlap or query_overlap):
            roles.add("motive")
        if any(term in text for term in OUTCOME_HINT_TERMS) and (bridge_overlap or query_overlap):
            roles.add("outcome")
        if any(term in text for term in TARGET_CONTEXT_HINT_TERMS) and bridge_overlap:
            roles.add("target")
        if self._is_reveal_query(query):
            answerability = self._compute_answerability_bonus(query, document)
            if answerability >= 1.5:
                roles.add("outcome")
            if answerability >= 2.5:
                roles.add("motive")
            if answerability >= 3.5:
                roles.add("action")
        if "本舰" in text and (bridge_overlap or query_overlap):
            roles.add("context")
        if "[uc]info" in str(document.get("source_path") or "") and (bridge_overlap or query_overlap):
            roles.add("context")
            if any(term in text for term in OUTCOME_HINT_TERMS | TARGET_CONTEXT_HINT_TERMS):
                roles.add("outcome")
        return roles

    def _document_link_score(self, left: dict, right: dict, *, bridge_terms: list[str]) -> float:
        if left is right:
            return 1.0
        score = 0.0
        left_story = str(left.get("story_id") or "")
        right_story = str(right.get("story_id") or "")
        if left_story and left_story == right_story:
            score += 0.45

        left_activity = str(left.get("activity_id") or "")
        right_activity = str(right.get("activity_id") or "")
        if left_activity and left_activity == right_activity:
            score += 0.2
            left_stage = str(left.get("stage_code") or "")
            right_stage = str(right.get("stage_code") or "")
            if left_stage and left_stage == right_stage:
                score += 0.25

            left_sort = left.get("story_sort")
            right_sort = right.get("story_sort")
            if isinstance(left_sort, int) and isinstance(right_sort, int):
                distance = abs(left_sort - right_sort)
                if distance <= 1:
                    score += 0.25
                elif distance <= 3:
                    score += 0.12

            left_stage_number = self._document_stage_number(left)
            right_stage_number = self._document_stage_number(right)
            if left_stage_number is not None and right_stage_number is not None:
                distance = abs(left_stage_number - right_stage_number)
                if distance == 0:
                    score += 0.22
                elif distance == 1:
                    score += 0.14

        shared_terms = self._document_chain_terms(left, bridge_terms) & self._document_chain_terms(right, bridge_terms)
        if shared_terms:
            score += min(len(shared_terms), 3) * 0.08
        return min(score, 1.0)

    def _build_evidence_chains(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        bridge_terms: list[str],
    ) -> list[dict[str, Any]]:
        query_terms = self._extract_chain_query_terms(query)
        scored_items: list[dict[str, Any]] = []
        for item in candidates:
            document = item["document"]
            roles = self._classify_evidence_roles(
                query,
                document,
                query_terms=query_terms,
                bridge_terms=bridge_terms,
            )
            text = self._document_text(document)
            scored_items.append(
                {
                    "item": item,
                    "roles": roles,
                    "query_overlap": sum(1 for term in query_terms if term in text),
                    "bridge_overlap": sum(1 for term in bridge_terms if term in text),
                    "base_score": float(item.get("rerank_score") or item.get("fusion_score") or 0.0),
                }
            )

        seeds = [
            scored
            for scored in scored_items
            if scored["roles"] or scored["query_overlap"] >= 2 or scored["bridge_overlap"] >= 2
        ]
        seeds = sorted(seeds, key=lambda scored: scored["base_score"], reverse=True)[:24]
        chains: list[dict[str, Any]] = []
        for seed in seeds:
            members: list[dict[str, Any]] = [seed]
            for scored in scored_items:
                if scored is seed:
                    continue
                link_score = self._document_link_score(
                    seed["item"]["document"],
                    scored["item"]["document"],
                    bridge_terms=bridge_terms,
                )
                if link_score < 0.4:
                    continue
                member = dict(scored)
                member["link_score"] = link_score
                members.append(member)

            members = sorted(
                members,
                key=lambda member: (
                    len(member["roles"]),
                    member.get("link_score", 1.0),
                    member["bridge_overlap"],
                    member["base_score"],
                ),
                reverse=True,
            )[:8]
            role_union = set().union(*(member["roles"] for member in members))
            if len(role_union) < 2:
                continue
            if "action" in role_union and not ({"outcome", "target"} & role_union):
                continue
            if "action" not in role_union and not {"motive", "outcome"}.issubset(role_union):
                continue
            link_scores = [
                self._document_link_score(
                    left["item"]["document"],
                    right["item"]["document"],
                    bridge_terms=bridge_terms,
                )
                for left in members
                for right in members
                if left is not right
            ]
            average_link_score = sum(link_scores) / len(link_scores) if link_scores else 0.0
            role_score = 0.0
            role_score += 0.35 if "action" in role_union else 0.0
            role_score += 0.3 if "outcome" in role_union else 0.0
            role_score += 0.25 if "motive" in role_union else 0.0
            role_score += 0.18 if "target" in role_union else 0.0
            role_score += 0.12 if "context" in role_union else 0.0
            if {"action", "outcome"}.issubset(role_union):
                role_score += 0.25
            if {"action", "motive"}.issubset(role_union):
                role_score += 0.18
            query_coverage = len({term for term in query_terms if any(term in self._document_text(member["item"]["document"]) for member in members)})
            bridge_coverage = len({term for term in bridge_terms if any(term in self._document_text(member["item"]["document"]) for member in members)})
            chain_score = (
                role_score
                + min(query_coverage, 8) * 0.04
                + min(bridge_coverage, 6) * 0.06
                + average_link_score * 0.45
                + min(len(members), 5) * 0.03
            )
            chains.append(
                {
                    "score": chain_score,
                    "roles": role_union,
                    "members": members,
                }
            )
        return sorted(chains, key=lambda chain: chain["score"], reverse=True)

    def _build_generic_evidence_chains(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        bridge_terms: list[str],
    ) -> list[dict[str, Any]]:
        query_terms = self._extract_chain_query_terms(query)
        scored_items: list[dict[str, Any]] = []
        for item in candidates:
            document = item["document"]
            text = self._document_text(document)
            roles = self._classify_evidence_roles(
                query,
                document,
                query_terms=query_terms,
                bridge_terms=bridge_terms,
            )
            query_overlap = sum(1 for term in query_terms if term in text)
            bridge_overlap = sum(1 for term in bridge_terms if term in text)
            scored_items.append(
                {
                    "item": item,
                    "roles": roles or ({"context"} if query_overlap or bridge_overlap else set()),
                    "query_overlap": query_overlap,
                    "bridge_overlap": bridge_overlap,
                    "base_score": float(item.get("rerank_score") or item.get("fusion_score") or 0.0),
                }
            )

        seeds = [
            scored
            for scored in scored_items
            if scored["query_overlap"] > 0 or scored["bridge_overlap"] > 0 or scored["base_score"] > 0
        ]
        seeds = sorted(
            seeds,
            key=lambda scored: (
                scored["query_overlap"],
                scored["bridge_overlap"],
                len(scored["roles"]),
                scored["base_score"],
            ),
            reverse=True,
        )[:32]

        chains: list[dict[str, Any]] = []
        seen_chain_keys: set[tuple[int, ...]] = set()
        for seed in seeds:
            linked_members: list[dict[str, Any]] = [seed]
            for scored in scored_items:
                if scored is seed:
                    continue
                link_score = self._document_link_score(
                    seed["item"]["document"],
                    scored["item"]["document"],
                    bridge_terms=bridge_terms,
                )
                if link_score < 0.25:
                    continue
                member = dict(scored)
                member["link_score"] = link_score
                linked_members.append(member)

            linked_members = sorted(
                linked_members,
                key=lambda member: (
                    member is seed,
                    member.get("link_score", 1.0),
                    member["query_overlap"],
                    member["bridge_overlap"],
                    len(member["roles"]),
                    member["base_score"],
                ),
                reverse=True,
            )[:5]
            if len(linked_members) < 2:
                continue
            chain_key = tuple(
                dict.fromkeys(
                    int(member["item"]["doc_index"])
                    for member in sorted(linked_members, key=self._chain_member_sort_key)
                )
            )
            if chain_key in seen_chain_keys:
                continue
            seen_chain_keys.add(chain_key)
            role_union = set().union(*(member["roles"] for member in linked_members))
            query_coverage = len(
                {
                    term
                    for term in query_terms
                    if any(term in self._document_text(member["item"]["document"]) for member in linked_members)
                }
            )
            bridge_coverage = len(
                {
                    term
                    for term in bridge_terms
                    if any(term in self._document_text(member["item"]["document"]) for member in linked_members)
                }
            )
            average_base_score = sum(member["base_score"] for member in linked_members) / len(linked_members)
            chains.append(
                {
                    "score": (
                        min(query_coverage, 8) * 0.12
                        + min(bridge_coverage, 6) * 0.08
                        + min(len(role_union), 4) * 0.05
                        + average_base_score * 0.1
                    ),
                    "roles": role_union,
                    "members": linked_members,
                }
            )
        return sorted(chains, key=lambda chain: chain["score"], reverse=True)

    def _chain_member_sort_key(self, member: dict[str, Any]) -> tuple[str, int, int, int]:
        document = member["item"]["document"]
        return (
            str(document.get("story_id") or document.get("activity_id") or ""),
            int(document.get("story_sort") if isinstance(document.get("story_sort"), int) else 10**9),
            self._document_stage_number(document) or 10**9,
            int(member["item"].get("doc_index") or 0),
        )

    def _render_chain_text(self, chain: dict[str, Any]) -> str:
        members = sorted(chain["members"], key=self._chain_member_sort_key)
        lines: list[str] = []
        seen: set[int] = set()
        for index, member in enumerate(members, start=1):
            item = member["item"]
            doc_index = int(item["doc_index"])
            if doc_index in seen:
                continue
            seen.add(doc_index)
            document = item["document"]
            prefix_parts = [
                str(document.get("activity_name") or ""),
                str(document.get("story_name") or ""),
                str(document.get("stage_code") or ""),
            ]
            prefix = " / ".join(part for part in prefix_parts if part)
            text = str(document.get("clean_text") or document.get("search_text") or "").strip()
            if prefix:
                lines.append(f"[E{index}] {prefix}\n{text}")
            else:
                lines.append(f"[E{index}] {text}")
        return "\n".join(lines)

    def _apply_chain_model_scores(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        bridge_terms: list[str],
        batch_size: int,
        query_mode: str | None = None,
    ) -> bool:
        if not self.reranker:
            return False
        chains = self._build_evidence_chains(query, candidates, bridge_terms=bridge_terms)
        if not chains:
            chains = self._build_generic_evidence_chains(query, candidates, bridge_terms=bridge_terms)
        if not chains:
            return False

        chain_payloads: list[dict[str, Any]] = []
        seen_chain_keys: set[tuple[int, ...]] = set()
        for chain in chains[: max(24, min(80, len(chains)))]:
            member_items = sorted(chain["members"], key=self._chain_member_sort_key)
            member_indices = tuple(dict.fromkeys(int(member["item"]["doc_index"]) for member in member_items))
            if len(member_indices) < 2 or member_indices in seen_chain_keys:
                continue
            seen_chain_keys.add(member_indices)
            chain_text = self._render_chain_text(chain)
            if not chain_text:
                continue
            chain_payloads.append(
                {
                    "chain": chain,
                    "member_indices": member_indices,
                    "chain_text": chain_text,
                }
            )
        if not chain_payloads:
            return False

        chain_scores = self.reranker.score(
            query=query,
            documents=[payload["chain_text"] for payload in chain_payloads],
            batch_size=batch_size,
        )
        doc_index_to_item = {int(item["doc_index"]): item for item in candidates}
        for rank, (payload, score) in enumerate(
            sorted(
                zip(chain_payloads, chain_scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ):
            chain = payload["chain"]
            normalized_rank_bonus = 1.0 / (rank + 1)
            for member in chain["members"]:
                item = doc_index_to_item.get(int(member["item"]["doc_index"]))
                if item is None:
                    continue
                role_bonus = 0.07 * len(member["roles"])
                if "action" in member["roles"]:
                    role_bonus += 0.12
                if "outcome" in member["roles"]:
                    role_bonus += 0.11
                if "motive" in member["roles"]:
                    role_bonus += 0.1
                if query_mode in LOW_RERANK_QUERY_TYPES:
                    chain_member_score = float(score) * 0.45 + normalized_rank_bonus * 0.35 + role_bonus * 0.35
                else:
                    chain_member_score = float(score) * 1.5 + normalized_rank_bonus * 1.3 + role_bonus
                item["evidence_chain_score"] = max(
                    float(item.get("evidence_chain_score", float("-inf"))),
                    chain_member_score,
                )
                item["evidence_chain_model_score"] = max(
                    float(item.get("evidence_chain_model_score", float("-inf"))),
                    float(score),
                )
                item["evidence_chain_roles"] = sorted(member["roles"])
                item["evidence_chain_text"] = payload["chain_text"]
        return True

    def rerank_with_evidence_chains(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        top_k: int,
        batch_size: int,
        bridge_terms: list[str] | None = None,
        query_mode: str | None = None,
        fallback_to_document_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        if not hits:
            return []
        candidates = [dict(item) for item in hits]
        if self.reranker:
            scores = self.reranker.score(
                query=query,
                documents=[item["document"]["search_text"] for item in candidates],
                batch_size=batch_size,
            )
            for item, score in zip(candidates, scores, strict=True):
                item["rerank_score"] = float(score)
            return sorted(
                candidates,
                key=lambda item: item.get("rerank_score", float("-inf")),
                reverse=True,
            )[:top_k]

        for item in candidates:
            item["rerank_score"] = float(item.get("fusion_score") or 0.0)
        return sorted(
            candidates,
            key=lambda item: item.get("rerank_score", float("-inf")),
            reverse=True,
        )[:top_k]

    def _apply_evidence_chain_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        bridge_terms: list[str],
    ) -> None:
        chains = self._build_evidence_chains(query, candidates, bridge_terms=bridge_terms)
        if not chains:
            self._apply_causal_focus_adjustment(query, candidates)
            return
        query_terms = self._extract_chain_query_terms(query)
        for chain_rank, chain in enumerate(chains[:3]):
            chain_bonus = min(chain["score"], 1.25) * (0.85 ** chain_rank)
            for member in chain["members"]:
                item = member["item"]
                text = self._document_text(item["document"])
                roles = member["roles"]
                long_exact_overlap = sum(1 for term in query_terms if len(term) >= 4 and term in text)
                bridge_overlap = sum(1 for term in bridge_terms if term in text)
                contributes_target_or_outcome = bool({"target", "outcome"} & roles)
                contributes_exact_action = "action" in roles and long_exact_overlap > 0
                contributes_motive = "motive" in roles and bridge_overlap >= 2 and any(
                    term in text for term in ("杀死", "约定", "交易", "计划", "目的", "决定")
                )
                if not (contributes_target_or_outcome or contributes_exact_action or contributes_motive):
                    continue
                role_bonus = 0.08 * len(member["roles"])
                if "action" in member["roles"]:
                    role_bonus += 0.12
                if "outcome" in member["roles"]:
                    role_bonus += 0.11
                if "motive" in member["roles"]:
                    role_bonus += 0.1
                bonus = chain_bonus + role_bonus
                item["chain_bonus"] = max(float(item.get("chain_bonus") or 0.0), bonus)
                item["evidence_chain_roles"] = sorted(member["roles"])
        for item in candidates:
            item["rerank_score"] = float(item.get("rerank_score") or 0.0) + float(item.get("chain_bonus") or 0.0)
        self._apply_causal_focus_adjustment(query, candidates)

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
                top_k=query_config.fusion_top_k,
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
                top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
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
