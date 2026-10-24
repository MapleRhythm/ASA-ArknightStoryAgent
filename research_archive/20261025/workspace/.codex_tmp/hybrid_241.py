from __future__ import annotations

import json
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from goldenglow.config import (
    BM25_TOKENS_PATH,
    CORPUS_METADATA_PATH,
    DOCUMENTS_PATH,
    FAISS_INDEX_PATH,
    SPARSE_INDEX_PATH,
    QueryConfig,
)
from goldenglow.retrieval.minirag import MiniRAGIndex, document_chapter_scope_key
from goldenglow.retrieval.reranker import CrossEncoderReranker
from goldenglow.retrieval.storyline import document_storyline_scopes


ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
ASCII_EXACT_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[_.\-/][a-z0-9]+)+|[a-z0-9_]+", re.IGNORECASE)
CJK_SPAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
NATURAL_ALIAS_RE = re.compile(r"^(?:[\u3400-\u4dbf\u4e00-\u9fff·]{1,24}|[A-Za-z][A-Za-z .'-]{0,31})$")
ENTITY_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
CAUSAL_QUERY_RE = re.compile(r"为什么|为何|原因|动机|目的|怎么会|为何要|为什么要|为什么会")
REVEAL_QUERY_RE = re.compile(r"阴谋|真相|秘密|识破|揭穿|曝光|暴露|幕后|主使|黑幕|骗局|诡计|怎么回事")
CONCEPT_CRISIS_QUERY_RE = re.compile(r"是什么|本质|来历|为何成为|为什么成为|为什么会成为|危机|祸|患|威胁")
LINE_SPLIT_RE = re.compile(r"[\n\r。！？；]+")
STAGE_NUMBER_RE = re.compile(r"(?:^|[_\-/])(?:level_)?[a-z0-9]+_(\d{2})(?:[_a-z\-/]|$)", re.IGNORECASE)
MAIN_CHAPTER_REF_RE = re.compile(
    r"(?:第\s*([一二三四五六七八九十百零〇两0-9]{1,4})\s*章|([0-9]{1,2})\s*章|level_main[_-]([0-9]{1,2})|main[_-]([0-9]{1,2})|EPISODE\s*([0-9]{1,2}))",
    re.IGNORECASE,
)
MAIN_CHAPTER_SOURCE_RE = re.compile(r"(?:^|[/_])(?:level_)?main[_-](\d{1,2})(?:[-_/]|$)", re.IGNORECASE)
QUERY_CHAR_STOP_CHARS = set("的是了嘛吗呢啊吧呀么什怎为哪件这那其一在和与及或把被让给从向对将会要请问具体指")
LOW_RERANK_QUERY_TYPES = {"fact", "relation"}
HIGH_RERANK_QUERY_TYPES = {"causality", "reasoning", "reveal", "mystery", "answerability"}
QUERY_TYPES = LOW_RERANK_QUERY_TYPES | HIGH_RERANK_QUERY_TYPES
PROFILE_SOURCE_MARKERS = ("handbook_info_table.json", "charword_table.json")
BRANCH_SOURCE_MARKERS = ("/rogue/", "/roguelike/", "/endbook/")
MOEGIRL_SOURCE_MARKERS = ("moegirl.icu", "moegirl/", "萌百世界观资料")
STORY_SOURCE_MARKERS = (
    "activities/",
    "obt/main/",
    "obt/memory/",
    "obt/rogue/",
    "[uc]info/activities/",
)
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
    "启用",
    "动用",
    "使用",
    "发动",
    "打开",
    "建造",
    "修建",
    "建设",
    "制造",
    "改造",
    "布局",
    "设下",
    "安排",
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
REVEAL_DIRECT_CONTEXT_TERMS = {
    "澄闪",
    "苏茜",
    "苏茜·格里特",
    "卡拉顿",
    "卡拉顿城",
    "贝希曼",
    "贝希曼伯爵",
    "阴云火花",
}


def _dedupe_tokens(tokens: list[str]) -> list[str]:
    return list(dict.fromkeys(token for token in tokens if token))


def tokenize_char_bigrams(text: str) -> list[str]:
    """Boundary-safe lexical tokens.

    Chinese n-grams are emitted inside each contiguous CJK span, so punctuation,
    newlines, and metadata field boundaries can no longer create false bigrams.
    ASCII identifiers retain their exact punctuated form as well as useful parts.
    """
    lowered = str(text or "").lower()
    tokens: list[str] = []
    for exact in ASCII_EXACT_TOKEN_RE.findall(lowered):
        tokens.append(exact)
        compact = re.sub(r"[_.\-/]+", "", exact)
        if compact != exact:
            tokens.append(compact)
            tokens.extend(part for part in re.split(r"[_.\-/]+", exact) if part)
    for span in CJK_SPAN_RE.findall(lowered):
        if len(span) == 1:
            tokens.append(span)
            continue
        tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
    return tokens


def tokenize_domain_words(text: str, *, cut_for_search: Callable[[str], Any] | None = None) -> list[str]:
    lowered = str(text or "").lower()
    tokens = list(ASCII_EXACT_TOKEN_RE.findall(lowered))
    for span in CJK_SPAN_RE.findall(lowered):
        if cut_for_search is None:
            tokens.append(span)
        else:
            tokens.extend(str(token).strip().lower() for token in cut_for_search(span))
    return [token for token in tokens if token and not token.isspace()]


def tokenize_exact_terms(text: str) -> list[str]:
    lowered = str(text or "").lower()
    tokens = list(ASCII_EXACT_TOKEN_RE.findall(lowered))
    tokens.extend(span for span in CJK_SPAN_RE.findall(lowered) if len(span) >= 2)
    return _dedupe_tokens(tokens)


def tokenize_for_bm25(text: str) -> list[str]:
    """Compatibility alias for the boundary-safe character lane."""
    return tokenize_char_bigrams(text)


def _natural_alias(value: str) -> bool:
    normalized = str(value or "").strip()
    if not normalized or not NATURAL_ALIAS_RE.fullmatch(normalized):
        return False
    lowered = normalized.lower()
    return not any(
        marker in lowered
        for marker in ("avatar_", "char_", "npc_", "trap_", "token_", "skchr_", "level_")
    )


def build_domain_terms(documents: list[dict], alias_map: dict[str, list[str]]) -> list[str]:
    terms: list[str] = []
    metadata_fields = (
        "activity_name",
        "story_name",
        "story_code",
        "stage_code",
        "stage_name",
        "zone_name",
        "chapter_name",
    )
    for document in documents:
        for field in metadata_fields:
            value = str(document.get(field) or "").strip()
            if value:
                terms.append(value)
        for segment in document.get("segments") or []:
            if isinstance(segment, dict):
                speaker = str(segment.get("speaker") or "").strip()
                if speaker:
                    terms.append(speaker)
    for canonical, aliases in alias_map.items():
        if _natural_alias(canonical):
            terms.append(canonical)
        terms.extend(alias for alias in aliases if _natural_alias(alias))
    return sorted(set(term for term in terms if len(term) >= 2), key=lambda term: (-len(term), term))


def build_sparse_document_fields(document: dict, alias_lookup: dict[str, list[str]]) -> dict[str, str]:
    body = str(document.get("clean_text") or document.get("search_text") or "").strip()
    title_values = [
        str(document.get(field) or "").strip()
        for field in (
            "activity_name",
            "story_name",
            "stage_name",
            "zone_name",
            "chapter_name",
            "avg_tag",
        )
    ]
    speaker_values = [
        str(segment.get("speaker") or "").strip()
        for segment in document.get("segments") or []
        if isinstance(segment, dict)
    ]
    exact_values = [
        str(document.get(field) or "").strip()
        for field in (
            "activity_id",
            "story_code",
            "stage_id",
            "stage_code",
            "zone_id",
        )
    ]
    visible_text = "\n".join([body, *title_values, *speaker_values, *exact_values])
    alias_values: list[str] = []
    for alias in sorted(alias_lookup, key=len, reverse=True):
        if alias not in visible_text:
            continue
        if _natural_alias(alias):
            alias_values.append(alias)
        alias_values.extend(value for value in alias_lookup[alias] if _natural_alias(value))
    return {
        "body": body,
        "title": "\n".join(value for value in [*title_values, *speaker_values] if value),
        "alias": "\n".join(_dedupe_tokens(alias_values)),
        "exact": "\n".join(value for value in exact_values if value),
    }


@dataclass(slots=True)
class SparseLane:
    name: str
    bm25: BM25Okapi
    tokenizer: Callable[[str], list[str]]
    weight: float


def serialize_sparse_bundle(
    documents: list[dict],
    *,
    alias_lookup: dict[str, list[str]],
    cut_for_search: Callable[[str], Any],
) -> dict[str, Any]:
    lane_tokens: dict[str, list[list[str]]] = {
        "char": [],
        "word": [],
        "title": [],
        "alias": [],
        "exact": [],
    }
    for document in documents:
        fields = build_sparse_document_fields(document, alias_lookup)
        combined = "\n".join(fields.values())
        lane_tokens["char"].append(tokenize_char_bigrams(combined))
        lane_tokens["word"].append(tokenize_domain_words(combined, cut_for_search=cut_for_search))
        lane_tokens["title"].append(tokenize_domain_words(fields["title"], cut_for_search=cut_for_search))
        lane_tokens["alias"].append(tokenize_exact_terms(fields["alias"]))
        lane_tokens["exact"].append(tokenize_exact_terms(fields["exact"] + "\n" + fields["title"]))
    return {
        "version": 2,
        "lanes": lane_tokens,
        "lane_weights": {
            "char": 1.0,
            "word": 1.0,
            "title": 1.35,
            "alias": 1.6,
            "exact": 2.2,
        },
        "domain_terms": build_domain_terms(documents, alias_lookup),
    }


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_main_chapter_number(value: str) -> int | None:
    raw = value.strip()
    if not raw:
        return None
    if raw.isdigit():
        number = int(raw)
        return number if 0 < number < 100 else None
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if raw == "十":
        return 10
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        number = tens * 10 + ones
        return number if 0 < number < 100 else None
    if len(raw) == 1 and raw in digits:
        return digits[raw]
    return None


def extract_main_chapter_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in MAIN_CHAPTER_REF_RE.finditer(text or ""):
        raw_number = next((group for group in match.groups() if group), "")
        number = parse_main_chapter_number(raw_number)
        if number is not None:
            numbers.append(number)
    return list(dict.fromkeys(numbers))


class ArknightsHybridRetriever:
    def __init__(
        self,
        documents: list[dict],
        index: faiss.Index,
        bm25: BM25Okapi,
        embedding_model: SentenceTransformer,
        reranker: CrossEncoderReranker | None = None,
        minirag_index: MiniRAGIndex | None = None,
        sparse_lanes: list[SparseLane] | None = None,
        domain_terms: list[str] | None = None,
        dense_query_prompt: str = "",
        dense_truncate_dim: int | None = None,
        dense_query_max_length: int | None = None,
    ) -> None:
        self.documents = documents
        self.index = index
        self.bm25 = bm25
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.minirag_index = minirag_index
        self.sparse_lanes = sparse_lanes or [
            SparseLane("legacy", bm25, tokenize_for_bm25, 1.0)
        ]
        self.domain_terms = domain_terms or []
        self.dense_query_prompt = dense_query_prompt
        self.dense_truncate_dim = dense_truncate_dim
        self.dense_query_max_length = dense_query_max_length
        self._sparse_query_cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        self.chapter_doc_indices: dict[str, list[int]] = {}
        self.story_doc_indices: dict[str, list[int]] = {}
        self.storyline_doc_indices: dict[str, list[int]] = {}
        self.stage_doc_indices: dict[tuple[str, str], list[int]] = {}
        self.activity_story_sort_doc_indices: dict[str, dict[int, list[int]]] = {}
        self._dense_scope_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for doc_index, document in enumerate(documents):
            chapter_scope = document_chapter_scope_key(document)
            if chapter_scope:
                self.chapter_doc_indices.setdefault(chapter_scope, []).append(doc_index)
            story_id = str(document.get("story_id") or "").strip()
            if story_id:
                self.story_doc_indices.setdefault(story_id, []).append(doc_index)
            for storyline_scope in document_storyline_scopes(document):
                self.storyline_doc_indices.setdefault(storyline_scope, []).append(doc_index)
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
        sparse_index_path: Path | None = None,
        index_metadata_path: Path | None = None,
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
        resolved_sparse_path = sparse_index_path or bm25_tokens_path.with_name(SPARSE_INDEX_PATH.name)
        sparse_payload: Any = None
        if resolved_sparse_path.exists():
            print(f"[retriever-load] sparse {resolved_sparse_path}", file=sys.stderr, flush=True)
            with resolved_sparse_path.open("rb") as handle:
                sparse_payload = pickle.load(handle)
        print(f"[retriever-load] bm25 {bm25_tokens_path}", file=sys.stderr, flush=True)
        if sparse_payload is None:
            with bm25_tokens_path.open("rb") as handle:
                tokenized_corpus = pickle.load(handle)
            sparse_payload = {
                "version": 1,
                "lanes": {"legacy": tokenized_corpus},
                "lane_weights": {"legacy": 1.0},
                "domain_terms": [],
            }
        lanes_payload = sparse_payload.get("lanes") if isinstance(sparse_payload, dict) else None
        if not isinstance(lanes_payload, dict) or not lanes_payload:
            raise ValueError(f"Invalid sparse index payload: {resolved_sparse_path}")
        lane_weights = sparse_payload.get("lane_weights") or {}
        domain_terms = [str(term) for term in sparse_payload.get("domain_terms") or []]
        word_tokenizer = cls._make_domain_tokenizer(domain_terms)
        exact_tokenizer = cls._make_exact_tokenizer(domain_terms)
        lane_tokenizers: dict[str, Callable[[str], list[str]]] = {
            "legacy": tokenize_for_bm25,
            "char": tokenize_char_bigrams,
            "word": word_tokenizer,
            "title": word_tokenizer,
            "alias": exact_tokenizer,
            "exact": exact_tokenizer,
        }
        sparse_lanes = [
            SparseLane(
                str(name),
                BM25Okapi(tokenized_corpus),
                lane_tokenizers.get(str(name), tokenize_for_bm25),
                float(lane_weights.get(name, 1.0)),
            )
            for name, tokenized_corpus in lanes_payload.items()
        ]
        bm25 = sparse_lanes[0].bm25
        print(f"[retriever-load] bm25 loaded elapsed={time.time() - started:.1f}s", file=sys.stderr, flush=True)
        resolved_metadata_path = index_metadata_path or documents_path.with_name(CORPUS_METADATA_PATH.name)
        metadata: dict[str, Any] = {}
        if resolved_metadata_path.exists():
            payload = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                metadata = payload
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
            sparse_lanes=sparse_lanes,
            domain_terms=domain_terms,
            dense_query_prompt=str(metadata.get("dense_query_prompt") or ""),
            dense_truncate_dim=(
                int(metadata["embedding_truncate_dim"])
                if metadata.get("embedding_truncate_dim") is not None
                else None
            ),
            dense_query_max_length=(
                int(metadata["dense_query_max_length"])
                if metadata.get("dense_query_max_length") is not None
                else None
            ),
        )

    @staticmethod
    def _make_domain_tokenizer(domain_terms: list[str]) -> Callable[[str], list[str]]:
        try:
            import jieba
        except ImportError:
            return lambda text: tokenize_domain_words(text)
        tokenizer = jieba.Tokenizer()
        for term in domain_terms:
            tokenizer.add_word(term, freq=10_000_000)
        return lambda text: tokenize_domain_words(text, cut_for_search=tokenizer.cut_for_search)

    @staticmethod
    def _make_exact_tokenizer(domain_terms: list[str]) -> Callable[[str], list[str]]:
        natural_terms = {
            term.lower() for term in domain_terms if _natural_alias(term)
        }

        def tokenize(text: str) -> list[str]:
            lowered = str(text or "").lower()
            tokens = list(ASCII_EXACT_TOKEN_RE.findall(lowered))
            tokens.extend(
                span
                for span in CJK_SPAN_RE.findall(lowered)
                if len(span) >= 2 and span in natural_terms
            )
            return _dedupe_tokens(tokens)

        return tokenize

    def _encode_dense_queries(self, queries: list[str]) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.dense_query_prompt:
            kwargs["prompt"] = self.dense_query_prompt
        if self.dense_truncate_dim is not None:
            kwargs["truncate_dim"] = self.dense_truncate_dim
        previous_max_length = getattr(self.embedding_model, "max_seq_length", None)
        if self.dense_query_max_length is not None:
            self.embedding_model.max_seq_length = min(
                self.dense_query_max_length,
                max(128, max((len(query) for query in queries), default=0) * 2 + 64),
            )
        try:
            try:
                embeddings = self.embedding_model.encode(queries, **kwargs)
            except TypeError as exc:
                if "truncate_dim" not in str(exc):
                    raise
                kwargs.pop("truncate_dim", None)
                embeddings = self.embedding_model.encode(queries, **kwargs)
                if self.dense_truncate_dim is not None:
                    embeddings = embeddings[:, : self.dense_truncate_dim]
                    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
        finally:
            if previous_max_length is not None:
                self.embedding_model.max_seq_length = previous_max_length
        return np.asarray(embeddings, dtype=np.float32)

    def dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        vector = self._encode_dense_queries([query])
        return self._dense_hits_from_vectors(vector, top_k=top_k)[0]

    def dense_search_many(self, queries: list[str], top_k: int) -> list[list[dict[str, Any]]]:
        if not queries:
            return []
        vectors = self._encode_dense_queries(queries)
        return self._dense_hits_from_vectors(vectors, top_k=top_k)

    def _dense_hits_from_vectors(
        self,
        vectors: np.ndarray,
        *,
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        scores, indices = self.index.search(vectors, top_k)
        batches: list[list[dict[str, Any]]] = []
        for row_scores, row_indices in zip(scores, indices):
            hits: list[dict[str, Any]] = []
            for score, doc_index in zip(row_scores.tolist(), row_indices.tolist()):
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
            batches.append(hits)
        return batches

    def dense_search_chapter(
        self,
        query: str,
        top_k: int,
        *,
        chapter_scope: str,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not chapter_scope:
            return []
        doc_indices = self.chapter_doc_indices.get(chapter_scope, [])
        if not doc_indices:
            return []
        scoped_indices, scoped_vectors = self._dense_scope_vectors(chapter_scope, doc_indices)
        if scoped_vectors.size == 0:
            return []
        vector = self._encode_dense_queries([query])[0]
        scores = scoped_vectors @ vector
        take = min(top_k, len(scores))
        if take <= 0:
            return []
        if take >= len(scores):
            order = np.argsort(scores)[::-1]
        else:
            order = np.argpartition(scores, -take)[-take:]
            order = order[np.argsort(scores[order])[::-1]]
        hits: list[dict[str, Any]] = []
        for offset in order.tolist():
            doc_index = int(scoped_indices[offset])
            hits.append(
                {
                    "doc_index": doc_index,
                    "score": float(scores[offset]),
                    "document": self.documents[doc_index],
                    "scoped_source": "chapter_dense",
                }
            )
        return hits

    def _dense_scope_vectors(
        self,
        chapter_scope: str,
        doc_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self._dense_scope_cache.get(chapter_scope)
        if cached is not None:
            return cached
        valid_indices: list[int] = []
        vectors: list[np.ndarray] = []
        for doc_index in doc_indices:
            try:
                vector = self.index.reconstruct(int(doc_index))
            except Exception:
                continue
            valid_indices.append(int(doc_index))
            vectors.append(np.asarray(vector, dtype=np.float32))
        if vectors:
            payload = (np.asarray(valid_indices, dtype=np.int64), np.vstack(vectors).astype(np.float32))
        else:
            payload = (np.asarray([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32))
        if len(self._dense_scope_cache) >= 32:
            self._dense_scope_cache.pop(next(iter(self._dense_scope_cache)))
        self._dense_scope_cache[chapter_scope] = payload
        return payload

    def sparse_search(
        self,
        query: str,
        top_k: int,
        *,
        storyline_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        scores, lane_scores = self._sparse_scores(query)
        allowed_indices: list[int] | None = None
        if storyline_scope:
            allowed_indices = self.storyline_doc_indices.get(storyline_scope, [])
            if not allowed_indices:
                return []

        if allowed_indices is not None:
            candidate_indices = np.array(allowed_indices, dtype=np.int64)
            candidate_scores = scores[candidate_indices]
            positive_mask = candidate_scores > 0
            candidate_indices = candidate_indices[positive_mask]
            candidate_scores = candidate_scores[positive_mask]
            if len(candidate_indices) <= 0:
                return []
            if top_k >= len(candidate_indices):
                order = np.argsort(candidate_scores)[::-1]
            else:
                order = np.argpartition(candidate_scores, -top_k)[-top_k:]
                order = order[np.argsort(candidate_scores[order])[::-1]]
            top_indices = candidate_indices[order]
        elif top_k >= len(scores):
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
                    "sparse_lane_scores": {
                        name: float(values[doc_index])
                        for name, values in lane_scores.items()
                        if float(values[doc_index]) > 0
                    },
                }
            )
        return hits

    def _sparse_scores(self, query: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cached = self._sparse_query_cache.get(query)
        if cached is not None:
            return cached
        combined = np.zeros(len(self.documents), dtype=np.float32)
        per_lane: dict[str, np.ndarray] = {}
        for lane in self.sparse_lanes:
            tokens = lane.tokenizer(query)
            raw_scores = np.asarray(lane.bm25.get_scores(tokens), dtype=np.float32)
            positive = raw_scores > 0
            normalized = np.zeros_like(raw_scores)
            if np.any(positive):
                positive_scores = raw_scores[positive]
                scale = float(np.percentile(positive_scores, 99.5))
                if scale <= 0:
                    scale = float(positive_scores.max())
                normalized[positive] = np.minimum(positive_scores / max(scale, 1e-6), 1.5)
            weighted = normalized * lane.weight
            combined += weighted
            per_lane[lane.name] = raw_scores
        payload = (combined, per_lane)
        if len(self._sparse_query_cache) >= 64:
            self._sparse_query_cache.pop(next(iter(self._sparse_query_cache)))
        self._sparse_query_cache[query] = payload
        return payload

    def sparse_search_chapter(
        self,
        query: str,
        top_k: int,
        *,
        chapter_scope: str,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not chapter_scope:
            return []
        allowed_indices = self.chapter_doc_indices.get(chapter_scope, [])
        if not allowed_indices:
            return []
        scores, lane_scores = self._sparse_scores(query)
        candidate_indices = np.array(allowed_indices, dtype=np.int64)
        candidate_scores = scores[candidate_indices]
        positive_mask = candidate_scores > 0
        candidate_indices = candidate_indices[positive_mask]
        candidate_scores = candidate_scores[positive_mask]
        if len(candidate_indices) <= 0:
            return []
        if top_k >= len(candidate_indices):
            order = np.argsort(candidate_scores)[::-1]
        else:
            order = np.argpartition(candidate_scores, -top_k)[-top_k:]
            order = order[np.argsort(candidate_scores[order])[::-1]]
        hits: list[dict[str, Any]] = []
        for offset in order.tolist():
            doc_index = int(candidate_indices[offset])
            hits.append(
                {
                    "doc_index": doc_index,
                    "score": float(candidate_scores[offset]),
                    "document": self.documents[doc_index],
                    "scoped_source": "chapter_sparse",
                    "sparse_lane_scores": {
                        name: float(values[doc_index])
                        for name, values in lane_scores.items()
                        if float(values[doc_index]) > 0
                    },
                }
            )
        return hits

    def minirag_search(
        self,
        query: str,
        top_k: int,
        *,
        chapter_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.minirag_index is None:
            return []
        return self.minirag_index.search(
            query,
            self.documents,
            top_k=top_k,
            chapter_scope=chapter_scope,
        )

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
        dense_min_quota: int = 0,
        sparse_min_quota: int = 0,
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

        ranked = sorted(
            fused.values(),
            key=lambda item: item["fusion_score"],
            reverse=True,
        )
        return self.apply_source_quotas(
            ranked,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            top_k=top_k,
            dense_min_quota=dense_min_quota,
            sparse_min_quota=sparse_min_quota,
        )

    @staticmethod
    def apply_source_quotas(
        ranked: list[dict[str, Any]],
        *,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
        top_k: int,
        dense_min_quota: int,
        sparse_min_quota: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        dense_quota = min(max(0, dense_min_quota), top_k)
        sparse_quota = min(max(0, sparse_min_quota), top_k)
        if dense_quota <= 0 and sparse_quota <= 0:
            return ranked[:top_k]
        by_doc_index = {int(item["doc_index"]): item for item in ranked}
        selected_ids: set[int] = set()

        def reserve_from(hits: list[dict[str, Any]], quota: int, source: str) -> None:
            for hit in hits[:quota]:
                doc_index = int(hit["doc_index"])
                if doc_index not in by_doc_index:
                    continue
                item = by_doc_index[doc_index]
                quota_sources = item.setdefault("quota_sources", [])
                if source not in quota_sources:
                    quota_sources.append(source)
                selected_ids.add(doc_index)

        reserve_from(sparse_hits, sparse_quota, "sparse")
        reserve_from(dense_hits, dense_quota, "dense")
        for item in ranked:
            if len(selected_ids) >= top_k:
                break
            doc_index = int(item["doc_index"])
            if doc_index in selected_ids:
                continue
            selected_ids.add(doc_index)
        return [
            item for item in ranked if int(item["doc_index"]) in selected_ids
        ][:top_k]

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
        same_story_sweep: bool = False,
        same_story_max_seed_docs: int = 8,
        same_story_max_docs_per_story: int = 24,
        top_k: int = 120,
    ) -> list[dict[str, Any]]:
        if not hits:
            return hits
        neighbor_doc_indices = self._collect_story_and_stage_neighbors(
            hits,
            max_seed_docs=max_seed_docs,
            story_window=story_window,
            activity_story_sort_window=activity_story_sort_window,
            same_story_sweep=same_story_sweep,
            same_story_max_seed_docs=same_story_max_seed_docs,
            same_story_max_docs_per_story=same_story_max_docs_per_story,
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

    def _document_source_type(self, document: dict) -> str:
        fields = " ".join(
            str(document.get(key) or "")
            for key in ("id", "source_path", "activity_name", "story_id", "activity_id", "avg_tag")
        )
        if any(marker in fields for marker in MOEGIRL_SOURCE_MARKERS):
            return "moegirl_background"
        if any(marker in fields for marker in PROFILE_SOURCE_MARKERS):
            return "profile"
        if "charword/" in fields:
            return "voice"
        if any(marker in fields for marker in STORY_SOURCE_MARKERS):
            return "story_text"
        return "other"

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
        same_story_sweep: bool = False,
        same_story_max_seed_docs: int = 8,
        same_story_max_docs_per_story: int = 24,
    ) -> list[int]:
        candidate_indices: list[int] = []
        for seed_rank, hit in enumerate(seed_hits[:max_seed_docs]):
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
                    if same_story_sweep and seed_rank < same_story_max_seed_docs:
                        for neighbor in self._story_sweep_indices(
                            story_indices,
                            current_pos,
                            max_docs=max(1, same_story_max_docs_per_story),
                        ):
                            if neighbor not in candidate_indices:
                                candidate_indices.append(neighbor)
                elif same_story_sweep and seed_rank < same_story_max_seed_docs:
                    for neighbor in story_indices[: max(1, same_story_max_docs_per_story)]:
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

    @staticmethod
    def _story_sweep_indices(story_indices: list[int], center_pos: int, *, max_docs: int) -> list[int]:
        if center_pos < 0 or not story_indices or max_docs <= 0:
            return []
        ordered: list[int] = []
        for distance in range(len(story_indices)):
            positions = [center_pos] if distance == 0 else [center_pos - distance, center_pos + distance]
            for pos in positions:
                if 0 <= pos < len(story_indices):
                    doc_index = story_indices[pos]
                    if doc_index not in ordered:
                        ordered.append(doc_index)
                        if len(ordered) >= max_docs:
                            return ordered
        return ordered

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

    def _document_text(self, document: dict) -> str:
        return str(document.get("search_text") or document.get("clean_text") or "")

    def _document_main_chapter_number(self, document: dict) -> int | None:
        fields = " ".join(
            str(document.get(key) or "")
            for key in ("activity_id", "story_id", "story_key", "source_path", "id", "search_text")
        )
        match = MAIN_CHAPTER_SOURCE_RE.search(fields)
        if match:
            return int(match.group(1))
        numbers = extract_main_chapter_numbers(fields)
        return numbers[0] if numbers else None

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
        source_type = self._document_source_type(document)

        if source_type == "moegirl_background" and self._is_story_detail_query(query):
            if query_overlap or bridge_overlap:
                return {"context"}
            return set()

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
        left_source_type = self._document_source_type(left)
        right_source_type = self._document_source_type(right)
        generic_background_pair = (
            left_source_type == "moegirl_background"
            and right_source_type == "moegirl_background"
            and left_story != right_story
        )
        if left_activity and left_activity == right_activity:
            if not generic_background_pair:
                score += 0.2
            left_stage = str(left.get("stage_code") or "")
            right_stage = str(right.get("stage_code") or "")
            if left_stage and left_stage == right_stage and not generic_background_pair:
                score += 0.25

            left_sort = left.get("story_sort")
            right_sort = right.get("story_sort")
            if isinstance(left_sort, int) and isinstance(right_sort, int) and not generic_background_pair:
                distance = abs(left_sort - right_sort)
                if distance <= 1:
                    score += 0.25
                elif distance <= 3:
                    score += 0.12

            left_stage_number = self._document_stage_number(left)
            right_stage_number = self._document_stage_number(right)
            if left_stage_number is not None and right_stage_number is not None and not generic_background_pair:
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
            chain_structure = {
                "chain_length": len(member_indices),
                "causal_order": "model_candidate",
                "evidence_types": sorted(chain["roles"]) if chain.get("roles") else ["context"],
            }
            chain_text = (
                f"[CHAIN_LEN={chain_structure['chain_length']}] "
                f"[CAUSAL_ORDER={chain_structure['causal_order']}] "
                f"[EVIDENCE_TYPES=({'|'.join(chain_structure['evidence_types'])})]\n"
                + chain_text
            )
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
            chain_score = float(score)
            for member in chain["members"]:
                item = doc_index_to_item.get(int(member["item"]["doc_index"]))
                if item is None:
                    continue
                document = item["document"]
                role_bonus = 0.07 * len(member["roles"])
                if "action" in member["roles"]:
                    role_bonus += 0.12
                if "outcome" in member["roles"]:
                    role_bonus += 0.11
                if "motive" in member["roles"]:
                    role_bonus += 0.1
                member_relevance = self._compute_query_overlap_bonus(
                    query,
                    document,
                    bridge_terms,
                    query_mode=query_mode,
                )
                adjusted_chain_score = self._adjust_chain_score_for_query_type(
                    query,
                    document,
                    chain_score,
                    query_mode,
                )
                if query_mode in LOW_RERANK_QUERY_TYPES:
                    chain_member_score = (
                        adjusted_chain_score * 0.35
                        + normalized_rank_bonus * 0.25
                        + role_bonus * 0.25
                        + member_relevance
                    )
                else:
                    chain_member_score = (
                        adjusted_chain_score * 0.45
                        + normalized_rank_bonus * 0.45
                        + role_bonus
                        + member_relevance
                    )
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

    def _final_rerank_score(self, item: dict[str, Any]) -> float:
        chain_score = item.get("evidence_chain_score")
        rerank_score = float(item.get("rerank_score") or 0.0)
        if chain_score is not None:
            chain_bonus = max(min(float(chain_score), 3.5), -1.5)
            return rerank_score + chain_bonus
        return rerank_score

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
        if query_mode is None:
            query_mode = self._infer_query_mode(query)
        candidates = [dict(item) for item in hits]
        if self.reranker:
            scores = self.reranker.score(
                query=query,
                documents=[item["document"]["search_text"] for item in candidates],
                batch_size=batch_size,
            )
            for item, score in zip(candidates, scores, strict=True):
                item["rerank_score"] = float(score)
            resolved_bridge_terms = bridge_terms or self.extract_bridge_terms(query, candidates)
            chain_scored = self._apply_chain_model_scores(
                query,
                candidates,
                bridge_terms=resolved_bridge_terms,
                batch_size=batch_size,
                query_mode=query_mode,
            )
            if not chain_scored:
                self._apply_evidence_chain_rerank(query, candidates, bridge_terms=resolved_bridge_terms)
            self._apply_main_chapter_focus_adjustment(query, candidates)
            self._apply_source_type_focus_adjustment(query, candidates, query_mode=query_mode)
            return sorted(
                candidates,
                key=self._final_rerank_score,
                reverse=True,
            )[:top_k]

        for item in candidates:
            item["rerank_score"] = float(item.get("fusion_score") or 0.0)
        resolved_bridge_terms = bridge_terms or self.extract_bridge_terms(query, candidates)
        self._apply_evidence_chain_rerank(query, candidates, bridge_terms=resolved_bridge_terms)
        self._apply_main_chapter_focus_adjustment(query, candidates)
        self._apply_source_type_focus_adjustment(query, candidates, query_mode=query_mode)
        return sorted(
            candidates,
            key=self._final_rerank_score,
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
                dense_min_quota=query_config.dense_min_quota,
                sparse_min_quota=query_config.sparse_min_quota,
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
                top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
                rrf_k=query_config.rrf_k,
                dense_weight=query_config.dense_weight,
                sparse_weight=query_config.sparse_weight,
                minirag_weight=minirag_weight,
                dense_min_quota=query_config.dense_min_quota,
                sparse_min_quota=query_config.sparse_min_quota,
            )
        if query_config.enable_neighbor_expansion:
            fused_hits = self.expand_hits_with_neighbors(
                fused_hits,
                max_seed_docs=query_config.neighbor_max_seed_docs,
                story_window=query_config.neighbor_story_window,
                activity_story_sort_window=query_config.neighbor_activity_story_sort_window,
                same_story_sweep=query_config.enable_same_story_sweep,
                same_story_max_seed_docs=query_config.same_story_sweep_max_seed_docs,
                same_story_max_docs_per_story=query_config.same_story_sweep_max_docs_per_story,
                top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k)
                + (
                    query_config.same_story_sweep_extra_candidates
                    if query_config.enable_same_story_sweep
                    else 0
                ),
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
