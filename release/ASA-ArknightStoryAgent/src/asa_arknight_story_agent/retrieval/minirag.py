from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from asa_arknight_story_agent.config import DOCUMENTS_PATH, MINIRAG_GRAPH_PATH, OPERATOR_ALIAS_MAP_PATH


ENTITY_RUN_RE = re.compile(r"[\u4e00-\u9fff·]{2,32}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
SPEAKER_PREFIX_RE = re.compile(r"(?m)^([\u4e00-\u9fff·A-Za-z0-9_.\-]{2,16})[：:]")
TITLE_RE = re.compile(r"[《「“]([^》」”]{2,24})[》」”]")
GENERIC_ENTITY_STOP_WORDS = frozenset(
    {
        "什么",
        "为什么",
        "怎么",
        "如何",
        "这里",
        "那里",
        "这个",
        "那个",
        "这些",
        "那些",
        "自己",
        "我们",
        "你们",
        "他们",
        "她们",
        "它们",
        "博士",
        "干员",
        "作战",
        "行动",
        "当前",
        "证据",
        "剧情",
        "事情",
        "一个",
        "一些",
        "不是",
        "没有",
        "已经",
        "因为",
        "所以",
        "但是",
        "然后",
        "如果",
        "只是",
        "可以",
        "知道",
        "觉得",
        "还是",
        "不会",
        "不能",
        "必须",
        "具体",
        "一事",
        "指什么",
        "是什么",
        "为什么",
        "怎么样",
        "怎么办",
        "哪里",
        "哪位",
    }
)
METADATA_ENTITY_PREFIXES = ("activity:", "story:", "stage:")
RELATION_GATE_STOP_WORDS = frozenset(
    {
        "什么",
        "为什么",
        "怎么",
        "如何",
        "是谁",
        "哪里",
        "哪位",
        "这个",
        "那个",
        "这些",
        "那些",
        "事情",
        "关系",
        "原因",
        "目的",
        "动机",
        "真相",
        "秘密",
    }
)
RELATION_GATE_KEYWORDS = frozenset(
    {
        "父亲",
        "母亲",
        "儿子",
        "女儿",
        "老师",
        "学生",
        "同伴",
        "朋友",
        "敌人",
        "上司",
        "下属",
        "领袖",
        "成员",
        "属于",
        "来自",
        "控制",
        "背叛",
        "保护",
        "杀死",
        "刺杀",
        "攻击",
        "阻止",
        "帮助",
        "合作",
        "交易",
        "约定",
        "计划",
        "导致",
        "引发",
        "揭露",
        "识破",
        "隐瞒",
        "伪装",
        "身份",
        "真身",
    }
)

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_alias_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): [str(item) for item in value if str(item).strip()]
        for key, value in payload.items()
        if isinstance(value, list)
    }


def build_alias_lookup(alias_map: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        names = [canonical, *aliases]
        for name in names:
            normalized = name.strip()
            if len(normalized) >= 2:
                lookup[normalized] = canonical
    return lookup


@lru_cache(maxsize=16)
def _compile_alias_regex(alias_items: tuple[tuple[str, str], ...]) -> re.Pattern[str] | None:
    aliases = [alias for alias, _canonical in alias_items]
    pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
    return re.compile(pattern) if pattern else None


@lru_cache(maxsize=16)
def _compile_alias_automaton(alias_items: tuple[tuple[str, str], ...]) -> Any:
    try:
        import ahocorasick  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    automaton = ahocorasick.Automaton()
    for alias, canonical in alias_items:
        automaton.add_word(alias, (alias, canonical))
    automaton.make_automaton()
    return automaton


def extract_alias_entities(text: str, alias_lookup: dict[str, str]) -> list[str]:
    """Exact alias matching using cached Aho-Corasick or regex fallback."""
    if not alias_lookup:
        return []
    alias_items = tuple(sorted(alias_lookup.items()))
    automaton = _compile_alias_automaton(alias_items)
    if automaton is None:
        regex = _compile_alias_regex(alias_items)
        if regex is None:
            return []
        found: list[str] = []
        seen: set[str] = set()
        for match in regex.finditer(text):
            canonical = alias_lookup.get(match.group(0))
            if canonical and canonical not in seen:
                seen.add(canonical)
                found.append(canonical)
        return found

    found: list[str] = []
    seen: set[str] = set()
    for _end, (_alias, canonical) in automaton.iter(text):
        if canonical not in seen:
            seen.add(canonical)
            found.append(canonical)
    return found


def metadata_entities(document: dict[str, Any]) -> list[str]:
    entities: list[str] = []
    for key, prefix in (
        ("activity_name", "activity"),
        ("story_name", "story"),
        ("stage_code", "stage"),
        ("stage_name", "stage"),
    ):
        value = str(document.get(key) or "").strip()
        if value:
            entities.append(f"{prefix}:{value}")
    return entities


def _relation_gate_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in ENTITY_RUN_RE.finditer(text):
        token = match.group(0).strip()
        if len(token) < 2 or token in RELATION_GATE_STOP_WORDS:
            continue
        if any(stop in token for stop in RELATION_GATE_STOP_WORDS):
            continue
        terms.add(token)
        if "\u4e00" <= token[0] <= "\u9fff" and len(token) >= 4:
            for size in (2, 3, 4):
                for index in range(0, len(token) - size + 1):
                    gram = token[index : index + size]
                    if gram not in RELATION_GATE_STOP_WORDS:
                        terms.add(gram)
    for keyword in RELATION_GATE_KEYWORDS:
        if keyword in text:
            terms.add(keyword)
    return terms


def _is_generic_entity_candidate(token: str) -> bool:
    token = token.strip()
    if len(token) < 2 or len(token) > 16:
        return False
    if token in GENERIC_ENTITY_STOP_WORDS:
        return False
    if any(marker in token for marker in ("什么", "怎么", "为何", "为什么", "如何")):
        return False
    if token.endswith(("的", "了", "吗", "呢", "吧", "啊", "着", "过")):
        return False
    return True


def extract_generic_text_entities(text: str, *, limit: int = 48) -> list[str]:
    """Extract lightweight text entities without an LLM.

    This is intentionally broad: MiniRAG only needs graph anchors, and final
    answerability is still handled by hybrid retrieval/reranking.
    """
    candidates: dict[str, tuple[float, int]] = {}

    def add(raw: str, *, boost: float = 0.0, pos: int = 10**9) -> None:
        token = raw.strip()
        if not _is_generic_entity_candidate(token):
            return
        score = float(len(token)) + boost
        old = candidates.get(token)
        if old is None:
            candidates[token] = (score, pos)
        else:
            candidates[token] = (old[0] + score, min(old[1], pos))

    for pattern in (SPEAKER_PREFIX_RE, TITLE_RE):
        for match in pattern.finditer(text):
            value = match.group(1) if match.lastindex else match.group(0)
            add(value, boost=6.0, pos=match.start())

    ranked = sorted(
        candidates.items(),
        key=lambda item: (-item[1][0], item[1][1], -len(item[0]), item[0]),
    )
    return [token for token, _ in ranked[:limit]]


def _collect_teacher_relations(
    teacher_annotations: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, str]], dict[str, set[str]], dict[str, list[str]]]:
    """Aggregate teacher-extracted (head, relation, tail, evidence_id) triples
    and build a (head|tail) -> set(other_entity) co-occurrence map."""
    if not teacher_annotations:
        return [], {}, {}
    relations: list[dict[str, str]] = []
    cooccurrence: dict[str, set[str]] = {}
    evidence_to_entities: dict[str, list[str]] = {}
    for annotation in teacher_annotations:
        if not isinstance(annotation, dict):
            continue
        batch_id = str(annotation.get("batch_id") or "").strip()
        if batch_id:
            for relation_item in annotation.get("relations") or []:
                if not isinstance(relation_item, dict):
                    continue
                head = str(relation_item.get("head") or "").strip()
                relation = str(relation_item.get("relation") or "").strip()
                tail = str(relation_item.get("tail") or "").strip()
                if not head or not relation or not tail:
                    continue
                evidence_doc_ids = [
                    str(doc_id).strip()
                    for doc_id in relation_item.get("evidence_doc_ids") or []
                    if str(doc_id).strip()
                ]
                if not evidence_doc_ids:
                    evidence_doc_id = str(relation_item.get("doc_id") or "").strip()
                    if evidence_doc_id:
                        evidence_doc_ids = [evidence_doc_id]
                relations.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "evidence_id": "|".join(evidence_doc_ids),
                        "source_name": batch_id,
                    }
                )
                cooccurrence.setdefault(head, set()).add(tail)
                cooccurrence.setdefault(tail, set()).add(head)
                for doc_id in evidence_doc_ids:
                    evidence_to_entities.setdefault(f"doc_id::{doc_id}", []).extend([head, tail])
            continue
        doc_id = str(annotation.get("doc_id") or "").strip()
        if doc_id:
            entity_names: list[str] = []
            for entity in annotation.get("entities") or []:
                if isinstance(entity, dict):
                    name = str(entity.get("name") or "").strip()
                    if name:
                        entity_names.append(name)
                        for alias in entity.get("aliases") or []:
                            alias_text = str(alias or "").strip()
                            if alias_text:
                                entity_names.append(alias_text)
            for relation_item in annotation.get("relations") or []:
                if not isinstance(relation_item, dict):
                    continue
                head = str(relation_item.get("head") or "").strip()
                relation = str(relation_item.get("relation") or "").strip()
                tail = str(relation_item.get("tail") or "").strip()
                if not head or not relation or not tail:
                    continue
                relations.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "evidence_id": doc_id,
                        "source_name": doc_id,
                    }
                )
                cooccurrence.setdefault(head, set()).add(tail)
                cooccurrence.setdefault(tail, set()).add(head)
                entity_names.extend([head, tail])
            if entity_names:
                evidence_to_entities.setdefault(f"doc_id::{doc_id}", []).extend(entity_names)
            continue
        for triple in annotation.get("entity_relations") or []:
            if not isinstance(triple, dict):
                continue
            head = str(triple.get("head") or "").strip()
            relation = str(triple.get("relation") or "").strip()
            tail = str(triple.get("tail") or "").strip()
            if not head or not relation or not tail:
                continue
            relations.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "evidence_id": str(triple.get("evidence_id") or ""),
                    "source_name": str(annotation.get("source_name") or ""),
                }
            )
            cooccurrence.setdefault(head, set()).add(tail)
            cooccurrence.setdefault(tail, set()).add(head)
            source_name = str(annotation.get("source_name") or "")
            evidence_id = str(triple.get("evidence_id") or "")
            if source_name and evidence_id:
                evidence_to_entities.setdefault(f"{source_name}::{evidence_id}", []).extend([head, tail])
    return relations, cooccurrence, evidence_to_entities


def _source_name_candidates(document: dict[str, Any]) -> list[str]:
    source_path = Path(str(document.get("source_path") or ""))
    names = []
    if source_path.name:
        names.append(source_path.name)
    story_id = str(document.get("story_id") or "")
    if story_id:
        names.append(Path(story_id).name + ".json")
    return names


def build_minirag_graph(
    documents: list[dict[str, Any]],
    alias_map: dict[str, list[str]],
    teacher_annotations: list[dict[str, Any]] | None = None,
    *,
    progress: bool = False,
    progress_interval: int = 1000,
) -> dict[str, Any]:
    started = time.time()
    alias_lookup = build_alias_lookup(alias_map)
    entity_to_doc_indices: dict[str, list[int]] = {}
    entity_to_doc_weights: dict[str, dict[str, float]] = {}
    doc_to_entities: list[list[str]] = []

    if progress:
        print(
            f"[minirag-build] collect teacher relations annotations={len(teacher_annotations or [])}",
            file=sys.stderr,
            flush=True,
        )
    teacher_relations, teacher_cooccurrence, teacher_evidence_entities = _collect_teacher_relations(teacher_annotations)
    if progress:
        print(
            f"[minirag-build] teacher_relations={len(teacher_relations)} "
            f"teacher_entities={len(teacher_cooccurrence)}",
            file=sys.stderr,
            flush=True,
        )

    # Seed extra alias lookup from teacher entities so we can match them in documents
    extra_lookup = dict(alias_lookup)
    for entity in teacher_cooccurrence:
        if entity not in extra_lookup and len(entity) >= 2:
            extra_lookup[entity] = entity

    total_documents = len(documents)
    progress_interval = max(1, progress_interval)
    for doc_index, document in enumerate(documents):
        text = "\n".join(
            str(document.get(key) or "")
            for key in ("search_text", "clean_text", "activity_name", "story_name", "stage_code", "stage_name")
        )
        entities = []
        entities.extend(extract_alias_entities(text, extra_lookup))
        entities.extend(extract_generic_text_entities(text))
        for source_name in _source_name_candidates(document):
            for evidence_id in (f"E{int(document.get('chunk_index', -1)) + 1}", str(document.get("chunk_index") or "")):
                entities.extend(teacher_evidence_entities.get(f"{source_name}::{evidence_id}", []))
        doc_id = str(document.get("id") or "")
        if doc_id:
            entities.extend(teacher_evidence_entities.get(f"doc_id::{doc_id}", []))
        entities.extend(metadata_entities(document))
        deduped = []
        seen = set()
        counts = Counter(entities)
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                deduped.append(entity)
                entity_to_doc_indices.setdefault(entity, []).append(doc_index)
                entity_to_doc_weights.setdefault(entity, {})[str(doc_index)] = float(counts[entity])
        doc_to_entities.append(deduped)
        if progress and (
            doc_index == 0
            or (doc_index + 1) % progress_interval == 0
            or doc_index + 1 == total_documents
        ):
            elapsed = time.time() - started
            done = doc_index + 1
            docs_per_second = done / max(elapsed, 1e-6)
            eta_seconds = (total_documents - done) / max(docs_per_second, 1e-6)
            print(
                f"[minirag-build] docs={done}/{total_documents} "
                f"entities={len(entity_to_doc_indices)} "
                f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    return {
        "version": 3 if teacher_relations else 1,
        "documents_path": str(DOCUMENTS_PATH),
        "document_count": len(documents),
        "entity_count": len(entity_to_doc_indices),
        "alias_map": alias_map,
        "entity_to_doc_indices": entity_to_doc_indices,
        "entity_to_doc_weights": entity_to_doc_weights,
        "doc_to_entities": doc_to_entities,
        "doc_id_to_index": {
            str(document.get("id")): doc_index
            for doc_index, document in enumerate(documents)
            if str(document.get("id") or "").strip()
        },
        "teacher_relations": teacher_relations,
        "entity_cooccurrence": {
            head: sorted(tails) for head, tails in teacher_cooccurrence.items()
        },
    }


@dataclass(slots=True)
class MiniRAGIndex:
    entity_to_doc_indices: dict[str, list[int]]
    entity_to_doc_weights: dict[str, dict[int, float]]
    doc_to_entities: list[list[str]]
    alias_lookup: dict[str, str]
    generic_entities: set[str]
    teacher_relations: list[dict[str, str]]
    relation_adjacency: dict[str, list[dict[str, str]]]
    relation_evidence_doc_indices: dict[tuple[str, str, str, str], list[int]]

    @classmethod
    def load(cls, path: Path = MINIRAG_GRAPH_PATH) -> "MiniRAGIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        alias_map = payload.get("alias_map") if isinstance(payload.get("alias_map"), dict) else {}
        teacher_relations = [
            item
            for item in payload.get("teacher_relations", [])
            if isinstance(item, dict)
        ]
        relation_adjacency: dict[str, list[dict[str, str]]] = {}
        doc_id_to_index = {
            str(doc_id): int(index)
            for doc_id, index in (payload.get("doc_id_to_index") or {}).items()
            if str(doc_id).strip()
        }
        relation_evidence_doc_indices: dict[tuple[str, str, str, str], list[int]] = {}
        for relation in teacher_relations:
            head = str(relation.get("head") or "")
            tail = str(relation.get("tail") or "")
            relation_text = str(relation.get("relation") or "")
            source_name = str(relation.get("source_name") or "")
            if head:
                relation_adjacency.setdefault(head, []).append(relation)
            if tail:
                relation_adjacency.setdefault(tail, []).append(relation)
            key = (head, relation_text, tail, source_name)
            evidence_indices: list[int] = []
            for doc_id in str(relation.get("evidence_id") or "").split("|"):
                doc_id = doc_id.strip()
                if doc_id in doc_id_to_index:
                    evidence_indices.append(doc_id_to_index[doc_id])
            if evidence_indices:
                relation_evidence_doc_indices[key] = list(dict.fromkeys(evidence_indices))
        return cls(
            entity_to_doc_indices={
                str(key): [int(index) for index in value]
                for key, value in payload.get("entity_to_doc_indices", {}).items()
                if isinstance(value, list)
            },
            entity_to_doc_weights={
                str(entity): {
                    int(doc_index): float(weight)
                    for doc_index, weight in weights.items()
                }
                for entity, weights in payload.get("entity_to_doc_weights", {}).items()
                if isinstance(weights, dict)
            },
            doc_to_entities=[
                [str(entity) for entity in entities]
                for entities in payload.get("doc_to_entities", [])
                if isinstance(entities, list)
            ],
            alias_lookup=build_alias_lookup(alias_map),
            generic_entities=set(str(key) for key in payload.get("entity_to_doc_indices", {})),
            teacher_relations=teacher_relations,
            relation_adjacency=relation_adjacency,
            relation_evidence_doc_indices=relation_evidence_doc_indices,
        )

    def _query_entities(self, query: str) -> list[str]:
        query_entities = extract_alias_entities(query, self.alias_lookup)
        for match in ENTITY_RUN_RE.finditer(query):
            token = match.group(0).strip()
            if _is_generic_entity_candidate(token) and token in self.generic_entities and token not in query_entities:
                query_entities.append(token)
            for entity in (f"activity:{token}", f"story:{token}", f"stage:{token}"):
                if entity in self.entity_to_doc_indices and entity not in query_entities:
                    query_entities.append(entity)
        if len(query_entities) < 3:
            for entity in sorted(self.generic_entities, key=len, reverse=True):
                if len(query_entities) >= 8:
                    break
                if entity.startswith(METADATA_ENTITY_PREFIXES):
                    continue
                if len(entity) < 3 and entity not in self.alias_lookup.values():
                    continue
                if entity in query and entity not in query_entities:
                    query_entities.append(entity)
        return query_entities

    def _add_relation_scores(
        self,
        scores: dict[int, float],
        query: str,
        query_entities: list[str],
        *,
        edge_weight: float = 1.5,
        endpoint_weight: float = 0.5,
        evidence_weight: float = 2.4,
        weak_evidence_weight: float = 0.45,
    ) -> None:
        seen_relations: set[tuple[str, str, str, str]] = set()
        query_entity_set = set(query_entities)
        query_terms = _relation_gate_terms(query)
        for entity in query_entities:
            for relation in self.relation_adjacency.get(entity, []):
                key = (
                    str(relation.get("head") or ""),
                    str(relation.get("relation") or ""),
                    str(relation.get("tail") or ""),
                    str(relation.get("source_name") or ""),
                )
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                endpoints = [str(relation.get("head") or ""), str(relation.get("tail") or "")]
                matched_endpoints = sum(
                    1
                    for endpoint in endpoints
                    if endpoint in query_entity_set or endpoint in query or endpoint in entity
                )
                relation_text = str(relation.get("relation") or "")
                relation_terms = _relation_gate_terms(" ".join([*endpoints, relation_text]))
                relation_overlap = bool(query_terms & relation_terms)
                endpoint_pair_match = matched_endpoints >= 2
                endpoint_phrase_match = any(
                    endpoint and len(endpoint) >= 3 and endpoint in query
                    for endpoint in endpoints
                )
                strong_relation_match = endpoint_pair_match or (
                    matched_endpoints >= 1 and (relation_overlap or endpoint_phrase_match)
                )
                relation_bonus = edge_weight * max(1, matched_endpoints)
                effective_evidence_weight = (
                    evidence_weight if strong_relation_match else weak_evidence_weight
                )
                for rank, doc_index in enumerate(self.relation_evidence_doc_indices.get(key, [])[:8]):
                    scores[doc_index] = scores.get(doc_index, 0.0) + relation_bonus * effective_evidence_weight / (rank + 1)
                for endpoint in endpoints:
                    for rank, doc_index in enumerate(self.entity_to_doc_indices.get(endpoint, [])[:128]):
                        scores[doc_index] = scores.get(doc_index, 0.0) + endpoint_weight / (rank + 1)
                if strong_relation_match:
                    for endpoint in endpoints:
                        for rank, doc_index in enumerate(self.entity_to_doc_indices.get(endpoint, [])[:32]):
                            scores[doc_index] = scores.get(doc_index, 0.0) + relation_bonus / (rank + 1)

    @staticmethod
    def _entity_node_weight(entity: str) -> float:
        if entity.startswith(METADATA_ENTITY_PREFIXES):
            return 0.25
        return 1.0

    def _entity_doc_limit(self, entity: str) -> int:
        if entity.startswith(METADATA_ENTITY_PREFIXES):
            return 48
        return 128

    def _relation_neighbors(self, entity: str) -> list[str]:
        neighbors: list[str] = []
        seen: set[str] = set()
        for relation in self.relation_adjacency.get(entity, []):
            for endpoint in (str(relation.get("head") or ""), str(relation.get("tail") or "")):
                if endpoint and endpoint != entity and endpoint not in seen:
                    seen.add(endpoint)
                    neighbors.append(endpoint)
        return neighbors

    def _add_ppr_scores(
        self,
        scores: dict[int, float],
        query_entities: list[str],
        *,
        propagation_weight: float = 0.35,
        max_hops: int = 3,
        doc_entity_limit: int = 16,
        max_active_entities: int = 256,
        relation_traversal_weight: float = 1.4,
        min_entity_mass: float = 1e-4,
    ) -> None:
        """Lightweight Personalized PageRank-style propagation on the hetero graph.

        The graph is traversed as entity -> chunk -> entity for a few hops, with
        teacher relation edges adding high-confidence entity -> entity jumps.
        This keeps MiniRAG dependency-free while giving multi-hop questions a
        chance to reach answer-bearing chunks that do not share query surface
        terms directly.
        """
        if not query_entities or max_hops <= 0 or propagation_weight <= 0:
            return

        query_entity_set = set(query_entities)
        entity_mass: dict[str, float] = {
            entity: self._entity_node_weight(entity)
            for entity in query_entities
            if entity in self.entity_to_doc_indices or entity in self.relation_adjacency
        }

        for hop in range(1, max_hops + 1):
            if not entity_mass:
                return
            hop_decay = propagation_weight ** hop
            doc_mass: dict[int, float] = {}

            for entity, mass in entity_mass.items():
                if mass < min_entity_mass:
                    continue
                node_weight = self._entity_node_weight(entity)
                docs = self.entity_to_doc_indices.get(entity, [])[: self._entity_doc_limit(entity)]
                for rank, doc_index in enumerate(docs):
                    edge_weight = min(self.entity_to_doc_weights.get(entity, {}).get(doc_index, 1.0), 4.0)
                    contribution = mass * node_weight * edge_weight / (rank + 1)
                    if contribution <= 0:
                        continue
                    doc_mass[doc_index] = doc_mass.get(doc_index, 0.0) + contribution
                    scores[doc_index] = scores.get(doc_index, 0.0) + contribution * hop_decay

            next_entity_mass: dict[str, float] = {}
            for doc_index, mass in doc_mass.items():
                if not (0 <= doc_index < len(self.doc_to_entities)):
                    continue
                entities = self.doc_to_entities[doc_index][:doc_entity_limit]
                if not entities:
                    continue
                share = mass / max(len(entities), 1)
                for offset, neighbor_entity in enumerate(entities, start=1):
                    if hop > 1 and neighbor_entity in query_entity_set:
                        continue
                    neighbor_weight = self._entity_node_weight(neighbor_entity)
                    next_entity_mass[neighbor_entity] = next_entity_mass.get(neighbor_entity, 0.0) + (
                        share * neighbor_weight / offset
                    )

            # Relation edges are high-confidence shortcuts; inject them into the
            # next frontier so relation facts can bridge entity names that never
            # appear together in the same retrieved chunk.
            for entity, mass in entity_mass.items():
                for neighbor_entity in self._relation_neighbors(entity):
                    if hop > 1 and neighbor_entity in query_entity_set:
                        continue
                    next_entity_mass[neighbor_entity] = next_entity_mass.get(neighbor_entity, 0.0) + (
                        mass * relation_traversal_weight
                    )

            entity_mass = dict(
                sorted(
                    (
                        (entity, mass)
                        for entity, mass in next_entity_mass.items()
                        if mass >= min_entity_mass
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:max_active_entities]
            )

    def search(
        self,
        query: str,
        documents: list[dict[str, Any]],
        *,
        top_k: int = 40,
        propagation_weight: float = 0.35,
    ) -> list[dict[str, Any]]:
        query_entities = self._query_entities(query)
        if not query_entities:
            return []

        scores: dict[int, float] = {}
        for entity in query_entities:
            direct_docs = self.entity_to_doc_indices.get(entity, [])
            for rank, doc_index in enumerate(direct_docs):
                edge_weight = self.entity_to_doc_weights.get(entity, {}).get(doc_index, 1.0)
                scores[doc_index] = scores.get(doc_index, 0.0) + min(edge_weight, 4.0) / (rank + 1)
        self._add_ppr_scores(scores, query_entities, propagation_weight=propagation_weight)
        self._add_relation_scores(scores, query, query_entities)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {
                "doc_index": doc_index,
                "score": float(score),
                "minirag_score": float(score),
                "document": documents[doc_index],
            }
            for doc_index, score in ranked
            if 0 <= doc_index < len(documents)
        ]


def build_and_save_minirag_graph(
    *,
    documents_path: Path = DOCUMENTS_PATH,
    alias_map_path: Path = OPERATOR_ALIAS_MAP_PATH,
    output_path: Path = MINIRAG_GRAPH_PATH,
    teacher_annotations_path: Path | list[Path] | None = None,
    progress: bool = False,
    progress_interval: int = 1000,
) -> dict[str, Any]:
    started = time.time()
    if progress:
        print(f"[minirag-build] load documents {documents_path}", file=sys.stderr, flush=True)
    documents = load_jsonl(documents_path)
    if progress:
        print(f"[minirag-build] documents={len(documents)}", file=sys.stderr, flush=True)
        print(f"[minirag-build] load aliases {alias_map_path}", file=sys.stderr, flush=True)
    alias_map = load_alias_map(alias_map_path)
    teacher_annotations: list[dict[str, Any]] | None = None
    if isinstance(teacher_annotations_path, list):
        teacher_annotations = []
        for path in teacher_annotations_path:
            if path.exists():
                if progress:
                    print(f"[minirag-build] load annotations {path}", file=sys.stderr, flush=True)
                teacher_annotations.extend(load_jsonl(path))
    elif teacher_annotations_path and teacher_annotations_path.exists():
        if progress:
            print(f"[minirag-build] load annotations {teacher_annotations_path}", file=sys.stderr, flush=True)
        teacher_annotations = load_jsonl(teacher_annotations_path)
    graph = build_minirag_graph(
        documents,
        alias_map,
        teacher_annotations=teacher_annotations,
        progress=progress,
        progress_interval=progress_interval,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        print(f"[minirag-build] write graph {output_path}", file=sys.stderr, flush=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    if progress:
        print(
            f"[minirag-build] done elapsed={time.time() - started:.1f}s "
            f"documents={graph['document_count']} entities={graph['entity_count']} "
            f"teacher_relations={len(graph.get('teacher_relations', []))}",
            file=sys.stderr,
            flush=True,
        )
    return graph
