from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_retrieval_recall import evaluate, extract_gold_units, text_hit_score
from goldenglow.config import QueryConfig
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever
from mine_coverage_reranker_data import select_group_candidates


def _item(text: str, *, story_id: str = "story-a") -> dict[str, Any]:
    return {
        "document": {
            "clean_text": text,
            "search_text": text,
            "story_id": story_id,
        },
        "rerank_score": 1.0,
    }


class _FakeRetriever:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits

    def search(self, query: str, *, config: QueryConfig) -> list[dict[str, Any]]:
        del query, config
        return self.hits

    def search_pre_rerank(self, query: str, *, config: QueryConfig) -> list[dict[str, Any]]:
        return self.search(query, config=config)


def test_extract_gold_units_uses_only_answer_evidence_ids() -> None:
    record = {
        "answer_evidence": ["E1", "E3"],
        "candidates": [
            {
                "label": "positive",
                "text": "[E1] 第一条必要证据内容足够长。\n[E2] 标注外的过渡内容。\n[E3] 第三条必要证据内容足够长。",
            }
        ],
    }
    units = extract_gold_units(record)
    assert [unit["evidence_id"] for unit in units] == ["E1", "E3"]
    assert all("过渡内容" not in unit["text"] for unit in units)


def test_short_exact_gold_unit_is_not_rejected_by_long_chain_thresholds() -> None:
    gold = "苏茜说感染者的部分薪酬会换成药物发放。"
    candidate = "前文内容。苏茜说感染者的部分薪酬会换成药物发放。后文内容。"
    assert text_hit_score(
        candidate,
        gold,
        jaccard_threshold=0.25,
        overlap_threshold=0.32,
        min_overlap_grams=60,
        min_candidate_grams=80,
    ) == 1.0


def test_evaluate_reports_complete_coverage_separately_from_any_hit() -> None:
    record = {
        "query": "两项措施分别是什么？",
        "query_type": "fact",
        "answer_evidence": ["E1", "E2"],
        "candidates": [
            {
                "label": "positive",
                "text": (
                    "[E1] 第一项措施是为感染者提供完整防护设备并限制工作时间。\n"
                    "[E2] 第二项措施是保证感染者获得同等医疗救助和工伤处理。"
                ),
            }
        ],
    }
    result = evaluate(
        [record],
        _FakeRetriever(
            [
                _item("第一项措施是为感染者提供完整防护设备并限制工作时间。"),
                _item("第二项措施是保证感染者获得同等医疗救助和工伤处理。"),
            ]
        ),
        query_config=QueryConfig(rerank_top_k=2),
        top_ks=[1, 2],
        jaccard_threshold=0.25,
        overlap_threshold=0.32,
        min_overlap_grams=4,
        min_candidate_grams=4,
    )
    assert result["overall"]["recall"]["@1"] == 1.0
    assert result["overall"]["complete_recall"]["@1"] == 0.0
    assert result["overall"]["mean_coverage"]["@1"] == 0.5
    assert result["overall"]["complete_recall"]["@2"] == 1.0
    assert result["overall"]["mean_coverage"]["@2"] == 1.0


def test_coverage_selector_replaces_near_duplicate_with_complementary_evidence() -> None:
    retriever = object.__new__(ArknightsHybridRetriever)
    duplicate = "感染者工人每天工作时间不能超过十五小时并需要提供防护设备。"
    ranked = [
        _item(duplicate),
        _item(duplicate + "。"),
        _item("议会随后通过了医疗救助和工伤处理的补充条款。"),
    ]
    selected = retriever._select_coverage_candidates(
        ranked,
        top_k=2,
        redundancy_weight=0.9,
        story_coherence_weight=0.0,
        role_novelty_weight=0.0,
        redundancy_threshold=0.6,
    )
    assert selected[0]["document"]["clean_text"] == duplicate
    assert "医疗救助" in selected[1]["document"]["clean_text"]


def test_group_selection_never_labels_other_required_evidence_as_negative() -> None:
    gold_units = [
        {"evidence_id": "E1", "text": "第一项"},
        {"evidence_id": "E2", "text": "第二项"},
    ]
    candidates = [
        {
            "text": "第一项证据",
            "kind": "document",
            "retrieval_rank": 1,
            "covered_unit_indices": [0],
        },
        {
            "text": "第二项证据",
            "kind": "document",
            "retrieval_rank": 2,
            "covered_unit_indices": [1],
        },
        {
            "text": "无关高排名内容",
            "kind": "document",
            "retrieval_rank": 3,
            "covered_unit_indices": [],
        },
    ]
    selected = select_group_candidates(
        candidates,
        gold_units,
        max_group_size=3,
        max_positives_per_unit=1,
    )
    positives = [item for item in selected if item["relevance"] > 0]
    negatives = [item for item in selected if item["relevance"] == 0]
    assert {item["covered_evidence_ids"][0] for item in positives} == {"E1", "E2"}
    assert [item["text"] for item in negatives] == ["无关高排名内容"]
