from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from answer_support_teacher_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    build_messages,
    build_result_record,
    clean_record,
    leaked_supervision_fields,
    teacher_payload,
    validate_judgement,
)
from materialize_answer_support_teacher import materialize_record  # noqa: E402
from train_coverage_reranker import read_groups  # noqa: E402


def _record() -> dict:
    return {
        "query": "法案规定了什么？",
        "query_type": "fact",
        "source_name": "sample.json",
        "story_family": "sample",
        "answer": "法案限制工时，并要求提供防护设备。",
        "gold_evidence": [{"evidence_id": "E7", "text": "不应进入教师管线"}],
        "gold_evidence_ids": ["E7"],
        "coverable_evidence_ids": ["E7"],
        "candidates": [
            {
                "candidate_id": "C001",
                "text": "法案规定每日工时不得超过十五小时。",
                "kind": "document",
                "retrieval_rank": 1,
                "covered_evidence_ids": ["E7"],
                "relevance": 1.0,
                "rerank_score": 9.9,
                "fusion_score": 0.8,
            },
            {
                "candidate_id": "C002",
                "text": "工厂必须给工人提供同等的防护设备。",
                "kind": "document",
                "retrieval_rank": 2,
                "covered_evidence_ids": [],
                "relevance": 0.0,
            },
            {
                "candidate_id": "C003",
                "text": "议会正在讨论另一项税收法案。",
                "kind": "document",
                "retrieval_rank": 3,
                "covered_evidence_ids": [],
                "relevance": 0.0,
            },
        ],
    }


def _judgement() -> dict:
    return {
        "protocol": PROTOCOL_VERSION,
        "claims": [["Q1", "法案限制工时"], ["Q2", "法案要求提供防护设备"]],
        "ranking": [],
        "judgements": [],
        "uncovered_claim_ids": [],
    }


def _aliased_judgement() -> dict:
    from answer_support_teacher_protocol import prompt_candidate_aliases

    mapping = prompt_candidate_aliases(_record())
    inverse = {stored: alias for alias, stored in mapping.items()}
    judgement = _judgement()
    judgement["ranking"] = [inverse["C001"], inverse["C002"], inverse["C003"]]
    judgement["judgements"] = [
        [inverse["C001"], "D", ["Q1"], "每日工时不得超过十五小时"],
        [inverse["C002"], "D", ["Q2"], "必须给工人提供同等的防护设备"],
        [inverse["C003"], "X", [], ""],
    ]
    return judgement


def test_clean_record_removes_all_gold_and_legacy_labels() -> None:
    cleaned = clean_record(_record())
    assert leaked_supervision_fields(cleaned) == []
    assert "gold_evidence" not in cleaned
    assert "relevance" not in cleaned["candidates"][0]
    assert "covered_evidence_ids" not in cleaned["candidates"][0]


def test_teacher_payload_and_prompt_do_not_expose_rank_scores_or_gold() -> None:
    record = _record()
    payload = teacher_payload(record)
    serialized_payload = json.dumps(payload, ensure_ascii=False)
    serialized_messages = json.dumps(build_messages(record, 3), ensure_ascii=False)
    for forbidden in (
        "gold_evidence",
        "required_evidence",
        "covered_evidence_ids",
        "relevance",
        "original_rank",
        "retrieval_rank",
        "rerank_score",
        "fusion_score",
        "E7",
        "不应进入教师管线",
    ):
        assert forbidden not in serialized_payload
        assert forbidden not in serialized_messages


def test_prompt_teaches_pointwise_ranking_not_coverage_interleaving() -> None:
    serialized_messages = json.dumps(build_messages(clean_record(_record()), 3), ensure_ascii=False)
    assert "每个 (query,candidate) 独立打分" in serialized_messages
    assert "不要为了让不同主张轮流出现而改变排序" in serialized_messages
    assert "优先让不同主张尽早覆盖" not in serialized_messages


def test_validator_expands_omitted_irrelevant_candidates() -> None:
    result = validate_judgement(clean_record(_record()), _aliased_judgement())
    assert [item["candidate_id"] for item in result["ranking"]] == [
        "C001",
        "C002",
        "C003",
    ]
    assert result["ranking"][2]["label"] == "hard_context"
    assert result["uncovered_claim_ids"] == []


def test_validator_rejects_non_verbatim_positive_quote() -> None:
    judgement = _aliased_judgement()
    judgement["judgements"][0][3] = "正文中不存在的依据"
    with pytest.raises(ValueError, match="not verbatim"):
        validate_judgement(clean_record(_record()), judgement)


def test_validator_rejects_inconsistent_uncovered_claims() -> None:
    judgement = _aliased_judgement()
    judgement["uncovered_claim_ids"] = ["Q2"]
    with pytest.raises(ValueError, match="inconsistent"):
        validate_judgement(clean_record(_record()), judgement)


def test_result_record_and_materializer_never_restore_gold() -> None:
    record = _record()
    judgement = validate_judgement(clean_record(record), _aliased_judgement())
    result = build_result_record(
        record,
        judgement=judgement,
        provider="test",
        model="test-model",
        usage={},
        finish_reason="stop",
    )
    assert leaked_supervision_fields(result) == []
    train = materialize_record(
        result,
        max_group_size=3,
        max_positives_per_claim=1,
        max_hard_context=1,
    )
    assert train is not None
    assert [candidate["relevance"] for candidate in train["candidates"]] == [1.0, 1.0, 0.0]
    assert "gold_evidence" not in train


def test_trainer_rejects_any_raw_teacher_record(tmp_path: Path) -> None:
    raw = {
        "query": "问题",
        "teacher": {"ranking": []},
        # Legacy relevance must not make raw teacher output trainable.
        "candidates": [
            {"text": "正例", "relevance": 1.0},
            {"text": "负例", "relevance": 0.0},
        ],
    }
    path = tmp_path / "raw_teacher.jsonl"
    path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw teacher output"):
        read_groups(path)
