import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_prompt_selection_ablation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_prompt_selection_ablation", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _candidate(doc_id: str, text: str, score: float) -> dict:
    return {
        "id": doc_id,
        "clean_text": text,
        "search_text": text,
        "fusion_score": score,
    }


def test_coverage_can_keep_complementary_gold_evidence() -> None:
    row = {
        "id": "q1",
        "question": "甲行动的原因和乙最后的结果是什么？",
        "question_family": "family-a",
        "gold_evidence_ids": ["a", "c"],
        "hypothesis": {
            "question": "甲行动的原因和乙最后的结果是什么？",
            "intent": "plot_reasoning",
            "query_type": "causality",
            "entities": ["甲", "乙"],
            "keywords": ["行动", "原因", "结果"],
            "expected_answer_type": "原因和结果",
            "dialogue_context": "",
        },
        "stages": {
            "rerank": [
                _candidate("a", "甲采取行动，原因是保护同伴。", 10.0),
                _candidate("b", "甲的行动发生在夜间，周围还有许多旁观者。", 9.0),
                _candidate("c", "乙最后的结果是安全离开。", 1.0),
            ]
        },
    }

    result = MODULE.evaluate_rows([row], top_ks=[2], mmr_lambda=0.72)

    assert result["strategies"]["top_score@2"]["summary"]["recall"] == 0.5
    assert (
        result["strategies"]["top_score@2"]["rows"][0]["coverage_attribution"]
        == "selector_drop"
    )
    assert result["strategies"]["coverage@2"]["summary"]["recall"] == 1.0
    assert result["strategies"]["oracle@2"]["summary"]["all_gold_rate"] == 1.0
    assert result["common_pool"]["summary"]["dedupe_drop_rate"] == 0.0


def test_family_macro_gives_each_family_equal_weight() -> None:
    records = [
        {
            "question_family": "large",
            "recall": 1.0,
            "any_hit": 1,
            "all_gold": 1,
            "selected_count": 1,
        },
        {
            "question_family": "large",
            "recall": 1.0,
            "any_hit": 1,
            "all_gold": 1,
            "selected_count": 1,
        },
        {
            "question_family": "small",
            "recall": 0.0,
            "any_hit": 0,
            "all_gold": 0,
            "selected_count": 1,
        },
    ]

    assert MODULE.aggregate(records)["recall"] == 2 / 3
    assert MODULE.family_macro(records)["recall"] == 0.5


def test_common_dedupe_reports_dropped_gold_and_kept_document() -> None:
    row = {
        "id": "q-dedupe",
        "question": "甲是谁？",
        "question_family": "family-dedupe",
        "gold_evidence_ids": ["gold"],
        "stages": {
            "rerank": [
                _candidate("decoy", "甲是来自拉特兰的信使。", 10.0),
                _candidate("gold", "甲是来自拉特兰的信使。", 9.0),
                _candidate("other", "乙来自维多利亚。", 8.0),
            ]
        },
    }

    result = MODULE.evaluate_rows([row], top_ks=[2], mmr_lambda=0.72)

    pool = result["common_pool"]
    assert pool["summary"]["rerank_recall"] == 1.0
    assert pool["summary"]["deduped_recall"] == 0.0
    assert pool["coverage_attribution"] == {"dedupe_drop": 1}
    assert pool["rows"][0]["dedupe_dropped_gold_ids"] == ["gold"]
    assert pool["rows"][0]["dedupe_drop_explanations"] == [
        {
            "dropped_id": "gold",
            "reason": "text_similarity",
            "kept_id": "decoy",
            "similarity": 1.0,
            "dropped_id_in_kept_chain_members": False,
            "dropped_clean_text_fully_preserved": True,
            "dropped_clean_text_token_recall": 1.0,
        }
    ]
    assert result["strategies"]["oracle@2"]["coverage_attribution"] == {
        "dedupe_drop": 1
    }


def test_visible_metrics_include_exact_chain_member_ids() -> None:
    outer = _candidate("outer", "包含两个证据的链。", 10.0)
    outer["evidence_chain_member_ids"] = ["outer", "gold"]
    row = {
        "id": "q-chain",
        "question": "链中是否包含答案？",
        "question_family": "family-chain",
        "gold_evidence_ids": ["gold"],
        "stages": {"rerank": [outer, _candidate("other", "其他内容。", 1.0)]},
    }

    result = MODULE.evaluate_rows([row], top_ks=[1], mmr_lambda=0.72)

    pool = result["common_pool"]["summary"]
    assert pool["deduped_recall"] == 0.0
    assert pool["deduped_visible_recall"] == 1.0
    strategy = result["strategies"]["top_score@1"]
    assert strategy["summary"]["recall"] == 0.0
    assert strategy["visible_summary"]["recall"] == 1.0
