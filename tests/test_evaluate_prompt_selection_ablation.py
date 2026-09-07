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
    assert result["strategies"]["coverage@2"]["summary"]["recall"] == 1.0
    assert result["strategies"]["oracle@2"]["summary"]["all_gold_rate"] == 1.0


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
