import importlib.util
from pathlib import Path

PATH = Path(__file__).with_name("prepare_fairytaleqa.py")
SPEC = importlib.util.spec_from_file_location("prep", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_tokenizer():
    assert MOD.tokens("A fox's DOG!") == ["a", "fox", "s", "dog"]


def test_bm25_prefers_matching_document():
    scores = MOD.bm25_scores("red fox", ["blue dog", "the red fox ran"])
    assert scores[1] > scores[0]


def test_bm25_stable_when_no_terms_match():
    assert MOD.bm25_scores("zebra", ["a", "b"]) == [0.0, 0.0]


def test_metrics_exact_and_multiple_references():
    from run_fairytaleqa import metrics
    assert metrics("The red fox.", ["a blue dog", "Red fox"]) == {
        "normalized_exact_match": 1.0, "token_f1": 1.0, "rouge_l_f1": 1.0}


def test_metrics_order_and_duplicates():
    from run_fairytaleqa import metrics
    score = metrics("fox red red", ["red fox"])
    assert score["normalized_exact_match"] == 0
    assert score["token_f1"] == 0.8
    assert score["rouge_l_f1"] == 0.4


def test_metrics_empty_prediction():
    from run_fairytaleqa import metrics
    assert all(v == 0 for v in metrics("", ["red fox"]).values())
