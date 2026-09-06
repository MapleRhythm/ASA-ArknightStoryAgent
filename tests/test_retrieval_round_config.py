from asa_arknight_story_agent.runtime_config import resolve_max_retrieval_rounds


def test_explicit_third_round_is_reachable_from_runtime_config() -> None:
    assert resolve_max_retrieval_rounds(None, {"max_retrieval_rounds": 3}) == 3
    assert resolve_max_retrieval_rounds(3, {}) == 3


def test_retrieval_round_budget_keeps_safe_default_and_bounds() -> None:
    assert resolve_max_retrieval_rounds(None, {}) == 2
    assert resolve_max_retrieval_rounds(0, {}) == 1
    assert resolve_max_retrieval_rounds(99, {}) == 3
