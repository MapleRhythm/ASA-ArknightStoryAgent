from asa_arknight_story_agent.inference.pipeline.scheduler import (
    AdaptiveRoundScheduler,
    evidence_observation_key,
    query_key,
)


def test_query_key_only_normalizes_spacing_and_case() -> None:
    assert query_key("  玛恩纳   为什么？ ") == "玛恩纳为什么？"
    assert query_key("A  B") == "ab"


def test_prepare_queries_filters_duplicates_without_rewriting_content() -> None:
    scheduler = AdaptiveRoundScheduler()
    fresh, duplicates = scheduler.prepare_queries(["甲 乙", "甲乙", "丙"])
    assert fresh == ["甲 乙", "丙"]
    assert duplicates == 1
    fresh_again, duplicates_again = scheduler.prepare_queries(["丙", "丁"])
    assert fresh_again == ["丁"]
    assert duplicates_again == 1


def test_observe_evidence_counts_only_new_document_identities() -> None:
    scheduler = AdaptiveRoundScheduler()
    assert scheduler.observe_evidence([{"document": {"id": "d1"}}, {"document": {"id": "d2"}}]) == 2
    assert scheduler.observe_evidence([{"document": {"id": "d2"}}, {"document": {"id": "d3"}}]) == 1


def test_observe_evidence_counts_expanded_chain_on_same_document() -> None:
    scheduler = AdaptiveRoundScheduler()
    first = {"document": {"id": "d1", "clean_text": "甲"}}
    expanded = {
        "document": {"id": "d1", "clean_text": "甲"},
        "evidence_chain_text": "甲\n乙：补充上下文",
    }
    assert scheduler.observe_evidence([first]) == 1
    assert scheduler.observe_evidence([expanded]) == 1
    assert evidence_observation_key(first) != evidence_observation_key(expanded)


def test_scheduler_allows_one_empty_recovery_round_then_stops() -> None:
    scheduler = AdaptiveRoundScheduler()
    scheduler.prepare_queries(["first"])
    scheduler.observe_evidence([])
    assert scheduler.can_continue(
        round_index=1,
        max_rounds=3,
        pending_queries=["second"],
    ) == (True, "new_query_recovery")
    assert scheduler.can_continue(
        round_index=2,
        max_rounds=3,
        pending_queries=["third"],
    ) == (False, "no_new_evidence")
