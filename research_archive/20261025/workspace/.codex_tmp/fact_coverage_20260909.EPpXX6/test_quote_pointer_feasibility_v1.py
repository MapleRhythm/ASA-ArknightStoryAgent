from audit_quote_pointer_feasibility_v1 import options_for, best_union
from quote_pointer_protocol_v1 import annotate


def test_exact_quote_within_one_chunk():
    evidence = "[E1] " + "字" * 70 + "甲乙" + "天" * 88
    _, catalog = annotate(evidence)
    options = options_for("甲乙", "E1", evidence, catalog)
    assert len(options) == 1 and options[0]["quote_ids"] == ["Q1"]
    assert options[0]["quote_chars"] == 80


def test_boundary_crossing_counts_both_chunks():
    evidence = "[E1] " + "字" * 78 + "甲乙丙丁" + "天" * 78
    _, catalog = annotate(evidence)
    options = options_for("甲乙丙丁", "E1", evidence, catalog)
    assert options == [{"quote_ids": ["Q1", "Q2"], "quote_chars": 160}]


def test_two_original_refs_can_require_three_chunks():
    evidence = "[E1] " + "字" * 78 + "甲乙丙丁" + "天" * 78 + "尾"
    _, catalog = annotate(evidence)
    choices = [options_for(q, "E1", evidence, catalog) for q in ("甲乙丙丁", "尾")]
    best = best_union(choices, catalog)
    assert best["quote_ids"] == ["Q1", "Q2", "Q3"] and not best["fits_fact_limits"]


def test_repeated_quote_retains_all_choices():
    evidence = "[E1] " + "甲" * 81
    _, catalog = annotate(evidence)
    options = options_for("甲", "E1", evidence, catalog)
    assert len(options) == 2
    assert best_union([options], catalog)["quote_chars"] == 1


def test_wrong_eid_and_whitespace_only_match_are_not_fixed():
    evidence = "[E1] 甲 乙\n[E2] 丙"
    _, catalog = annotate(evidence)
    assert options_for("甲乙", "E1", evidence, catalog) == []
    assert options_for("甲 乙", "E2", evidence, catalog) == []
    assert options_for("", "E1", evidence, catalog) == []


def test_missing_and_malformed_refs_not_dropped():
    _, catalog = annotate("[E1] 内容")
    valid = [{"quote_ids": ["Q1"], "quote_chars": 2}]
    assert best_union([], catalog) is None
    assert best_union([[], valid], catalog) is None
    assert best_union([valid, valid, valid], catalog) is None


def test_two_refs_same_chunk_deduplication():
    evidence = "[E1] 甲 乙"
    _, catalog = annotate(evidence)
    best = best_union([options_for(q, "E1", evidence, catalog) for q in ("甲", "乙")], catalog)
    assert best["quote_ids"] == ["Q1"] and best["quote_chars"] == 3
