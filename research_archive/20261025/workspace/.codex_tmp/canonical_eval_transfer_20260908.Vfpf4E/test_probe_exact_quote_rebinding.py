import copy

from probe_exact_quote_rebinding import rebind


QUOTE = "这是一段用于测试逐字匹配的完整原文，不允许改动。"


def proposal():
    return {"supported_facts": [{"fact": "仍需语义审核", "evidence_refs": [{"evidence_id": "E9", "quote": QUOTE}]}]}


def test_unique_exact_visible_match_only_changes_id():
    original = proposal()
    before = copy.deepcopy(original)
    updated, edits = rebind(original, {"E1": QUOTE})
    assert original == before
    assert updated["supported_facts"][0]["evidence_refs"] == [{"evidence_id": "E1", "quote": QUOTE}]
    assert updated["supported_facts"][0]["fact"] == "仍需语义审核"
    assert len(edits) == 1


def test_ambiguous_match_is_not_rebound():
    assert rebind(proposal(), {"E1": QUOTE, "E2": QUOTE})[1] == []


def test_existing_valid_id_is_preserved():
    assert rebind(proposal(), {"E9": QUOTE, "E1": QUOTE})[1] == []


def test_no_quote_synthesis_or_partial_match():
    assert rebind(proposal(), {"E1": QUOTE.replace("完整", "新编")})[1] == []


def test_empty_visible_context_cannot_use_hidden_source():
    assert rebind(proposal(), {})[1] == []


def test_short_quote_not_rebound():
    row = proposal()
    row["supported_facts"][0]["evidence_refs"][0]["quote"] = "是"
    assert rebind(row, {"E1": "是"})[1] == []
