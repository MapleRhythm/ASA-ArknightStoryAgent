"""Tests for the isolated issue classifier; no production behavior change."""
import pytest
from probe_quote80_ablation_v1 import remaining_issues


@pytest.mark.parametrize("issue", [
    "supported_fact_1_ref_1_quote_over_80",
    "inferred_fact_12_ref_2_quote_over_80",
])
def test_only_exact_individual_80_issue_is_removed(issue):
    assert remaining_issues([issue]) == []


@pytest.mark.parametrize("issue", [
    "supported_fact_1_ref_1_quote_not_found",
    "supported_fact_1_ref_1_quote_over_800",
    "supported_fact_1_quote_total_over_160",
    "answer_quote_total_over_400:421",
    "supported_fact_1_ref_1_missing_evidence_id",
    "supported_fact_1_ref_1_evidence_id_not_found:E999",
    "supported_fact_1_has_terms_outside_quotes:哥哥",
    "prefix_supported_fact_1_ref_1_quote_over_80",
    "too_many_supported_facts:7>6",
])
def test_all_other_failures_stay_unchanged(issue):
    source = [issue]
    assert remaining_issues(source) == source
    assert source == [issue]
