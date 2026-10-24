"""Selected-seed retention is an experimental invariant, not a gold label."""
import importlib.util
from pathlib import Path
import pytest

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_chain_building import (
    HybridEvidenceChainBuildingMixin as Original,
)

spec = importlib.util.spec_from_file_location(
    "asa_chain_seed_candidate", Path(__file__).with_name("chain_seed_preservation_candidate.py"))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
Candidate = candidate.HybridEvidenceChainBuildingMixin


def fixture(cls, count=10):
    mixin = cls()
    candidates = [
        {"doc_index": i, "document": {"id": str(i), "clean_text": f"甲乙丙丁 {i}"},
         "rerank_score": 100 - i}
        for i in range(count)]
    mixin._extract_chain_query_terms = lambda query: ["甲", "乙"]
    mixin._document_text = lambda doc: doc["clean_text"]
    mixin._classify_evidence_roles = lambda query, doc, **kwargs: (
        {"target"} if doc["id"] == "0" else {"motive", "outcome", "context"})
    mixin._document_link_score = lambda *args, **kwargs: 1.0
    return mixin, candidates


def test_original_drops_highest_score_seed_with_weaker_role_count():
    mixin, items = fixture(Original)
    chains = mixin._build_evidence_chains("甲乙", items, bridge_terms=[])
    assert chains and all(0 not in [m["item"]["doc_index"] for m in c["members"]] for c in chains)


def test_candidate_retains_highest_score_seed():
    mixin, items = fixture(Candidate)
    chains = mixin._build_evidence_chains("甲乙", items, bridge_terms=[])
    assert any(0 in [m["item"]["doc_index"] for m in c["members"]] for c in chains)
    assert len({tuple(m["item"]["doc_index"] for m in c["members"]) for c in chains}) > 1


@pytest.mark.parametrize("count", [2, 5, 8, 10, 30])
def test_candidate_retains_member_and_seed_limits(count):
    mixin, items = fixture(Candidate, count)
    chains = mixin._build_evidence_chains("甲乙", items, bridge_terms=[])
    assert len(chains) <= min(24, count)
    assert all(len(c["members"]) <= 8 for c in chains)
    assert all(len({m["item"]["doc_index"] for m in c["members"]}) == len(c["members"]) for c in chains)


def test_two_member_input_membership_matches_original():
    old, old_items = fixture(Original, 2)
    new, new_items = fixture(Candidate, 2)
    a = old._build_evidence_chains("甲乙", old_items, bridge_terms=[])
    b = new._build_evidence_chains("甲乙", new_items, bridge_terms=[])
    def memberships(chains):
        return sorted(sorted(m["item"]["doc_index"] for m in c["members"]) for c in chains)
    assert memberships(a) == memberships(b)


def test_no_items_no_chains():
    mixin, _ = fixture(Candidate)
    assert mixin._build_evidence_chains("q", [], bridge_terms=[]) == []
