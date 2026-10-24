from copy import deepcopy
from types import SimpleNamespace
import pytest

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_chain_scoring import (
    HybridEvidenceChainScoringMixin,
)


def fixture(scores=(8.0, 1.0), roles=(("action", "target"), ("context",)), *, adjusted=None):
    items = [{"doc_index": i, "document": {"id": f"doc{i}", "search_text": f"source{i}"},
              "rerank_score": float(3 - i)} for i in range(3)]
    chains = [
        {"name": "winning_source", "roles": set(roles[0]), "members": [
            {"item": items[0], "roles": set(roles[0])},
            {"item": items[1], "roles": {"context"}},
        ]},
        {"name": "other_source", "roles": set(roles[1]), "members": [
            {"item": items[0], "roles": set(roles[1])},
            {"item": items[2], "roles": {"context"}},
        ]},
    ]
    mixin = HybridEvidenceChainScoringMixin()
    mixin.reranker = SimpleNamespace(score=lambda **kwargs: list(scores))
    mixin._build_evidence_chains = lambda *a, **kw: chains
    mixin._build_generic_evidence_chains = lambda *a, **kw: []
    mixin._chain_member_sort_key = lambda member: member["item"]["doc_index"]
    mixin._render_chain_text = lambda chain: "SOURCE:" + chain["name"]
    mixin._compute_query_overlap_bonus = lambda *a, **kw: 0
    mixin._adjust_chain_score_for_query_type = adjusted or (lambda query, doc, score, mode: score)
    return mixin, items, chains


@pytest.mark.parametrize("mode", ["reasoning", "fact", "causality", "reveal", "relation", None])
def test_shared_document_keeps_winning_text_roles_and_model_score(mode):
    mixin, items, _ = fixture()
    assert mixin._apply_chain_model_scores("q", items, bridge_terms=[], batch_size=1, query_mode=mode)
    shared = items[0]
    assert shared["evidence_chain_text"].endswith("SOURCE:winning_source")
    assert shared["evidence_chain_roles"] == ["action", "target"]
    assert shared["evidence_chain_model_score"] == 8.0


def test_lower_raw_score_can_win_adjusted_member_score():
    def adjust(query, doc, score, mode):
        return -20.0 if doc["id"] == "doc0" and score == 8.0 else score
    mixin, items, _ = fixture(adjusted=adjust)
    mixin._apply_chain_model_scores("q", items, bridge_terms=[], batch_size=1, query_mode="reasoning")
    assert items[0]["evidence_chain_text"].endswith("SOURCE:other_source")
    assert items[0]["evidence_chain_model_score"] == 1.0
    assert items[0]["evidence_chain_roles"] == ["context"]


def test_unique_members_keep_their_own_chain():
    mixin, items, _ = fixture()
    mixin._apply_chain_model_scores("q", items, bridge_terms=[], batch_size=1, query_mode="reasoning")
    assert items[1]["evidence_chain_text"].endswith("SOURCE:winning_source")
    assert items[2]["evidence_chain_text"].endswith("SOURCE:other_source")


def test_one_chain_unchanged():
    mixin, items, chains = fixture(scores=(8.0,))
    mixin._build_evidence_chains = lambda *a, **kw: chains[:1]
    mixin._apply_chain_model_scores("q", items, bridge_terms=[], batch_size=1, query_mode="reasoning")
    assert items[0]["evidence_chain_model_score"] == items[1]["evidence_chain_model_score"] == 8.0
    assert "evidence_chain_text" not in items[2]


def test_no_reranker_leaves_inputs_unchanged():
    mixin, items, _ = fixture()
    mixin.reranker = None
    before = deepcopy(items)
    assert mixin._apply_chain_model_scores("q", items, bridge_terms=[], batch_size=1) is False
    assert items == before


def test_no_eligible_multimember_chain_leaves_inputs_unchanged():
    mixin, items, chains = fixture()
    for chain in chains:
        chain["members"] = chain["members"][:1]
    before = deepcopy(items)
    assert mixin._apply_chain_model_scores("q", items, bridge_terms=[], batch_size=1) is False
    assert items == before


def test_document_fields_and_document_scores_never_mutated():
    mixin, items, _ = fixture()
    before = deepcopy(items)
    mixin._apply_chain_model_scores("q", items, bridge_terms=[], batch_size=1, query_mode="reasoning")
    for a, b in zip(before, items):
        assert a["document"] == b["document"] and a["rerank_score"] == b["rerank_score"]


def test_stronger_existing_binding_is_not_partially_overwritten():
    mixin, items, _ = fixture()
    items[0].update(evidence_chain_score=100.0, evidence_chain_model_score=200.0,
                    evidence_chain_text="earlier_source", evidence_chain_roles=["motive"])
    before = deepcopy(items[0])
    mixin._apply_chain_model_scores("q", items, bridge_terms=[], batch_size=1, query_mode="reasoning")
    assert items[0] == before


def test_weaker_existing_binding_replaced_atomically_even_if_raw_score_larger():
    mixin, items, _ = fixture()
    items[0].update(evidence_chain_score=-100.0, evidence_chain_model_score=200.0,
                    evidence_chain_text="earlier_source", evidence_chain_roles=["motive"])
    mixin._apply_chain_model_scores("q", items, bridge_terms=[], batch_size=1, query_mode="reasoning")
    assert items[0]["evidence_chain_model_score"] == 8.0
    assert items[0]["evidence_chain_text"].endswith("SOURCE:winning_source")
    assert items[0]["evidence_chain_roles"] == ["action", "target"]
