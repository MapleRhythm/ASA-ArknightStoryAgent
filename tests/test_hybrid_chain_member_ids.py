from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_chain_scoring import (
    HybridEvidenceChainScoringMixin,
)


class _Reranker:
    def score(self, *, query, documents, batch_size):
        del query, batch_size
        return [1.0 for _ in documents]


class _ScoringHarness(HybridEvidenceChainScoringMixin):
    def __init__(self, chain):
        self.reranker = _Reranker()
        self.chain = chain

    def _build_evidence_chains(self, query, candidates, *, bridge_terms):
        del query, candidates, bridge_terms
        return [self.chain]

    def _build_generic_evidence_chains(self, query, candidates, *, bridge_terms):
        del query, candidates, bridge_terms
        return []

    def _chain_member_sort_key(self, member):
        return int(member["item"]["doc_index"])

    def _render_chain_text(self, chain):
        del chain
        return "甲\n乙"

    def _compute_query_overlap_bonus(
        self, query, document, bridge_terms, *, query_mode
    ):
        del query, document, bridge_terms, query_mode
        return 0.0

    def _adjust_chain_score_for_query_type(
        self, query, document, chain_score, query_mode
    ):
        del query, document, query_mode
        return chain_score


def _candidate(doc_index: int, doc_id: str) -> dict:
    return {
        "doc_index": doc_index,
        "document": {"id": doc_id, "clean_text": doc_id},
    }


def test_chain_score_records_real_member_document_ids() -> None:
    candidates = [_candidate(1, "doc-a"), _candidate(2, "doc-b")]
    chain = {
        "roles": {"context", "outcome"},
        "members": [
            {"item": candidates[0], "roles": {"context"}},
            {"item": candidates[1], "roles": {"outcome"}},
        ],
    }
    harness = _ScoringHarness(chain)

    assert harness._apply_chain_model_scores(
        "问题",
        candidates,
        bridge_terms=[],
        batch_size=2,
    )
    assert candidates[0]["evidence_chain_member_ids"] == ["doc-a", "doc-b"]
    assert candidates[1]["evidence_chain_member_ids"] == ["doc-a", "doc-b"]
