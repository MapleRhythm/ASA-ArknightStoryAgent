from __future__ import annotations

from types import SimpleNamespace

from goldenglow.inference.cpu_pipeline import (
    CPUInferencePipeline,
    HypothesisDocument,
    _grounding_evidence_pool,
    render_short_evidence_brief,
    render_minirag_hints_for_prompt,
)


class _Reranker:
    def score(self, *, query: str, documents: list[str], batch_size: int) -> list[float]:
        del query, batch_size
        return [float(index) for index, _ in enumerate(documents)]


def test_crag_keeps_chain_as_ranking_metadata_not_prompt_text() -> None:
    retriever = SimpleNamespace(reranker=_Reranker())
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=object(),
        enable_crag_refinement=True,
        crag_refine_top_sentences=1,
        crag_refine_max_sentences=8,
    )
    hypothesis = HypothesisDocument(
        question="凯尔希为什么阻止Mon3tr攻击？",
        intent="plot_reasoning",
        query_type="causality",
        entities=["凯尔希", "Mon3tr"],
        keywords=["攻击", "阻止"],
        expected_answer_type="原因",
        dialogue_context="",
    )
    evidence = [
        {
            "doc_index": 1,
            "document": {
                "id": "story#chunk-0001",
                "clean_text": "凯尔希让Mon3tr停下。她解释攻击会让尖刺贯穿自己的胸膛。无关背景。",
                "search_text": "凯尔希让Mon3tr停下。她解释攻击会让尖刺贯穿自己的胸膛。无关背景。",
            },
            "evidence_chain_text": "CHAIN_SECRET_SENTINEL 这是仅供排序的跨段链文本。",
        }
    ]

    refined = pipeline.refine_evidence_strips(hypothesis.question, hypothesis, evidence)
    prompt_text = render_short_evidence_brief(
        refined,
        max_chars_per_doc=1000,
        max_total_chars=2000,
    )

    assert "evidence_chain_text" in evidence[0]
    assert "CHAIN_SECRET_SENTINEL" not in refined[0]["document"]["clean_text"]
    assert "CHAIN_SECRET_SENTINEL" not in prompt_text
    assert "CHAIN_SECRET_SENTINEL" not in _grounding_evidence_pool(evidence)
    assert "CHAIN_SECRET_SENTINEL" not in render_minirag_hints_for_prompt(evidence, hypothesis)
    assert "evidence_chain_text" not in refined[0]
    assert refined[0]["prompt_prefer_clean_text"] is True
