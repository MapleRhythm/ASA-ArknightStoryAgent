from __future__ import annotations

from types import SimpleNamespace

from goldenglow.inference.cpu_pipeline import (
    CPUInferencePipeline,
    ConclusionResult,
    HypothesisDocument,
    _grounding_evidence_pool,
    render_short_evidence_brief,
    render_minirag_hints_for_prompt,
    select_prompt_evidence,
    validate_conclusion_grounding,
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
    assert "她解释攻击会让尖刺贯穿自己的胸膛" in refined[0]["document"]["clean_text"]
    assert refined[0]["crag_refinement"]["context_sentence_indices"]


def test_prompt_selection_preserves_finalized_rescue_order_across_score_scales() -> None:
    hypothesis = HypothesisDocument(
        question="娜塔莉娅怎样解释乌萨斯依靠战争维持运转？",
        intent="plot_reasoning",
        query_type="reasoning",
        entities=["娜塔莉娅", "乌萨斯"],
        keywords=["战争", "维持运转"],
        expected_answer_type="剧情解释",
        dialogue_context="",
    )
    rescue = {
        "doc_index": 1,
        "document": {
            "id": "story#rescued",
            "clean_text": "乌萨斯之所以能走到今天，是因为战争。",
            "search_text": "乌萨斯之所以能走到今天，是因为战争。",
        },
        # RRF scores are intentionally much smaller than reranker/chain scores.
        "fusion_score": 0.02,
    }
    reranked = {
        "doc_index": 2,
        "document": {
            "id": "story#reranked",
            "clean_text": "相关但不能回答问题的背景。",
            "search_text": "相关但不能回答问题的背景。",
        },
        "rerank_score": -2.0,
        "evidence_chain_score": 2.5,
    }

    selected = select_prompt_evidence(
        hypothesis.question,
        hypothesis,
        [rescue, reranked],
        prompt_evidence_top_k=1,
    )

    assert selected == [rescue]


def test_quote_warning_keeps_grounded_partial_answer_caveat() -> None:
    question = "血骑士为何买田，村民又怎样认出他？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["血骑士", "村民"],
        keywords=["买田", "认出"],
        expected_answer_type="因果与事件细节",
        dialogue_context="",
    )
    evidence = [
        {
            "document": {
                "id": "story#blood-knight",
                "clean_text": "血骑士：我买下了这附近的某块田。村民：您是血骑士？血骑士：过去是。",
                "search_text": "血骑士：我买下了这附近的某块田。村民：您是血骑士？血骑士：过去是。",
            }
        }
    ]
    final_answer = "可确认血骑士买下了这附近的某块田，村民问他是否是血骑士；但为何选择买田仍无证据。"
    conclusion = ConclusionResult(
        next_action="answer_directly",
        answer=final_answer,
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
        supported_facts=[
            {
                "fact": "血骑士买了田，村民猜出他的身份。",
                "evidence_refs": [
                    {
                        "evidence_id": "story#blood-knight",
                        "quote": "血骑士：我买下了这附近的某块田。村民：您是血骑士？血骑士：过去是。",
                    }
                ],
            }
        ],
    )

    validated = validate_conclusion_grounding(
        question=question,
        hypothesis=hypothesis,
        evidence=evidence,
        conclusion=conclusion,
        max_round_reached=True,
        mode="strict",
    )

    assert validated.answer == final_answer
    assert validated.grounding_warnings


def test_exact_quotes_bypass_legacy_token_overlap_false_rejection() -> None:
    question = "娜塔莉娅怎样解释乌萨斯依靠战争维持运转？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="reasoning",
        entities=["娜塔莉娅", "乌萨斯"],
        keywords=["战争", "运转"],
        expected_answer_type="剧情解释",
        dialogue_context="",
    )
    quote = "乌萨斯之所以能走到今天，是因为战争；只要有战争，乌萨斯就能够运转起来"
    evidence = [
        {
            "document": {
                "id": "story#ursus",
                "clean_text": quote,
                "search_text": quote,
            }
        }
    ]
    conclusion = ConclusionResult(
        next_action="answer_directly",
        answer="娜塔莉娅解释，乌萨斯靠不断发动战争获取资源并投入下一场战争来维持运转。",
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
        supported_facts=[
            {
                "fact": "乌萨斯依靠战争维持运转。",
                "evidence_refs": [{"evidence_id": "story#ursus", "quote": quote}],
            }
        ],
    )

    validated = validate_conclusion_grounding(
        question=question,
        hypothesis=hypothesis,
        evidence=evidence,
        conclusion=conclusion,
        max_round_reached=False,
        mode="strict",
    )

    assert validated.next_action == "answer_directly"
    assert validated.answer == conclusion.answer


def test_quote_validation_tolerates_removed_speaker_labels() -> None:
    question = "村民怎样认出血骑士？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_fact",
        query_type="fact",
        entities=["村民", "血骑士"],
        keywords=["认出"],
        expected_answer_type="事件细节",
        dialogue_context="",
    )
    evidence_text = "卡西米尔村民：这就是买下荒田的骑士老爷。卡西米尔村民：是血骑士。"
    conclusion = ConclusionResult(
        next_action="answer_directly",
        answer="村民认出买下荒田的骑士老爷是血骑士。",
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
        supported_facts=[
            {
                "fact": "村民认出买下荒田的骑士老爷是血骑士。",
                "evidence_refs": [
                    {
                        "evidence_id": "story#village",
                        "quote": "这就是买下荒田的骑士老爷 是血骑士",
                    }
                ],
            }
        ],
    )

    validated = validate_conclusion_grounding(
        question=question,
        hypothesis=hypothesis,
        evidence=[
            {
                "document": {
                    "id": "story#village",
                    "clean_text": evidence_text,
                    "search_text": evidence_text,
                }
            }
        ],
        conclusion=conclusion,
        max_round_reached=True,
        mode="strict",
    )

    assert validated.next_action == "answer_directly"
    assert not any("quote_not_found" in issue for issue in validated.grounding_warnings)
