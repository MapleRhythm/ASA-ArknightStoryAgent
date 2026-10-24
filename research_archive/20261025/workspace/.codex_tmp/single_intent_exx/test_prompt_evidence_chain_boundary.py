from __future__ import annotations

from types import SimpleNamespace

from goldenglow.inference.cpu_pipeline import (
    CPUInferencePipeline,
    ConclusionResult,
    HypothesisDocument,
    _grounding_evidence_pool,
    _resolve_referential_question,
    build_answer_prompt,
    normalize_hypothesis_payload,
    render_evidence_blocks,
    render_short_evidence_brief,
    render_minirag_hints_for_prompt,
    select_prompt_evidence,
    validate_conclusion_grounding,
)
from goldenglow.inference.cpu_pipeline import build_conclusion_prompt


class _Reranker:
    def score(self, *, query: str, documents: list[str], batch_size: int) -> list[float]:
        del query, batch_size
        return [float(index) for index, _ in enumerate(documents)]


def test_model_hypothesis_is_not_polluted_by_heuristic_chinese_fragments() -> None:
    question = "哈洛德为何开始厌倦战争，到了晚年又为什么仍在四处奔走？"
    hypothesis = normalize_hypothesis_payload(
        {
            "entities": ["哈洛德"],
            "keywords": ["厌倦战争", "晚年", "四处奔走", "原因"],
            "query_type": "causality",
            "expected_answer_type": "原因与动机",
        },
        question=question,
        dialogue_context="",
    )

    assert hypothesis.entities == ["哈洛德"]
    assert hypothesis.keywords[:4] == ["厌倦战争", "晚年", "四处奔走", "原因"]
    assert not {"何开始厌", "倦战争", "到了晚年又"} & set(
        hypothesis.entities + hypothesis.keywords
    )


def test_referential_question_preserves_original_pronouns() -> None:
    questions = [
        "爱国者声称自己毫无复仇之心，那他为什么仍决定与罗德岛战斗？",
        "苏茜为什么买下绿意火花，后来又为什么把它改成理发店？",
        "古米独自难过时，凛冬和真理怎样照顾了她？",
    ]

    for question in questions:
        assert _resolve_referential_question(
            f"  {question}  ",
            ["爱国者", "苏茜", "古米", "罗德岛"],
        ) == question


def test_full_evidence_rendering_keeps_every_selected_document_intact() -> None:
    evidence = [
        {
            "doc_index": index,
            "document": {
                "id": f"story#full-{index:02d}",
                "clean_text": f"第{index}条证据开始。" + (f"完整内容{index}。" * 80) + f"第{index}条证据结束。",
                "search_text": "",
            },
        }
        for index in range(1, 13)
    ]

    brief = render_short_evidence_brief(
        evidence,
        max_chars_per_doc=None,
        max_total_chars=None,
    )
    blocks = render_evidence_blocks(
        evidence,
        max_chars_per_doc=None,
        max_total_chars=None,
    )

    for index, item in enumerate(evidence, start=1):
        clean_text = item["document"]["clean_text"]
        assert clean_text in brief
        assert clean_text in blocks
        assert f"story#full-{index:02d}" in brief
    assert brief.count("条证据结束。") == 12
    assert "…" not in brief


def test_full_evidence_pipeline_selects_twelve_and_bypasses_crag_stripping() -> None:
    retriever = SimpleNamespace(reranker=_Reranker())
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=object(),
        prompt_evidence_top_k=12,
        prompt_evidence_max_chars_per_doc=0,
        prompt_conclusion_evidence_max_total_chars=0,
        prompt_evidence_require_full_documents=True,
        enable_crag_refinement=True,
        crag_refine_top_sentences=1,
        crag_refine_max_sentences=8,
    )
    question = "哈洛德晚年为何奔走？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["哈洛德"],
        keywords=["晚年", "奔走"],
        expected_answer_type="原因",
        dialogue_context="",
    )
    evidence = [
        {
            "doc_index": index,
            "document": {
                "id": f"story#candidate-{index:02d}",
                "clean_text": (
                    f"候选{index}第一句。"
                    + (chr(0x4E00 + index) * (index + 2))
                    + f"这是编号{index}的独立剧情细节。候选{index}完整结尾。"
                ),
                "search_text": "",
            },
        }
        for index in range(1, 14)
    ]

    selected = pipeline.prepare_prompt_evidence(question, hypothesis, evidence)
    prompt = build_answer_prompt(
        question,
        hypothesis,
        evidence,
        prompt_evidence_top_k=pipeline.prompt_evidence_top_k,
        prompt_evidence=selected,
        evidence_max_chars_per_doc=pipeline.prompt_evidence_max_chars_per_doc,
        evidence_max_total_chars=pipeline.prompt_conclusion_evidence_max_total_chars,
    )

    assert len(selected) == 12
    assert pipeline.prompt_evidence_max_chars_per_doc is None
    assert pipeline.prompt_conclusion_evidence_max_total_chars is None
    assert all("crag_refinement" not in item for item in selected)
    for index in range(1, 13):
        assert f"候选{index}完整结尾。" in prompt
    assert "story#candidate-13" not in prompt


def test_positive_total_budget_never_slices_an_evidence_document() -> None:
    first_text = "甲" * 80 + "第一条完整结束"
    second_text = "乙" * 80 + "第二条完整结束"
    evidence = [
        {"document": {"id": "story#first", "clean_text": first_text, "search_text": ""}},
        {"document": {"id": "story#second", "clean_text": second_text, "search_text": ""}},
    ]

    rendered = render_short_evidence_brief(
        evidence,
        max_chars_per_doc=None,
        max_total_chars=len(first_text) + 30,
    )

    assert first_text in rendered
    assert "第一条完整结束" in rendered
    assert "story#second" not in rendered
    assert "…" not in rendered


def test_conclusion_prompt_forbids_alias_expansion_of_generic_referents() -> None:
    hypothesis = HypothesisDocument(
        question="乔迪为何认出这不是自己的梦？",
        intent="plot_reasoning",
        query_type="causality",
        entities=["乔迪", "Ishar-mla"],
        keywords=["梦境", "回忆"],
        expected_answer_type="剧情解释",
        dialogue_context="",
    )

    prompt = build_conclusion_prompt(
        hypothesis.question,
        hypothesis,
        [],
        [],
        1,
        1,
        1,
        prompt_mode="minimal",
    )

    assert "不得给巨物/那个人/它们等泛称添加括号别名或自行绑定具体类别" in prompt


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


def test_compound_question_pins_direct_evidence_for_each_subclaim() -> None:
    question = "血骑士退役后为何选择在乡间买田生活，村民最初又怎样认出他？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["血骑士", "村民"],
        keywords=["选择", "买田", "认出", "红盔甲"],
        expected_answer_type="因果与事件细节",
        dialogue_context="",
    )
    decoy_texts = [
        "阿米娅在罗德岛整理航行日志。",
        "博士与凯尔希讨论医疗方案。",
        "陈在龙门近卫局查看报告。",
        "能天使给企鹅物流清点货物。",
        "银灰返回谢拉格处理家族事务。",
        "塞雷娅在实验室检查防护设备。",
        "推进之王带队穿过伦蒂尼姆街道。",
        "煌完成任务后回到宿舍休息。",
        "年准备拍摄一部新的动作电影。",
        "夕在画卷里留下了一座山城。",
        "令独自在酒馆回忆旧日旅途。",
        "伊内丝观察战场上的源石尘埃。",
        "号角要求士兵重新检查补给。",
    ]
    evidence = [
        {
            "doc_index": index,
            "document": {
                "id": f"story#decoy-{index:02d}",
                "clean_text": text,
                "search_text": text,
            },
        }
        for index, text in enumerate(decoy_texts, start=1)
    ]
    recognition = {
        "doc_index": 14,
        "document": {
            "id": "story#recognition",
            "clean_text": "村民说，您这一身红盔甲和健硕身姿，您该不会是血骑士狄开俄波利斯？",
            "search_text": "村民说，您这一身红盔甲和健硕身姿，您该不会是血骑士狄开俄波利斯？",
        },
    }
    motive = {
        "doc_index": 15,
        "document": {
            "id": "story#motive",
            "clean_text": "血骑士说矿石病不好治，所以我才选了那块离村庄最远的地，避免别人靠近。",
            "search_text": "血骑士说矿石病不好治，所以我才选了那块离村庄最远的地，避免别人靠近。",
        },
    }

    selected = select_prompt_evidence(
        question,
        hypothesis,
        [*evidence, recognition, motive],
        prompt_evidence_top_k=10,
    )

    selected_ids = [item["document"]["id"] for item in selected]
    assert selected_ids[:8] == [f"story#decoy-{index:02d}" for index in range(1, 9)]
    assert selected_ids[-2:] == ["story#recognition", "story#motive"]
    assert selected[-2]["prompt_subclaim_pins"] == ["村民最初又怎样认出他"]
    assert selected[-1]["prompt_subclaim_pins"] == ["血骑士退役后为何选择在乡间买田生活"]


def test_single_clause_question_does_not_change_finalized_order() -> None:
    question = "乔迪为何判断自己看到的并非普通梦境，而是Ishar-mla的回忆？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["乔迪", "Ishar-mla"],
        keywords=["普通梦境", "回忆"],
        expected_answer_type="剧情解释",
        dialogue_context="",
    )
    evidence = [
        {
            "doc_index": index,
            "document": {
                "id": f"story#ordered-{index}",
                "clean_text": text,
                "search_text": text,
            },
        }
        for index, text in enumerate(
            [
                "乔迪说这不是自己的梦境。",
                "这是Ishar-mla的回忆。",
                "无关的海边背景。",
            ],
            start=1,
        )
    ]

    selected = select_prompt_evidence(
        question,
        hypothesis,
        evidence,
        prompt_evidence_top_k=2,
    )

    assert [item["document"]["id"] for item in selected] == ["story#ordered-1", "story#ordered-2"]
    assert all("prompt_subclaim_pins" not in item for item in selected)


def test_compound_selection_annotates_support_already_inside_top_k() -> None:
    question = "凯尔希为什么没有让Mon3tr攻击内卫，她又如何评价内卫作为人类的意义？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["凯尔希", "Mon3tr", "内卫"],
        keywords=["攻击", "评价", "人类", "意义"],
        expected_answer_type="原因与评价",
        dialogue_context="",
    )
    support = {
        "doc_index": 1,
        "document": {
            "id": "story#both-subclaims",
            "clean_text": (
                "凯尔希说，如果让Mon3tr攻击，漆黑尖刺就会贯穿她的胸膛。"
                "她对内卫仍有一丝认同：在对抗邪魔时，内卫仍是人类伟岸的壁垒，"
                "没有人能剥夺他们生而为人的荣耀。"
            ),
            "search_text": "",
        },
    }
    evidence = [
        support,
        {
            "doc_index": 2,
            "document": {
                "id": "story#decoy-1",
                "clean_text": "凯尔希与海蒂讨论庄园的修缮工作。",
                "search_text": "凯尔希与海蒂讨论庄园的修缮工作。",
            },
        },
        {
            "doc_index": 3,
            "document": {
                "id": "story#decoy-2",
                "clean_text": "博士在罗德岛查看当天的任务报告。",
                "search_text": "博士在罗德岛查看当天的任务报告。",
            },
        },
    ]

    selected = select_prompt_evidence(
        question,
        hypothesis,
        evidence,
        prompt_evidence_top_k=2,
    )

    assert [item["document"]["id"] for item in selected] == ["story#both-subclaims", "story#decoy-1"]
    assert selected[0]["prompt_subclaim_pins"] == [
        "凯尔希为什么没有让Mon3tr攻击内卫",
        "她又如何评价内卫作为人类的意义",
    ]


def test_crag_preserves_direct_sentence_for_pinned_subclaim() -> None:
    retriever = SimpleNamespace(reranker=_Reranker())
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=object(),
        enable_crag_refinement=True,
        crag_refine_top_sentences=1,
        crag_refine_max_sentences=16,
    )
    question = "血骑士为何选择买田，村民又怎样认出他？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["血骑士", "村民"],
        keywords=["选择", "买田", "认出"],
        expected_answer_type="因果与事件细节",
        dialogue_context="",
    )
    item = {
        "doc_index": 1,
        "document": {
            "id": "story#recognition",
            "clean_text": (
                "血骑士买下了附近的田。村民前来问候。天气很晴朗。"
                "村民说，您这一身红盔甲，还有健硕的身姿，您该不会是血骑士？"
                "血骑士回答过去是。众人随后去了商店。故事在夜里结束。"
            ),
            "search_text": "",
        },
        "prompt_subclaim_pins": ["村民又怎样认出他"],
    }

    refined = pipeline.refine_evidence_strips(question, hypothesis, [item])

    assert "一身红盔甲" in refined[0]["document"]["clean_text"]
    assert refined[0]["crag_refinement"]["subclaim_sentence_indices"]


def test_crag_keeps_separated_detail_lines_for_same_subclaim() -> None:
    retriever = SimpleNamespace(reranker=_Reranker())
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=object(),
        enable_crag_refinement=True,
        crag_refine_top_sentences=1,
        crag_refine_max_sentences=20,
    )
    question = "凯尔希为什么阻止Mon3tr攻击，她又如何评价内卫作为人类的意义？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["凯尔希", "Mon3tr", "内卫"],
        keywords=["攻击", "评价", "人类", "意义"],
        expected_answer_type="原因与评价",
        dialogue_context="",
    )
    item = {
        "doc_index": 1,
        "document": {
            "id": "story#kaltsit",
            "clean_text": (
                "凯尔希说，如果Mon3tr攻击，尖刺就会贯穿胸膛。"
                "内卫询问她为何以人类相称。"
                "凯尔希说仍有一丝认同。"
                "在对抗邪魔的瞬间，内卫仍是人类伟岸的壁垒之一。"
                "没有任何人能剥夺他们生而为人的荣耀。"
                "至少在被幻象欺骗之前。"
                "随后双方结束了对话。"
                "夜色覆盖庄园。"
            ),
            "search_text": "",
        },
        "prompt_subclaim_pins": [
            "凯尔希为什么阻止Mon3tr攻击",
            "她又如何评价内卫作为人类的意义",
        ],
    }

    refined = pipeline.refine_evidence_strips(question, hypothesis, [item])
    refined_text = refined[0]["document"]["clean_text"]

    assert "尖刺就会贯穿胸膛" in refined_text
    assert "人类伟岸的壁垒" in refined_text
    assert "生而为人的荣耀" in refined_text
    assert len(refined[0]["crag_refinement"]["subclaim_sentence_indices"]) >= 2


def test_subclaim_pinning_never_uses_ranking_chain_as_direct_support() -> None:
    question = "血骑士为何选择买田，村民又怎样认出他？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["血骑士", "村民"],
        keywords=["选择", "买田", "认出"],
        expected_answer_type="因果与事件细节",
        dialogue_context="",
    )
    decoy_texts = [
        "阿米娅在罗德岛整理航行日志。",
        "博士与凯尔希讨论医疗方案。",
        "陈在龙门近卫局查看报告。",
        "能天使给企鹅物流清点货物。",
        "银灰返回谢拉格处理家族事务。",
        "塞雷娅在实验室检查防护设备。",
        "推进之王带队穿过伦蒂尼姆街道。",
        "煌完成任务后回到宿舍休息。",
        "年准备拍摄一部新的动作电影。",
        "夕在画卷里留下了一座山城。",
        "令独自在酒馆回忆旧日旅途。",
    ]
    evidence = [
        {
            "doc_index": index,
            "document": {
                "id": f"story#direct-{index:02d}",
                "clean_text": text,
                "search_text": text,
            },
        }
        for index, text in enumerate(decoy_texts, start=1)
    ]
    evidence.append(
        {
            "doc_index": 12,
            "document": {
                "id": "story#chain-only",
                "clean_text": "这段直接文本与问题无关。",
                "search_text": "这段直接文本与问题无关。",
            },
            "evidence_chain_text": "血骑士因为矿石病选择买田，村民根据红盔甲认出他。",
        }
    )

    selected = select_prompt_evidence(
        question,
        hypothesis,
        evidence,
        prompt_evidence_top_k=10,
    )

    assert "story#chain-only" not in [item["document"]["id"] for item in selected]


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


def test_strict_grounding_removes_unbound_alias_for_generic_referent() -> None:
    question = "乔迪为何判断这不是普通梦境？"
    hypothesis = HypothesisDocument(
        question=question,
        intent="plot_reasoning",
        query_type="causality",
        entities=["乔迪"],
        keywords=["梦境", "回忆"],
        expected_answer_type="剧情解释",
        dialogue_context="",
    )
    quote = "不，那个人不是自己 这不是自己的梦境，这是——Ishar-mla的回忆"
    conclusion = ConclusionResult(
        next_action="answer_directly",
        answer="乔迪看到巨物（海嗣）的场景，并意识到那个人不是自己。",
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
        supported_facts=[
            {
                "fact": "乔迪看到巨物（海嗣）的场景，并意识到那个人不是自己。",
                "evidence_refs": [{"evidence_id": "story#jordi", "quote": quote}],
            }
        ],
    )

    validated = validate_conclusion_grounding(
        question=question,
        hypothesis=hypothesis,
        evidence=[
            {
                "document": {
                    "id": "story#jordi",
                    "clean_text": "海嗣会做梦吗？路的尽头，巨物将触肢贴在舱壁上。" + quote,
                    "search_text": "",
                }
            }
        ],
        conclusion=conclusion,
        max_round_reached=True,
        mode="strict",
    )

    assert "巨物（海嗣）" not in validated.answer
    assert "乔迪看到巨物的场景" in validated.answer
    assert "巨物（海嗣）" not in validated.supported_facts[0]["fact"]
    assert "removed_unbound_generic_alias:巨物(海嗣)" in validated.grounding_warnings


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
