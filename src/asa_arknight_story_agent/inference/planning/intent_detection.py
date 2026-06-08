from __future__ import annotations

from asa_arknight_story_agent.inference.common.lexicon import STORY_HINT_WORDS


def detect_intent(question: str) -> tuple[str, str]:
    if any(token in question for token in ("是什么", "本质", "来历")) and any(
        token in question for token in ("危机", "祸", "患", "威胁", "为什么", "为何")
    ):
        return "plot_reasoning", "概念定义/危机原因"
    if any(token in question for token in STORY_HINT_WORDS):
        return "event_summary", "共同经历"
    if any(token in question for token in ("关系", "什么关系", "关联")):
        return "character_relation", "身份关系"
    if any(token in question for token in ("时间线", "先后", "之前", "之后", "何时", "什么时候")):
        return "timeline", "时间线"
    if any(token in question for token in ("对比", "区别", "不同", "相比")):
        return "compare", "对比分析"
    if any(token in question for token in ("总结", "概括", "发生了什么", "讲了什么")):
        return "event_summary", "剧情总结"
    if any(token in question for token in ("为什么", "为何", "原因", "动机", "目的")):
        return "plot_reasoning", "原因/动机"
    if any(token in question for token in ("怎么", "如何", "经过", "发生了什么", "流程")):
        return "plot_reasoning", "过程解释"
    if any(token in question for token in ("谁", "哪里", "哪儿", "何时", "什么时候", "什么", "是否", "有没有")):
        return "plot_fact", "事实问答"
    return "plot_fact", "综合剧情问答"


def infer_query_type(question: str, intent: str, expected_answer_type: str) -> str:
    if intent == "character_relation" or any(token in expected_answer_type for token in ("身份关系", "关系")):
        return "relation"
    if any(token in question for token in ("阴谋", "真相", "秘密", "识破", "揭穿", "曝光", "暴露", "幕后", "主使", "黑幕", "骗局", "诡计")):
        return "reveal"
    if any(token in question for token in ("谜", "怎么回事", "究竟", "到底")):
        return "mystery"
    if any(token in expected_answer_type for token in ("概念定义/危机原因", "answerability")):
        return "answerability"
    if intent == "plot_reasoning" or any(token in expected_answer_type for token in ("原因", "动机", "过程", "解释")):
        return "causality" if any(token in question for token in ("为什么", "为何", "原因", "导致", "造成")) else "reasoning"
    if intent in {"plot_fact", "timeline", "compare"}:
        return "fact"
    if any(token in expected_answer_type for token in ("事实", "时间线", "对比")):
        return "fact"
    return "reasoning"
