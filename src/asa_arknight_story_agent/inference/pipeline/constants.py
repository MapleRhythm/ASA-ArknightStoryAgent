from __future__ import annotations


HYPOTHESIS_INTENTS = {
    "plot_fact",
    "plot_reasoning",
    "timeline",
    "character_relation",
    "event_summary",
    "compare",
    "persona_chat",
    "out_of_scope",
}
QUERY_TYPES = {
    "fact",
    "relation",
    "causality",
    "reasoning",
    "reveal",
    "mystery",
    "answerability",
}
RETRIEVAL_ACTIONS = {
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
}
RETRIEVAL_ACTIONS_ORDER = (
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
)
INITIAL_HYPOTHESIS_TASK_TYPE = "user_question_hypothesis_generation"
FOLLOW_UP_HYPOTHESIS_TASK_TYPE = "follow_up_hypothesis_generation"
CONCLUSION_TASK_TYPE = "conclusion_generation"
WEB_CONTEXT_TASK_TYPE = "web_context_retrieval"
MINIRAG_CHAPTER_EXPANSION_TASK_TYPE = "minirag_chapter_expansion_retrieval"
INITIAL_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
    "intent",
    "query_type",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
    "reflect_tokens",
)
FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
    "query_type",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
    "reflect_tokens",
)
CONCLUSION_SCHEMA_FIELDS = (
    "question",
    "next_action",
    "answer",
    "final_answer",
    "reason",
    "supported_facts",
    "inferred_facts",
    "missing_slots",
    "clarification_question",
    "follow_up_hypothesis",
    "reflect_tokens",
)
CONCLUSION_IGNORED_EXTRA_FIELDS = {
    "additional_evidence_needed",
    "clarification_questions",
    "confidence",
    "conflicting_info",
    "current_round",
    "decision",
    "dialogue_context",
    "follow_up_question",
    "new_entities",
    "new_keywords",
    "slot_values",
}
ROLE_LABEL_MAP = {
    "user": "用户",
    "assistant": "助手",
}
PRONOUN_REFERENCES = {
    "她们",
    "他们",
    "它们",
    "她",
    "他",
    "它",
    "这位",
    "那位",
    "这个人",
    "那个人",
}
NOISY_RETRIEVAL_TOKENS = {
    "user",
    "assistant",
    "同伴关系",
    "身份关系",
    "事实问答",
    "综合剧情问答",
}
NOISY_TOKEN_MARKERS = (
    "什么",
    "为何",
    "为什么",
    "怎么",
    "如何",
    "哪里",
    "哪儿",
    "是否",
    "有没有",
    "故事",
)
ENTITY_EXCLUDE_MARKERS = (
    "之间",
    "故事",
    "经历",
    "过往",
    "渊源",
    "关系",
)
PROMPT_DIALOGUE_CONTEXT_MAX_CHARS = 600
PROMPT_HISTORY_MAX_ROUNDS = 2
PROMPT_GENERATION_HISTORY_MAX_CHARS = 1200
PROMPT_RETRIEVAL_HISTORY_MAX_CHARS = 1200
PROMPT_FOLLOW_UP_EVIDENCE_MAX_TOTAL_CHARS = 2600
PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS = 5000
PROMPT_EVIDENCE_MAX_CHARS_PER_DOC = 520
MULTI_QUERY_MERGE_RRF_K = 60
