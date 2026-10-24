"""Record source-reviewed issues separately from immutable model-judge outputs."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run = Path(__file__).resolve().parent
fixed_path = run / "fixed79_configured_lora_ctx12288/validation_after_v2.json"
strict_path = run / "configured_lora_ctx12288/validation_after.json"
sources = {"fixed79": fixed_path, "strict9": strict_path}
lookups = {name: {r["id"]: r for r in json.loads(path.read_text())["results"]}
           for name, path in sources.items()}
specifications = [
    {
        "dataset": "fixed79",
        "id": "466b0df26a5d22ba__teacher_state_grounded_action_sft::current_quote_fixed79",
        "review": "unsupported",
        "source_fragment": "你的推理是没有错，但是，真正的凶手不可能就这样穿着带血的衣物出现在他人面前",
        "answer_fragment": "真理质疑梅缇丝将‘没有权力给别人定罪’作为推理依据的合理性",
        "note": "逐读E1-E10，原文是对血衣证据的疑问，未给出真理对侦探定罪权力的态度。"
                "答案把题干前提强加给原文。两次模型judge也判unsupported，但其unsupported_span"
                "把答案中的弯单引号改成ASCII单引号，导致严格schema校验失败。保留原审核失败，不偷偷放宽精确绑定。",
    },
    {
        "dataset": "fixed79",
        "id": "relabel-25610af6b165944933207d23::current_quote_fixed79",
        "review": "no_answer_with_sufficient_evidence",
        "source_fragment": "谢谢你照顾我的母亲，费尔南小姐",
        "answer_fragment": "现有检索证据不足以给出可校验的回答",
        "note": "可见原文将跋涉寻找者与娜斯提的母亲绑定，所寻者身份可回答，模型却弃答。"
                "两次judge的no_answer与充分性判断一致，但把拒答措辞列入unsupported_spans，"
                "违反本rubric中no_answer必须为空列表的schema。该条仍保留为原审核失败，本复核单独记录。",
    },
    {
        "dataset": "strict9",
        "id": "42dcaacc6867aefabc75eb5b::after::runtime_raw",
        "review": "confirmed_identity_error_and_corrected_review_claim",
        "source_fragment": "我要习武练剑，我要自己成为剑客",
        "answer_fragment": "渡魂剑’郑愁",
        "note": "原文确实支持女子想自己成为剑客，所以不能把这一愿望列作幻觉。"
                "确认的错误在于把剑客郑愁的姓名归给女子，以及用主仆/共生为关系定性。"
                "该条引用词法校验放行，并不代表人物关系正确。",
    },
    {
        "dataset": "strict9",
        "id": "9af639a51c9a8d7bf0cfb860::after::runtime_raw",
        "review": "interpretation_sensitive_model_complete",
        "source_fragment": "保护了朋友和家人的时候才会感觉到满足",
        "answer_fragment": "迷迭香认为战场能满足自己",
        "note": "judge判完整正确，但答案省略满足感的保护对象与条件。"
                "核心对比受支持，泛化句存在误读风险；不将此争议悄悄改成人工gold，也不声称4/9已全部人工确认。",
    },
    {
        "dataset": "strict9",
        "id": "60cb10813e52f206ff5c105e::after::runtime_raw",
        "review": "partial_with_disputed_speculative_clause",
        "source_fragment": "你的神志已经到极限了，别再动手了",
        "answer_fragment": "曾经守护或与之相关的存在",
        "note": "judge把该析取说法视作可接受弱推断，判supported_partial。"
                "复核保留其缺乏直接支持的疑点；可以确定的是答案未充分解释为什么不愿再杀徒弟。"
                "不能用或、可能之类词语自动豁免主体和因果的证据要求。",
    },
    {
        "dataset": "strict9",
        "id": "dd12b4f52b9911c60a70a9dd::after::runtime_raw",
        "review": "global_support_does_not_imply_bound_citation_entailment",
        "source_fragment": "民众乐于看见哥伦比亚对于科学的支持",
        "answer_fragment": "作为政治工具",
        "note": "整份可见证据支持政治形象/竞选动机，judge对最终答案判完整正确。"
                "但生成的政治动机事实绑定的部分E块仅包含公开参观和称赞，不能独自支持真实动机。"
                "因此全上下文答案支持与逐条引用语义蕴含是两个指标，不能混用。",
    },
]
results = []
for spec in specifications:
    row = lookups[spec["dataset"]][spec["id"]]
    if spec["source_fragment"] not in row["evidence_brief"]:
        raise ValueError(f'proof fragment not in visible evidence: {spec["id"]}')
    if spec["answer_fragment"] not in row["grounded"]["answer"]:
        raise ValueError(f'answer fragment not in shown answer: {spec["id"]}')
    results.append({
        **spec, "question": row["question"], "shown_action": row["grounded"]["next_action"],
        "shown_answer": row["grounded"]["answer"],
        "evidence_sha256": hashlib.sha256(row["evidence_brief"].encode()).hexdigest(),
        "parsed_supported_facts": row["parsed"].get("supported_facts", []),
        "reviewer": "coding_agent_source_review_not_human", "training_ready": False,
    })
report = {
    "version": "canonical_source_review_notes_v1", "evaluation_only": True, "training_ready": False,
    "source_analyses": {name: {"path": str(path), "sha256": sha(path)} for name, path in sources.items()},
    "script_sha256": sha(Path(__file__)), "reviews": results,
    "limitations": [
        "Separate source review, not rewritten machine-judge artifacts or human gold.",
        "Two schema-error reviews do not change the 19/79 provisional complete-answer yield.",
        "Ambiguous interpretation and citation-binding issues remain visible, not silently resolved.",
    ],
}
with (run / "source_review_notes_v1.json").open("x") as f:
    f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reviewed_issues": len(results), "all_source_and_answer_fragments_verified": True}))
