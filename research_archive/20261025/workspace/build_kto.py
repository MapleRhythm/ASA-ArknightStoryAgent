"""把手工 chosen + 模型原弃权输出，组装成 KTO sharegpt 格式。
chosen: kto_tag=True (answer_directly 结论)
rejected: kto_tag=False (原 retrieve_more/abstain 输出)
prompt 复刻 conclusion_generation 的 minimal prompt user 内容。"""
import json, sys
sys.path.insert(0, "${ASA_ROOT}")
from chosen_data import CHOSEN

ABSTAIN = json.load(open("${ASA_ROOT}/abstain_cases.json"))
by_q = {c["question"]: c for c in ABSTAIN}

SYSTEM = "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。"

def build_user_prompt(case):
    hyp = case.get("hypothesis") or {}
    q = case["question"]
    # 渲染 evidence_brief（与 minimal prompt 一致的形态）
    lines = []
    for i, e in enumerate(case["evidence"][:10], 1):
        t = " ".join(e["text"].split())[:520]
        lines.append(f"{i}. {e['id']}: {t}")
    brief = "\n".join(lines)
    return "\n".join([
        "task: conclusion_generation",
        f"question: {q}",
        "hypothesis: " + json.dumps(hyp, ensure_ascii=False),
        "round: 1/2",
        "evidence_brief:",
        brief,
        "minirag_hints: ",
        "output_schema: grounded_action_v1",
        "action_set: answer_directly,retrieve_more,abstain",
        'answer_directly: {"next_action":"answer_directly","supported_facts":[{"fact":"","evidence_refs":[{"evidence_id":"","quote":""}]}],"inferred_facts":[],"final_answer":""}',
        'retrieve_more: {"next_action":"retrieve_more","follow_up_hypothesis":{"question":"","query_type":"","entities":[],"keywords":[],"expected_answer_type":"","dialogue_context":""}}',
        'abstain: {"next_action":"abstain","final_answer":"现有证据不足以确认。"}',
        "rules: JSON only；只能使用 evidence_brief 中的证据；单条 quote 必须从 evidence_brief 原文精确复制。",
    ])

def conclusion_json(ch):
    return json.dumps({
        "next_action": "answer_directly",
        "supported_facts": ch["supported_facts"],
        "inferred_facts": [],
        "final_answer": ch["answer"],
    }, ensure_ascii=False, indent=2)

out = []
for ch in CHOSEN:
    case = by_q.get(ch["question"])
    if case is None:
        print("skip (no evidence):", ch["question"][:40]); continue
    user = build_user_prompt(case)
    # chosen
    out.append({
        "id": f"abs_fix_{abs(hash(ch['question']))%10**8}-chosen",
        "task_type": "conclusion_generation",
        "bucket": "answer_directly_balance_v1",
        "system": SYSTEM,
        "tools": [],
        "kto_tag": True,
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": conclusion_json(ch)},
        ],
        "meta": {"source": "manual_teacher_claude", "question": ch["question"]},
    })
    # rejected: 模型原弃权输出
    rejected_val = json.dumps({
        "next_action": "abstain", "final_answer": "现有证据不足以确认。"
    }, ensure_ascii=False)
    out.append({
        "id": f"abs_fix_{abs(hash(ch['question']))%10**8}-rejected",
        "task_type": "conclusion_generation",
        "bucket": "answer_directly_balance_v1",
        "system": SYSTEM,
        "tools": [],
        "kto_tag": False,
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": rejected_val},
        ],
        "meta": {"source": "model_abstain_replay", "question": ch["question"]},
    })

with open("${ASA_ROOT}/kto_answer_directly_balance_v1.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {len(out)} KTO records ({len(out)//2} pairs) -> kto_answer_directly_balance_v1.json")
