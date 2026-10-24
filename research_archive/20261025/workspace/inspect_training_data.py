import json, re
from collections import Counter

def load(name):
    return json.load(open(f"${ASA_ROOT}/{name}"))

# 非实体标记(时间短语/动作/关系/疑问词等不该当实体的)
NON_ENTITY_MARKERS = ("时","时候","分别","如何","怎么","什么","为什么","为何","帮助","看待","评价",
                      "经历","过程","原因","目的","发生","之后","之前","关系","影响","结果","表现",
                      "决定","改变","遭遇","对待","看待","困境","态度","意图","真相","目的")
# 启发式可疑模式
HEURISTIC_SIGNS = ("JSON-like","启发式","需要补充","无法确认","占位","placeholder","TODO","xxx","???")

def check_hypothesis(r):
    """检查 hypothesis_generation 数据"""
    issues = []
    val = r["conversations"][-1]["value"]
    try:
        p = json.loads(val)
    except:
        # 非纯 JSON,可能是 JSON-like
        if "intent" in val and "entities" in val:
            issues.append("hypothesis_not_pure_json")
        return issues
    ents = p.get("entities") or []
    for e in ents:
        e = str(e)
        # 非实体短语混入
        if any(m in e for m in NON_ENTITY_MARKERS) and len(e) >= 3:
            issues.append(f"nonentity_in_entities:{e}")
        # 过长(实体不应太长)
        if len(e) > 10:
            issues.append(f"toolong_entity:{e[:20]}")
        # 含疑问词
        if any(w in e for w in ("什么","怎么","为什么","吗","呢")):
            issues.append(f"questionword_entity:{e}")
    return issues

def check_conclusion(r):
    """检查 conclusion_generation 数据"""
    issues = []
    val = r["conversations"][-1]["value"]
    try:
        p = json.loads(val)
    except:
        issues.append("conclusion_not_pure_json")
        return issues
    na = p.get("next_action","")
    facts = p.get("supported_facts") or []
    # answer_directly 但没 facts
    if na == "answer_directly" and not facts:
        issues.append("answer_directly_no_facts")
    # facts 里 quote 为空
    for f in facts:
        refs = f.get("evidence_refs") or []
        for ref in refs:
            if not str(ref.get("quote") or "").strip():
                issues.append("empty_quote")
            if not str(ref.get("evidence_id") or "").strip():
                issues.append("empty_evidence_id")
    return issues

for name in ["teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_train.json",
             "soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1_train.json"]:
    data = load(name)
    print("="*84)
    print(f"=== {name[:50]} ({len(data)}条) ===")
    issue_counter = Counter()
    examples = {}
    for r in data:
        tt = r.get("task_type","")
        iss = []
        if "hypothesis" in tt:
            iss = check_hypothesis(r)
        elif "conclusion" in tt:
            iss = check_conclusion(r)
        for i in iss:
            key = i.split(":")[0]
            issue_counter[key] += 1
            if key not in examples:
                examples[key] = (tt, i, r["conversations"][0]["value"][:100])
    print("问题分布:")
    for k, v in issue_counter.most_common():
        print(f"  {k}: {v}")
    print("\n样例:")
    for k, (tt, i, q) in list(examples.items())[:8]:
        print(f"  [{k}] task={tt}")
        print(f"      问题: {i}")
        print(f"      来自: {q[:80]}")
