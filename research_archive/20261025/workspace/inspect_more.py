import json, re
from collections import Counter

def norm(t):
    return re.sub(r"\s+", "", str(t or ""))

for name in ["teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_train.json",
             "soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1_train.json"]:
    data = json.load(open(f"${ASA_ROOT}/{name}"))
    print("="*84)
    print(f"=== {name[:55]} ({len(data)}条) ===")
    issues = Counter()
    quote_fabricated = []   # quote 不在 evidence_brief 里
    eid_invalid = []        # evidence_id 不在证据编号集合
    empty_answer_kto = []   # kto chosen 但 answer 空
    retrieve_loop = []      # retrieve_more 但没 follow_up
    for r in data:
        if "conclusion" not in str(r.get("task_type","")): continue
        human = r["conversations"][0]["value"]
        val = r["conversations"][-1]["value"]
        try: p = json.loads(val)
        except: continue
        na = p.get("next_action","")
        facts = p.get("supported_facts") or []
        # 提取 human 里的 evidence_brief 文本
        m = re.search(r"evidence_brief[::]\s*\n?(.*?)(?:\nminirag_hints|\noutput_schema|\n当前证据|\Z)", human, re.DOTALL)
        ev_brief = m.group(1) if m else ""
        ev_pool = norm(ev_brief)
        ev_eids = set(re.findall(r"\[(E\d+)\]", ev_brief))
        for f in facts:
            for ref in (f.get("evidence_refs") or []):
                q = str(ref.get("quote") or "").strip()
                eid = str(ref.get("evidence_id") or "").strip()
                if q and ev_pool and norm(q) not in ev_pool:
                    issues["quote_not_in_brief"] += 1
                    if len(quote_fabricated) < 3:
                        quote_fabricated.append((q[:40], name[:20]))
                if eid and ev_eids and eid not in ev_eids:
                    issues["eid_not_in_brief"] += 1
                    if len(eid_invalid) < 3:
                        eid_invalid.append(eid)
        # KTO chosen(kto_tag=True) 但 answer 空
        if r.get("kto_tag") is True and na in ("answer_directly","abstain") and not str(p.get("answer") or p.get("final_answer") or "").strip():
            issues["chosen_empty_answer"] += 1
        # retrieve_more 但没 follow_up_hypothesis
        if na == "retrieve_more" and not p.get("follow_up_hypothesis"):
            issues["retrieve_more_no_followup"] += 1
    print("问题分布:")
    for k,v in issues.most_common():
        print(f"  {k}: {v}")
    if quote_fabricated:
        print("  quote_not_in_brief 样例:", quote_fabricated[:2])
    if eid_invalid:
        print("  eid_not_in_brief 样例:", eid_invalid[:3])
