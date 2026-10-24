import json
from collections import Counter

for name in ["teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_train.json",
             "soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1_train.json"]:
    data = json.load(open(f"${ASA_ROOT}/{name}"))
    print("="*84)
    print(f"=== {name[:50]} ===")
    # answer_directly_no_facts 的细分: answer 为空 vs answer 非空但没 facts
    sub = Counter()
    samples = {"answer_nonempty_no_facts": [], "answer_empty": []}
    for r in data:
        if "conclusion" not in str(r.get("task_type","")): continue
        val = r["conversations"][-1]["value"]
        try: p = json.loads(val)
        except: continue
        na = p.get("next_action","")
        facts = p.get("supported_facts") or []
        ans = str(p.get("answer") or p.get("final_answer") or "").strip()
        if na == "answer_directly" and not facts:
            if ans:
                sub["answer_nonempty_no_facts"] += 1
                if len(samples["answer_nonempty_no_facts"]) < 3:
                    samples["answer_nonempty_no_facts"].append((r["conversations"][0]["value"][:80], ans[:120]))
            else:
                sub["answer_empty"] += 1
    print("answer_directly_no_facts 细分:", dict(sub))
    for q, a in samples["answer_nonempty_no_facts"]:
        print("  Q:", q)
        print("  A:", a)
        print()
