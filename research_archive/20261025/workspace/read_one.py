import json
data = json.load(open("${ASA_ROOT}/soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1_train.json"))
print("total:", len(data))
found = 0
for idx, r in enumerate(data):
    if "conclusion" not in str(r.get("task_type","")):
        continue
    try:
        p = json.loads(r["conversations"][-1]["value"])
    except Exception:
        continue
    if p.get("next_action")=="answer_directly" and p.get("supported_facts"):
        print(f"=== record #{idx} kto_tag={r.get('kto_tag')} ===")
        print("--- HUMAN (prompt) ---")
        print(r["conversations"][0]["value"][:1800])
        print("--- ASSISTANT ---")
        print(json.dumps(p, ensure_ascii=False, indent=1)[:1400])
        found += 1
        if found >= 1:
            break
print("found answer_directly with facts:", found)
