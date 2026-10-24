import json
data = json.load(open("${ASA_ROOT}/soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1_train.json"))
print("total:", len(data))
# 逐条读前几条 conclusion 类
count = 0
for idx, r in enumerate(data):
    tt = str(r.get("task_type",""))
    if "conclusion" not in tt:
        continue
    count += 1
    if count > 4:
        break
    print("="*84)
    print(f"[#{idx}] task={tt} kto_tag={r.get('kto_tag')}")
    print("--- HUMAN ---")
    print(r["conversations"][0]["value"][:900])
    print("--- ASSISTANT ---")
    print(r["conversations"][-1]["value"][:900])
