import json
from collections import Counter

data = json.load(open("${ASA_ROOT}/teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_train.json"))
print("SFT 主集 total:", len(data))
print("task_type:", dict(Counter(r.get("task_type","?") for r in data)))
print("bucket:", dict(Counter(r.get("bucket","?") for r in data)))
# 看一条 canon_qa 和 conclusion 的结构差异
for tt in ["canon_qa", "conclusion_generation", "user_question_hypothesis_generation"]:
    for r in data:
        if r.get("task_type")==tt:
            print("="*70)
            print(f"[{tt}] 结构:")
            print("  human:", r["conversations"][0]["value"][:200])
            print("  asst:", r["conversations"][-1]["value"][:300])
            break
