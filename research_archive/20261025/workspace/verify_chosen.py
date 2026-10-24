import json, sys
sys.path.insert(0, "${ASA_ROOT}")
from chosen_data import CHOSEN
from quote_checker import check_case

cases = json.load(open("${ASA_ROOT}/abstain_cases.json"))
# 按 question 匹配 evidence
by_q = {c["question"]: c["evidence"] for c in cases}

ok = bad = 0
for ch in CHOSEN:
    ev = by_q.get(ch["question"])
    if ev is None:
        print(f"[NO EVIDENCE] {ch['question'][:40]}"); bad += 1; continue
    case = {"question": ch["question"], "chosen": {"supported_facts": ch["supported_facts"]}, "evidence": ev}
    iss = check_case(case)
    if iss:
        bad += 1
        print(f"[FAIL] {ch['question'][:40]}")
        for i in iss: print("   ", i)
    else:
        ok += 1
        print(f"[OK] {ch['question'][:40]}")
print(f"\n=== {ok} OK / {bad} FAIL ===")
