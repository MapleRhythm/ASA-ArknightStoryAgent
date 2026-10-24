import json, sys

name = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
data = json.load(open(f"${ASA_ROOT}/{name}"))

for idx in range(start, min(end, len(data))):
    r = data[idx]
    tt = str(r.get("task_type", ""))
    if "conclusion" not in tt and "hypothesis" not in tt:
        continue
    human = r["conversations"][0]["value"]
    asst = r["conversations"][-1]["value"]
    # 提取 question
    q = ""
    for line in human.splitlines():
        if line.startswith("question:") or line.startswith("用户问题:"):
            q = line.split(":", 1)[1].strip()[:60]
            break
    print(f"=== #{idx} [{tt[:20]}] kto={r.get('kto_tag')} ===")
    print(f"Q: {q}")
    # hypothesis: 打印 entities/keywords
    if "hypothesis" in tt:
        try:
            p = json.loads(asst)
            print(f"  entities: {p.get('entities')}")
            print(f"  keywords: {p.get('keywords')}")
            print(f"  intent: {p.get('intent')} | query_type: {p.get('query_type')}")
        except Exception:
            print(f"  [RAW 非JSON] {asst[:200]}")
    else:
        # conclusion: 打印 action + answer + facts quote
        try:
            p = json.loads(asst)
            print(f"  action: {p.get('next_action')}")
            ans = str(p.get('answer') or p.get('final_answer') or '')
            print(f"  answer: {ans[:150]}")
            for f in (p.get('supported_facts') or [])[:3]:
                fq = [x.get('quote','')[:40] for x in (f.get('evidence_refs') or [])]
                print(f"    fact: {str(f.get('fact'))[:70]} | quotes={fq}")
        except Exception:
            print(f"  [RAW 非JSON] {asst[:200]}")
    print()
