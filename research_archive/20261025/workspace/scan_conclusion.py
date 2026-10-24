import json, re, sys

def norm(t): return re.sub(r"\s+","",str(t or ""))

def has_evidence_in_prompt(human):
    # prompt 里有没有"当前证据:"/"evidence_brief"实际内容
    m = re.search(r"(当前证据[:：]|evidence_brief[:：])\s*\n(.+)", human, re.DOTALL)
    if not m: return False, ""
    ev = m.group(2)
    # 证据非空判定
    ev_clean = ev.strip()
    if not ev_clean or ev_clean.startswith(("（没有","(没有","<empty>","无")):
        return False, ""
    return True, ev_clean[:400]

def answer_tokens_overreach(human, ans):
    """答案里的实体token是否大量不在证据里(脑补迹象)"""
    ev_ok, ev = has_evidence_in_prompt(human)
    if not ev_ok:
        return None  # 无证据, 无法判断 overreach
    ev_n = norm(ev)
    # 答案里的中文词(3字+)
    toks = re.findall(r"[一-龥]{3,}", ans)
    if not toks: return 0.0
    miss = sum(1 for t in toks if t not in ev_n)
    return miss/len(toks)

name = sys.argv[1]
tt_filter = sys.argv[2] if len(sys.argv)>2 else "conclusion_generation"
data = json.load(open(f"${ASA_ROOT}/{name}"))
count=0
for idx, r in enumerate(data):
    if r.get("task_type") != tt_filter: continue
    human = r["conversations"][0]["value"]
    asst = r["conversations"][-1]["value"]
    q = ""
    for line in human.splitlines():
        if line.startswith(("用户问题:","question:")): q = line.split(":",1)[1].strip()[:50]; break
    try: p = json.loads(asst)
    except:
        print(f"#{idx} [非JSON] {q}")
        continue
    na = p.get("next_action","")
    ans = str(p.get("answer") or p.get("final_answer") or "")
    facts = p.get("supported_facts") or []
    ev_ok, ev = has_evidence_in_prompt(human)
    over = answer_tokens_overreach(human, ans) if ans else None
    flag = ""
    if na=="answer_directly":
        if not ev_ok: flag="⚠️无证据却详答"
        elif not facts: flag="⚠️无facts"
        elif over is not None and over>0.5: flag=f"⚠️脑补{over:.0%}"
    print(f"#{idx} [{na}]{'ev' if ev_ok else 'NOEV'} facts={len(facts)} {flag} | {q}")
    count+=1
    if count>=int(sys.argv[3] if len(sys.argv)>3 else 50): break
