"""校验 chosen 结论的 quote 是否真实出现在证据文本中。
复用 grounding 的 normalize 逻辑，不是靠生成者自觉。"""
import json, re, sys

def normalize(text):
    t = re.sub(r"\s+", "", str(text or ""))
    t = re.sub(r"\[E\d+\]", "", t)
    t = t.replace("...", "").replace("…", "")
    return t

def check_case(case):
    """case: {question, chosen:{answer,supported_facts}, evidence:[{text}]}"""
    pool = normalize("\n".join(e["text"] for e in case["evidence"]))
    issues = []
    facts = case["chosen"].get("supported_facts", [])
    if not facts:
        issues.append("no_supported_facts")
    total_q = 0
    for fi, f in enumerate(facts, 1):
        refs = f.get("evidence_refs", [])
        if not refs:
            issues.append(f"fact{fi}_no_refs")
        for ri, ref in enumerate(refs, 1):
            q = ref.get("quote", "").strip()
            if not q:
                issues.append(f"fact{fi}_ref{ri}_empty")
                continue
            total_q += len(q)
            if len(q) > 80:
                issues.append(f"fact{fi}_ref{ri}_over80({len(q)})")
            if normalize(q) not in pool:
                issues.append(f"fact{fi}_ref{ri}_NOT_FOUND: {q[:40]}")
    if total_q > 400:
        issues.append(f"total_quote_over400({total_q})")
    return issues

if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    ok = bad = 0
    for case in data:
        iss = check_case(case)
        if iss:
            bad += 1
            print(f"[FAIL] {case['question'][:40]}")
            for i in iss: print("   ", i)
        else:
            ok += 1
    print(f"\n=== {ok} OK / {bad} FAIL ===")
