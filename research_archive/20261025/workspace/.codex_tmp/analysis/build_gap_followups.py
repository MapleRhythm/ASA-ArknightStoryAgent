"""Build gap-conditioned follow-up SFT targets (warm-start for B1+B2 RL).

For each multi-hop query: retrieve round-1 pool, find gold segments still MISSING,
then ask GLM to write a follow-up hypothesis (question+entities+keywords) that
targets ONLY the missing content, given what's already covered. This teaches the
policy to emit gap-targeted follow-ups instead of restating the query.
Reward-validated premise: gap-targeted queries recover 62% of gaps.
"""
import concurrent.futures
import json, os, sys, re, ssl, time, urllib.error, urllib.request
sys.path.insert(0, "${REMOTE_STORE}/exx_grounding_v1/tools")
from retrieval_reward import RewardEngine, split_gold, seg_covered, norm

API_KEY = os.environ["BIGMODEL_API_KEY"]
ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
OPEN = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    urllib.request.HTTPSHandler(
        context=ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
    ),
)
SYS = ("你是检索规划器。给你一个问题、已检索到的证据摘要、以及一段【尚未检索到但回答该问题必需】的证据。"
       "请只针对【缺失证据】写一个后续检索假设，用于定向补检索——不要重复已知信息，聚焦缺失的那一环。"
       '只输出JSON：{"question":"针对缺失环的具体子问题","entities":["缺失环涉及的实体"],"keywords":["定向关键词"],"expected_answer_type":""}')

def glm(question, known, missing):
    user = f"原问题：{question}\n\n已检索到（不要重复）：\n{known[:1500]}\n\n【缺失的必需证据】：\n{missing[:800]}"
    body = {"model":"glm-5.3","temperature":0,"max_tokens":8192,"reasoning_effort":"high",
            "messages":[{"role":"system","content":SYS},{"role":"user","content":user}]}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body,ensure_ascii=False).encode(),
        headers={"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json"}, method="POST")
    for att in range(4):
        try:
            with OPEN.open(req, timeout=180) as r:
                c = json.loads(r.read().decode())["choices"][0]["message"]["content"]
            s=c.find("{"); e=c.rfind("}"); v=json.loads(c[s:e+1])
            if "question" in v and "keywords" in v: return v
        except Exception as ex:
            if att == 3:
                print(f"[glm-fail] {type(ex).__name__}: {str(ex)[:240]}", flush=True)
                return None
            time.sleep(2**att)

eng = RewardEngine(device="cuda")
srcs = ["${REMOTE_STORE}/rank_v8_coverage_241_mirror/ablation_inputs/failure_hard_pool_listwise.jsonl",
        "${REMOTE_STORE}/rank_v8_coverage_241_mirror/ablation_inputs/sample200_listwise.jsonl"]
rows, seen = [], set()
for sp in srcs:
    for l in open(sp):
        r=json.loads(l)
        if r["query"] in seen: continue
        seen.add(r["query"]); rows.append(r)

out_path = "${REMOTE_STORE}/exx_grounding_v1/data/gap_followups_v1_20260904.jsonl"
done = set()
if os.path.exists(out_path):
    for l in open(out_path): done.add(json.loads(l)["query"])
pending = []
n_nogap = 0
for i, r in enumerate(rows):
    q = r["query"]
    if q in done:
        continue
    gold = split_gold(next(c["text"] for c in r["candidates"] if c.get("label")=="positive"))
    prior = eng.retrieve(q, top_k=30)
    pids=[doc_id for doc_id,_ in prior]; ptexts=[text for _,text in prior]
    missing = [s for s in gold if not seg_covered(s, ptexts)]
    if not missing:
        n_nogap += 1
        continue
    known = " / ".join(s for s in gold if seg_covered(s, ptexts))[:1500]
    pending.append((r, gold, pids, ptexts, known, missing[0]))

n_written = len(done); n_recovered = 0
print(f"[prepared] pending={len(pending)} nogap={n_nogap} workers=3", flush=True)
with open(out_path, "a", encoding="utf-8") as f, concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    future_map = {
        pool.submit(glm, r["query"], known, missing): (r, gold, pids, ptexts, known, missing)
        for r, gold, pids, ptexts, known, missing in pending
    }
    for future in concurrent.futures.as_completed(future_map):
        r, gold, pids, ptexts, known, missing = future_map[future]
        fu = future.result()
        if not fu:
            continue
        # Validate the generated follow-up actually helps (reward gate).
        fq = fu.get("question","") + " " + " ".join(fu.get("keywords",[]))
        sc = eng.score_followup(fq, gold, pids, ptexts)
        rec = {"query": r["query"], "query_type": r.get("query_type"), "source_name": r.get("source_name"),
               "known_covered": known, "missing_segment": missing[:400],
               "gap_followup": fu, "reward": sc}
        f.write(json.dumps(rec, ensure_ascii=False)+"\n"); f.flush()
        n_written += 1
        if sc["coverage_gain"] > 0: n_recovered += 1
        if n_written % 20 == 0:
            print(f"[{n_written}] recovered={n_recovered} nogap={n_nogap}", flush=True)
print(f"DONE written={n_written} nogap_skipped={n_nogap} recovered={n_recovered}")
