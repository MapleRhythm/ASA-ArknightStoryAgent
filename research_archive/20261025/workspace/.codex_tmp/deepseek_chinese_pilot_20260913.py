"""Six-question DeepSeek API pilot; credential is read from hidden stdin only."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
REFS = {
    "古米在下午训练时表现出的问题具体是什么？坚雷教官如何看待这个问题？": [
        "古米无法连续快速挥动手臂，挥两下就会停顿；坚雷认为这是会在战场上致命的坏习惯，需要矫正，并发现她第三下会下意识用尽全力。"
    ],
    "凛冬在学生自治团中扮演什么角色？她是怎样成为领袖的？": [
        "凛冬是真正组建学生自治团的人，也是自治团的领头人；她过去在邻近学校很有名，被认为天生适合当领袖。"
    ],
    "真理的本名是什么？她和古米、凛冬分别是什么学校的关系？": [
        "真理本名安娜·莫罗佐娃；真理和古米、娜塔莉娅是同一所高中，凛冬和烈夏属于另一所学校。"
    ],
    "薇卡对真理来说是什么人？为什么真理至今还保留着她的东西？": [
        "薇卡是真理加入学生自治团前最好的朋友；真理保留薇卡的娃娃，因为她从未忘记薇卡。"
    ],
    "梅在拿到无人机后发生了什么意外？真理如何处理了这个情况？": [
        "梅拿到无人机时从窗台上摔落并擦伤左手；真理发现后帮她清洁伤口、喷药，并让她再找医疗干员仔细处理。"
    ],
    "泡普卡为什么会突然出现在罗德岛？": [
        "泡普卡原先在伐木场；安赛尔和凯尔希路过看到她后，第二天凯尔希把她买下，带到罗德岛。"
    ],
}


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def redact(obj, secret):
    if isinstance(obj, str):
        return obj.replace(secret, "[REDACTED]")
    if isinstance(obj, dict):
        return {redact(k, secret): redact(v, secret) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, secret) for v in obj]
    return obj


def normalize(text):
    text = re.sub(r"\s+", "", text.lower())
    return text


def metrics(pred, refs):
    # Chinese semantic review is primary; these are only rough character overlap.
    p = normalize(pred)
    best = {"char_f1": 0.0, "exact": 0.0}
    for ref in refs:
        r = normalize(ref)
        overlap = sum(min(p.count(ch), r.count(ch)) for ch in set(r))
        f1 = 2 * overlap / (len(p) + len(r)) if p or r else 1.0
        best["char_f1"] = max(best["char_f1"], f1)
        best["exact"] = max(best["exact"], float(p == r))
    return best


def main():
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    for row in rows:
        row["reference_answers"] = REFS[row["question"]]
    probe = urllib.request.Request(ENDPOINT.rsplit("/", 1)[0], method="GET")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        opener.open(probe, timeout=8).close()
    except urllib.error.HTTPError as e:
        assert e.code in (401, 403, 404, 405)
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as e:
        raise SystemExit(f"endpoint unreachable: {type(e).__name__}")
    secret = sys.stdin.readline().strip()
    assert secret and not any(c.isspace() for c in secret)
    results = []
    for row in rows:
        evidence = []
        for item in row.get("evidence", [])[:10]:
            evidence.append(f"[{item.get('id')}]\n{item.get('clean_text', '')}")
        user = (
            "请只依据下面提供的剧情证据回答问题。回答应直接、完整、简洁；"
            "如果证据确实不足才回答“证据不足”。\n\n"
            f"证据：\n{chr(10).join(evidence)}\n\n"
            f"问题：{row['question']}\n答案："
        )
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "你是中文剧情问答助手。"},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        started = time.monotonic()
        rec = {
            "question": row["question"],
            "request_sha256": digest(body),
            "status": "error",
        }
        try:
            req = urllib.request.Request(
                ENDPOINT,
                json.dumps(body, ensure_ascii=False).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + secret,
                },
                method="POST",
            )
            with opener.open(req, timeout=120) as response:
                payload = json.loads(response.read(2 * 1024 * 1024 + 1))
            answer = str(payload["choices"][0]["message"].get("content") or "")
            rec.update(
                status="ok",
                http_status=200,
                model=payload.get("model"),
                answer=answer,
                finish_reason=payload["choices"][0].get("finish_reason"),
                metrics=metrics(answer, row.get("reference_answers", [])),
            )
        except urllib.error.HTTPError as e:
            rec.update(http_status=e.code, error_type="HTTPError")
        except Exception as e:
            rec.update(error_type=type(e).__name__)
        rec["seconds"] = round(time.monotonic() - started, 3)
        results.append(redact(rec, secret))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    output.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n",
        encoding="utf-8",
    )
    secret = None


if __name__ == "__main__":
    main()
