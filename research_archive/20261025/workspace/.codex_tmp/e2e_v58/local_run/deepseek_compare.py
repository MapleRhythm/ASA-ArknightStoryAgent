"""Small DeepSeek-flash paired reader test; credentials arrive on stdin only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import socket
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
BASELINE = Path(os.environ.get(
    "DEEPSEEK_BASELINE_SOURCE",
    "${REMOTE_STORE}/exx_grounding_v1/experiments/asa_v56_eval_20260911",
))
ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
MAX_REQUESTS = 28
CONNECT_TIMEOUT = float(os.environ.get("DEEPSEEK_CONNECT_TIMEOUT", "8"))
READ_TIMEOUT = float(os.environ.get("DEEPSEEK_READ_TIMEOUT", "120"))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def normalize(text):
    import string
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def metrics(prediction, references):
    pred = normalize(prediction).split()
    best = {"normalized_exact_match": 0.0, "token_f1": 0.0, "rouge_l_f1": 0.0}
    for answer in references:
        ref = normalize(answer).split()
        overlap = sum((Counter(pred) & Counter(ref)).values())
        f1 = 2 * overlap / (len(pred) + len(ref)) if pred or ref else 1.0
        previous = [0] * (len(ref) + 1)
        for word in pred:
            current = [0]
            for j, target in enumerate(ref):
                current.append(previous[j] + 1 if word == target else max(previous[j + 1], current[-1]))
            previous = current
        rouge = 2 * previous[-1] / (len(pred) + len(ref)) if pred or ref else 1.0
        best["normalized_exact_match"] = max(best["normalized_exact_match"], float(pred == ref))
        best["token_f1"] = max(best["token_f1"], f1)
        best["rouge_l_f1"] = max(best["rouge_l_f1"], rouge)
    return best


def body(row):
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": row["system"]},
                     {"role": "user", "content": row["user"]}],
        "temperature": 0.0, "max_tokens": 4096, "stream": False,
        "thinking": {"type": "enabled"}, "reasoning_effort": "high",
    }


def prepare():
    assert not (HERE / "preflight.json").exists()
    source = Path(os.environ.get(
        "DEEPSEEK_PILOT_SOURCE",
        "${REMOTE_STORE}/exx_grounding_v1/experiments/asa_v57_glm_pilot_20260911_attempt02",
    ))
    rows = load(source / "selected_rows.json")
    baseline = load(source / "baseline_subset.json")
    assert len(rows) == MAX_REQUESTS and len(baseline) == 56
    assert len({(r["id"], r["condition"]) for r in rows}) == MAX_REQUESTS
    requests = [{"id": r["id"], "condition": r["condition"], "body": body(r),
                 "request_sha256": digest(body(r))} for r in rows]
    save(HERE / "selected_rows.json", rows)
    save(HERE / "requests.json", requests)
    save(HERE / "baseline_subset.json", baseline)
    files = [HERE / p for p in ("deepseek_compare.py", "launch_deepseek.sh",
                                "selected_rows.json", "requests.json", "baseline_subset.json")]
    files += [source / p for p in ("selected_rows.json", "baseline_subset.json")]
    pre = {
        "at_utc": now(), "endpoint": ENDPOINT, "model": MODEL, "max_requests": MAX_REQUESTS,
        "questions": 14, "conditions": ["oracle", "bm25_top3"],
        "input_messages_same_as_4b_and_glm": True,
        "sampling": {"temperature": 0.0, "thinking": "enabled", "reasoning_effort": "high",
                     "max_tokens": 4096, "stream": False},
        "scope": "same 14-question paired pilot as GLM; not full FairytaleQA",
        "automatic_retry": False, "credentials_persisted": False, "production_changed": False,
        "bound_files": {str(p): sha(p) for p in files},
    }
    save(HERE / "preflight.json", pre)
    print(json.dumps({k: v for k, v in pre.items() if k != "bound_files"}), flush=True)


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, dict):
        return {redact(k, secret): redact(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def network_preflight():
    """Check TCP/TLS/API reachability before reading or sending a credential."""
    probe = ENDPOINT.rsplit("/", 1)[0]
    req = urllib.request.Request(probe, method="GET")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        with opener.open(req, timeout=CONNECT_TIMEOUT) as response:
            return {"reachable": True, "http_status": response.status}
    except urllib.error.HTTPError as error:
        # A 401/403/405 proves DNS, TCP, TLS and HTTP routing work.
        return {"reachable": True, "http_status": error.code}
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
        raise RuntimeError(
            f"DeepSeek endpoint unreachable before authentication; "
            f"connect_timeout={CONNECT_TIMEOUT}s; type={type(error).__name__}"
        ) from None


def call(item, secret, opener):
    result = {"id": item["id"], "condition": item["condition"], "request_sha256": item["request_sha256"],
              "status": "error", "at_utc": now()}
    started = time.monotonic()
    try:
        req = urllib.request.Request(ENDPOINT, json.dumps(item["body"], ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + secret},
            method="POST")
        with opener.open(req, timeout=READ_TIMEOUT) as response:
            result["http_status"] = response.status
            raw = response.read(2 * 1024**2 + 1)
        assert len(raw) <= 2 * 1024**2
        payload = json.loads(raw)
        result["response"] = redact(payload, secret)
        assert payload.get("model") == MODEL, "returned_model_mismatch"
        choice = payload["choices"][0]
        result["finish_reason"] = choice.get("finish_reason")
        result["raw_output"] = str((choice.get("message") or {}).get("content") or "")
        assert result["finish_reason"] == "stop" and result["raw_output"].strip()
        result["status"] = "ok"
    except urllib.error.HTTPError as error:
        result.update(http_status=error.code, error_type="HTTPError", error_body_withheld=True)
        try:
            code = str(json.loads(error.read(65536)).get("error", {}).get("code", ""))
        except Exception:
            code = ""
        if re.fullmatch(r"[A-Za-z0-9_-]{1,32}", code):
            result["provider_error_code"] = code
    except Exception as error:
        result.update(error_type=type(error).__name__, error_details_withheld=True)
    result["seconds"] = time.monotonic() - started
    return redact(result, secret)


def verify():
    pre = load(HERE / "preflight.json")
    for path, expected in pre["bound_files"].items():
        assert sha(path) == expected, path
    requests = load(HERE / "requests.json")
    rows = load(HERE / "selected_rows.json")
    assert len(requests) == len(rows) == MAX_REQUESTS
    for item, row in zip(requests, rows, strict=True):
        assert item["body"] == body(row) and digest(item["body"]) == item["request_sha256"]
    return pre, requests, rows


def run():
    assert not (HERE / "attempts.jsonl").exists()
    pre, requests, rows = verify()
    reachability = network_preflight()
    secret = sys.stdin.readline().strip()
    assert secret and not any(c.isspace() for c in secret)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    save(HERE / "start.json", {"at_utc": now(), "preflight_sha256": sha(HERE / "preflight.json"),
                              "credentials_persisted": False, "network_preflight": reachability,
                              "connect_timeout_seconds": CONNECT_TIMEOUT,
                              "read_timeout_seconds": READ_TIMEOUT})
    records = []
    with (HERE / "attempts.jsonl").open("x", encoding="utf-8") as journal:
        for item, row in zip(requests, rows, strict=True):
            journal.write(json.dumps({"event": "request_started", "id": item["id"],
                                      "condition": item["condition"], "at_utc": now(),
                                      "request_sha256": item["request_sha256"]}) + "\n")
            journal.flush()
            result = call(item, secret, opener)
            if result["status"] == "ok":
                result["metrics"] = metrics(result["raw_output"], row["answers"])
            result["event"] = "request_finished"
            journal.write(json.dumps(redact(result, secret), ensure_ascii=False) + "\n")
            journal.flush()
            records.append(result)
            print(json.dumps({k: result[k] for k in ("id", "condition", "status", "http_status", "seconds")
                              if k in result}), flush=True)
            if result["status"] != "ok":
                break
    secret = None
    summary = {"at_utc": now(), "requests_attempted": len(records),
               "successful_answers": sum(r["status"] == "ok" for r in records),
               "unattempted": MAX_REQUESTS - len(records), "complete": len(records) == MAX_REQUESTS,
               "attempts_sha256": sha(HERE / "attempts.jsonl"),
               "preflight_sha256": sha(HERE / "preflight.json"), "production_changed": False}
    save(HERE / "run_summary.json", summary)
    if summary["complete"]:
        score(pre, records, rows)
    print(json.dumps(summary), flush=True)


def score(pre, records, rows):
    baseline = load(HERE / "baseline_subset.json")
    lookup = {(r["id"], r["condition"]): r for r in rows}
    glm = [{"adapter": MODEL, **r} for r in records if r.get("event") == "request_finished"]
    assert len(glm) == MAX_REQUESTS and all(r["status"] == "ok" for r in glm)
    groups = {}
    for adapter in ("parent", "candidate", MODEL):
        for condition in ("oracle", "bm25_top3"):
            subset = [r for r in baseline + glm if r["adapter"] == adapter and r["condition"] == condition]
            assert len(subset) == 14
            groups[f"{adapter}/{condition}"] = {
                "n": len(subset),
                **{m: sum(r["metrics"][m] for r in subset) / len(subset)
                   for m in ("normalized_exact_match", "token_f1", "rouge_l_f1")},
            }
    usage = Counter()
    for r in records:
        for k, v in (r.get("response", {}).get("usage", {}) or {}).items():
            if type(v) is int:
                usage[k] += v
    save(HERE / "comparison.json", {
        "at_utc": now(), "groups": groups, "api_usage_reported": dict(usage),
        "api_seconds": sum(r["seconds"] for r in records), "production_changed": False,
        "scope": pre["scope"], "attempts_sha256": sha(HERE / "attempts.jsonl"),
    })


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "run"}:
        raise SystemExit("use: deepseek_compare.py prepare|run")
    try:
        {"prepare": prepare, "run": run}[sys.argv[1]]()
    except Exception:
        if not (HERE / "failure.json").exists():
            save(HERE / "failure.json", {"at_utc": now(), "error_type": "run_failed",
                                        "details_withheld": True})
        raise
