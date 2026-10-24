"""Bounded GLM reader comparison. No retries, coding endpoint, or production writes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import ssl
import stat
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-5.3"
SALT = "asa-v57-fairytaleqa-pilot-20260911:"
MAX_REQUESTS = 28


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


def select_ids(rows):
    by_type = defaultdict(set)
    for row in rows:
        by_type[row["attribute"]].add(row["id"])
    assert len(by_type) == 7
    ids = []
    for kind in sorted(by_type):
        ordered = sorted(by_type[kind], key=lambda rid: hashlib.sha256((SALT + rid).encode()).hexdigest())
        assert len(ordered) >= 2
        ids.extend(ordered[:2])
    assert len(ids) == len(set(ids)) == 14
    return ids


def body_for(row):
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": row["system"]},
                     {"role": "user", "content": row["user"]}],
        "temperature": 0.0, "top_p": 1.0,
        "max_tokens": 4096, "stream": False,
        "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }


def prepare(baseline):
    baseline = baseline.resolve()
    assert not (HERE / "preflight.json").exists()
    complete = load(baseline / "complete.json")
    results = load(baseline / "results.json")
    old_pre = load(baseline / "preflight.json")
    assert complete["exit_code"] == 0
    assert sha(baseline / "results.json") == complete["results_sha256"]
    assert sha(baseline / "generations.jsonl") == results["generations_sha256"]
    assert sha(baseline / "preflight.json") == results["preflight_sha256"]
    rows_path = baseline / "benchmark/rows.json"
    assert sha(rows_path) == old_pre["bound_files"][str(rows_path)]
    for name in ("run_fairytaleqa.py", "prepare_fairytaleqa.py"):
        assert sha(baseline / name) == old_pre["bound_files"][str(baseline / name)]
    rows = load(rows_path)
    ids = select_ids(rows)
    lookup = {(r["id"], r["condition"]): r for r in rows}
    selected = [lookup[rid, condition] for rid in ids for condition in ("oracle", "bm25_top3")]
    assert len(selected) == MAX_REQUESTS
    baseline_records = [json.loads(line) for line in
                        (baseline / "generations.jsonl").read_text().splitlines()]
    keys = [(r["adapter"], r["id"], r["condition"]) for r in baseline_records]
    assert len(keys) == len(set(keys)) == 4028
    assert all(r["finish_reason"] == "stop" for r in baseline_records)
    subset = [r for r in baseline_records if r["id"] in ids]
    assert len(subset) == 56
    requests = [{"id": r["id"], "condition": r["condition"], "body": body_for(r),
                 "request_sha256": digest(body_for(r))} for r in selected]
    save(HERE / "selected_rows.json", selected)
    save(HERE / "requests.json", requests)
    save(HERE / "baseline_subset.json", subset)
    files = [HERE / p for p in ("glm_compare.py", "test_glm_compare.py", "launch_glm.sh",
                               "glm-5.3.official.md", "selected_rows.json", "requests.json",
                               "baseline_subset.json")]
    files += [baseline / p for p in ("run_fairytaleqa.py", "prepare_fairytaleqa.py",
                                    "preflight.json", "results.json", "generations.jsonl")]
    pre = {
        "at_utc": now(), "baseline": str(baseline), "endpoint": ENDPOINT, "model": MODEL,
        "max_requests": MAX_REQUESTS, "questions": len(ids), "selected_ids": ids,
        "selection": "2 questions per each of 7 attributes, SHA256 salt+id order; ignores predictions",
        "selection_salt": SALT, "max_output_tokens_per_request": 4096,
        "api_reasoning": "enabled/low (GLM-5.3 has no no-thinking mode)",
        "input_messages_identical_to_4b": True,
        "budget_not_equal_to_4b": "4B no-thinking/96 tokens; GLM low-thinking/4096 including reasoning",
        "tokenizers_differ": True, "api_seed_not_set": True,
        "metrics": "unchanged v56 lexical metrics; maximum over answer1 and answer4",
        "scope": "14-question stratified pilot, not full 1007-question or leaderboard score",
        "retrieval_unchanged": True, "production_changed": False, "automatic_retry": False,
        "credential_policy": "Existing documented BIGMODEL_API_KEY configuration; never saved or logged.",
        "bound_files": {str(p): sha(p) for p in files},
    }
    save(HERE / "preflight.json", pre)
    print(json.dumps({k: v for k, v in pre.items() if k != "bound_files"}), flush=True)


def read_credential_file(path):
    """Read only literal BIGMODEL_API_KEY assignment; never execute shell code."""
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "unsafe credential file"
    assert info.st_mode & 0o077 == 0 and info.st_size <= 65536, "unsafe credential permissions"
    found = []
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?BIGMODEL_API_KEY\s*=(.*)$", line)
        if match:
            values = shlex.split(match.group(1), comments=True)
            assert len(values) == 1, "credential must be a literal assignment"
            value = values[0]
            assert value and not any(c.isspace() for c in value), "invalid credential"
            assert "$" not in value and "`" not in value, "shell expansion not allowed"
            found.append(value)
    assert len(found) == 1, "expected one BIGMODEL_API_KEY assignment"
    return found[0]


def redact(obj, secret):
    if isinstance(obj, str):
        return obj.replace(secret, "[REDACTED]")
    if isinstance(obj, dict):
        return {redact(k, secret): redact(v, secret) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, secret) for v in obj]
    return obj


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def call(item, secret, opener):
    record = {"id": item["id"], "condition": item["condition"], "at_utc": now(),
              "request_sha256": item["request_sha256"], "status": "error"}
    started = time.monotonic()
    try:
        request = urllib.request.Request(ENDPOINT, json.dumps(item["body"]).encode(),
                   headers={"Content-Type": "application/json", "Authorization": "Bearer " + secret},
                   method="POST")
        with opener.open(request, timeout=120) as response:
            record["http_status"] = response.status
            raw = response.read(2 * 1024**2 + 1)
            assert len(raw) <= 2 * 1024**2, "response too large"
            payload = json.loads(raw)
        record["response"] = redact(payload, secret)
        assert payload["model"] == MODEL, "returned model differs"
        assert len(payload["choices"]) == 1, "unexpected choices"
        choice = payload["choices"][0]
        record["finish_reason"] = choice["finish_reason"]
        record["raw_output"] = choice["message"].get("content") or ""
        assert isinstance(record["raw_output"], str), "non-text content"
        assert choice["finish_reason"] == "stop", "completion not finished"
        assert record["raw_output"].strip(), "empty answer"
        record["status"] = "ok"
    except urllib.error.HTTPError as error:
        record.update(http_status=error.code, error_type="HTTPError")
        try:
            error_code = str(json.loads(error.read(65536)).get("error", {}).get("code", ""))
        except Exception:
            error_code = ""
        if re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", error_code):
            record["provider_error_code"] = error_code
        record["error_body_withheld"] = True
    except Exception as error:
        record["error_type"] = type(error).__name__
        record["error_message_withheld"] = True
    record["seconds"] = time.monotonic() - started
    return redact(record, secret)


def verify():
    pre = load(HERE / "preflight.json")
    assert pre["endpoint"] == ENDPOINT and pre["model"] == MODEL
    for path, expected in pre["bound_files"].items():
        assert sha(path) == expected, f"changed input:{Path(path).name}"
    items = load(HERE / "requests.json")
    rows = load(HERE / "selected_rows.json")
    assert len(items) == len(rows) == MAX_REQUESTS
    for item, row in zip(items, rows, strict=True):
        assert item["id"] == row["id"] and item["condition"] == row["condition"]
        assert item["body"] == body_for(row) and digest(item["body"]) == item["request_sha256"]
    return pre, items, rows


def run(credential_file):
    assert not (HERE / "start.json").exists(), "refuse rerun; requests may have been billed"
    assert not (HERE / "attempts.jsonl").exists()
    pre, items, rows = verify()
    secret = os.environ.get("BIGMODEL_API_KEY")
    if not secret:
        assert credential_file is not None, "BIGMODEL_API_KEY not configured"
        secret = read_credential_file(credential_file)
    assert secret and not any(c.isspace() for c in secret)
    sys.path.insert(0, pre["baseline"])
    from run_fairytaleqa import metrics
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    save(HERE / "start.json", {"at_utc": now(), "pid": os.getpid(),
                              "preflight_sha256": sha(HERE / "preflight.json"),
                              "credentials_persisted": False})
    records = []
    with (HERE / "attempts.jsonl").open("x", encoding="utf-8") as journal:
        for item, row in zip(items, rows, strict=True):
            # Durable attempt marker precedes transmission to prevent silent duplicates.
            marker = {"event": "request_started", "id": item["id"], "condition": item["condition"],
                      "at_utc": now(), "request_sha256": item["request_sha256"]}
            journal.write(json.dumps(marker) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            record = call(item, secret, opener)
            if record["status"] == "ok":
                record["metrics"] = metrics(record["raw_output"], row["answers"])
            else:
                record["metrics"] = None
            record["event"] = "request_finished"
            text = json.dumps(redact(record, secret), ensure_ascii=False)
            assert secret not in text
            journal.write(text + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            records.append(record)
            print(json.dumps({k: record[k] for k in ("id", "condition", "status", "http_status",
                                                    "provider_error_code", "seconds")
                              if k in record}), flush=True)
            if record["status"] != "ok":
                break
    secret = None
    summary = {"at_utc": now(), "requests_attempted": len(records),
               "successful_answers": sum(r["status"] == "ok" for r in records),
               "unattempted": MAX_REQUESTS - len(records),
               "complete": len(records) == MAX_REQUESTS and all(r["status"] == "ok" for r in records),
               "requests_sha256": sha(HERE / "requests.json"),
               "attempts_sha256": sha(HERE / "attempts.jsonl"),
               "preflight_sha256": sha(HERE / "preflight.json"),
               "production_changed": False, "credentials_persisted": False}
    save(HERE / "run_summary.json", summary)
    if summary["complete"]:
        score(pre, records)
    print(json.dumps(summary), flush=True)


def score(pre, records):
    baseline = load(HERE / "baseline_subset.json")
    groups = {}
    paired_keys = {(r["id"], r["condition"]) for r in records}
    assert len(paired_keys) == MAX_REQUESTS
    glm = [{"adapter": MODEL, **r} for r in records]
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
    for record in records:
        for key, value in record["response"].get("usage", {}).items():
            if type(value) is int:
                usage[key] += value
    report = {"at_utc": now(), "pilot_questions": 14, "groups": groups,
              "api_usage_reported": dict(usage), "api_seconds": sum(r["seconds"] for r in records),
              "reasoning_budgets_not_matched": pre["budget_not_equal_to_4b"],
              "scope": pre["scope"], "generations_sha256": sha(HERE / "attempts.jsonl"),
              "production_changed": False}
    save(HERE / "comparison.json", report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "run"])
    p.add_argument("--baseline", type=Path)
    p.add_argument("--credential-env-file", type=Path)
    args = p.parse_args()
    try:
        if args.action == "prepare":
            prepare(args.baseline)
        else:
            run(args.credential_env_file)
    except Exception as error:
        # Do not expose credentials through traceback locals or remote error text.
        record = {"at_utc": now(), "error_type": type(error).__name__,
                  "error_details_withheld": True, "action": args.action}
        if not (HERE / "failure.json").exists():
            save(HERE / "failure.json", record)
        print(json.dumps(record), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
