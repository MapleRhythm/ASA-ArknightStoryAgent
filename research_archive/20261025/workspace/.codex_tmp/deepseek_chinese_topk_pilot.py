"""Bounded evidence-count pilot; preserves the previous request format."""
import argparse
import getpass
import hashlib
import importlib.util
import json
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.request


PREVIOUS_SCRIPT = Path(__file__).with_name("deepseek_chinese_pilot_20260913.py")
spec = importlib.util.spec_from_file_location("previous_chinese_pilot", PREVIOUS_SCRIPT)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def build_body(row, top_k):
    if top_k < 1 or top_k > len(row["evidence"]):
        raise ValueError("top_k exceeds the available evidence")
    evidence = [
        f"[{item.get('id')}]\n{item.get('clean_text', '')}"
        for item in row["evidence"][:top_k]
    ]
    user = (
        "请只依据下面提供的剧情证据回答问题。回答应直接、完整、简洁；"
        "如果证据确实不足才回答“证据不足”。\n\n"
        f"证据：\n{chr(10).join(evidence)}\n\n"
        f"问题：{row['question']}\n答案："
    )
    return {
        "model": previous.MODEL,
        "messages": [
            {"role": "system", "content": "你是中文剧情问答助手。"},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "stream": False,
        "thinking": {"type": "disabled"},
    }


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()


def validate_baseline(rows, baseline):
    if len(rows) != 6 or len(baseline) != 6:
        raise ValueError("This pilot must contain exactly the original six questions")
    for row, old in zip(rows, baseline):
        if row["question"] != old["question"]:
            raise ValueError("Question order differs from baseline")
        if previous.digest(build_body(row, 10)) != old["request_sha256"]:
            raise ValueError("Reconstructed top-10 request does not match baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    rows, baseline = load_rows(args.source), load_rows(args.baseline)
    validate_baseline(rows, baseline)
    bodies = [build_body(row, args.top_k) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.output_dir / name for name in ("manifest.json", "requests.jsonl", "results.jsonl")]
    if any(path.exists() for path in targets):
        raise SystemExit("Refusing to overwrite existing pilot artifacts")

    manifest = {
        "endpoint": previous.ENDPOINT,
        "requested_model": previous.MODEL,
        "top_k": args.top_k,
        "questions": len(rows),
        "baseline_request_hashes_verified": len(rows),
        "changed_request_field": "messages[1].content: evidence count only",
        "evidence_field": "clean_text",
        "includes_evidence_chain_text": False,
        "timeout_seconds": 120,
        "automatic_retries": 0,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "pilot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "previous_script_sha256": hashlib.sha256(PREVIOUS_SCRIPT.read_bytes()).hexdigest(),
    }
    with targets[0].open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with targets[1].open("x", encoding="utf-8") as handle:
        for row, body in zip(rows, bodies):
            write_json_line(handle, {
                "question": row["question"],
                "evidence_ids": [item["id"] for item in row["evidence"][:args.top_k]],
                "request_sha256": previous.digest(body),
                "body": body,
            })

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    probe = urllib.request.Request(previous.ENDPOINT.rsplit("/", 1)[0], method="GET")
    try:
        opener.open(probe, timeout=8).close()
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 403, 404, 405):
            raise SystemExit(f"Endpoint preflight HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"Endpoint preflight failed: {type(exc).__name__}") from None

    secret = getpass.getpass("DeepSeek API key (hidden): ").strip()
    if not secret or any(char.isspace() for char in secret):
        raise SystemExit("Invalid credential input")
    with targets[2].open("x", encoding="utf-8") as handle:
        for index, (row, body) in enumerate(zip(rows, bodies), 1):
            started = time.monotonic()
            rec = {
                "question": row["question"],
                "top_k": args.top_k,
                "request_sha256": previous.digest(body),
                "status": "error",
            }
            print(f"Request {index}/{len(rows)} started", flush=True)
            try:
                request = urllib.request.Request(
                    previous.ENDPOINT,
                    json.dumps(body, ensure_ascii=False).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + secret,
                    },
                    method="POST",
                )
                with opener.open(request, timeout=120) as response:
                    payload = json.loads(response.read(2 * 1024 * 1024 + 1))
                choice = payload["choices"][0]
                answer = str(choice["message"].get("content") or "")
                rec.update(
                    status="ok", http_status=200, model=payload.get("model"),
                    response_id=payload.get("id"), created=payload.get("created"),
                    answer=answer, finish_reason=choice.get("finish_reason"),
                    usage=payload.get("usage"),
                    metrics=previous.metrics(answer, previous.REFS[row["question"]]),
                )
            except urllib.error.HTTPError as exc:
                rec.update(http_status=exc.code, error_type="HTTPError")
            except Exception as exc:
                rec.update(error_type=type(exc).__name__)
            rec["seconds"] = round(time.monotonic() - started, 3)
            rec = previous.redact(rec, secret)
            write_json_line(handle, rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            # Do not repeat requests when credentials, credit, or rate limits fail.
            if rec.get("http_status") in (401, 402, 403, 429):
                print("Stopping pilot after authorization/billing/rate-limit failure", flush=True)
                break
    secret = None


if __name__ == "__main__":
    main()
