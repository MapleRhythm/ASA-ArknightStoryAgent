"""GLM claim/evidence annotation pilot. Cache only until canonical sync.

Never writes API credentials, request headers or unfiltered HTTP errors.
No redirects, retries, training, source rewrites, old-label changes or gold.
"""
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "e2e_v53" / "attempt02"
sys.path[:0] = [str(CACHE), str(CACHE.parent)]
import span_protocol_v2 as protocol

ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-5.3"
MAX_TOKENS = 8192
MAX_CALLS = 9


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, dict):
        return {redact(k, secret): redact(v, secret) for k, v in value.items()}
    return value


def messages(row, claim, claim_id):
    packet = protocol.from_dict(row["packet"])
    return [
        {"role": "system", "content": protocol.VERIFY_RULES},
        {"role": "user", "content": json.dumps({
            "packet_id": packet.packet_id, "claim_id": claim_id,
            "question": row["question"], "claim": claim,
            "source": protocol.render_packet(packet),
        }, ensure_ascii=False)},
    ]


def prepare():
    checkpoint = load(CACHE / "checkpoint_v53.json")
    frozen = load(CACHE / "preflight.json")
    expected = next(h for p, h in checkpoint["basis_sha256"].items() if p.endswith("/inputs.json"))
    assert sha(CACHE / "inputs.json") == expected
    for name in ("span_protocol_v2.py", "run_shadow_v2.py"):
        h = next(h for p, h in frozen["sha256"].items() if p.endswith("/" + name))
        assert sha(CACHE / name) == h
    h = next(h for p, h in frozen["sha256"].items() if p.endswith("/span_protocol.py"))
    assert sha(CACHE.parent / "span_protocol.py") == h
    rows = load(CACHE / "inputs.json")
    tasks = []
    for row in rows:
        packet = protocol.from_dict(row["packet"])
        assert packet == protocol.make_packet(row["canonical_kwargs"]["evidence_prompt_text"])
        source = row["source_row"]
        values = [(p["input"]["supported_facts"][0]["fact"], "assistant_authored_contrast")
                  for p in source["single_fact_projections_not_model_answers"]]
        values.append((source["original_parsed_conclusion"]["supported_facts"][
            source["fact_index_zero_based"]]["fact"], "historical_model_output"))
        for claim, origin in values:
            claim_id = f"C{len(tasks) + 1}"
            body = {
                "model": MODEL, "messages": messages(row, claim, claim_id),
                "thinking": {"type": "enabled"}, "reasoning_effort": "high",
                "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False,
            }
            tasks.append({
                "id": claim_id, "family": row["family"], "question": row["question"],
                "claim": claim, "origin_not_gold": origin, "packet": row["packet"], "request_body": body,
                "request_sha256": protocol.digest(json.dumps(body, ensure_ascii=False, sort_keys=True)),
            })
    assert len(tasks) == MAX_CALLS
    # Expected labels and previous model verdicts are excluded from the request.
    for item in tasks:
        request = item["request_body"]
        assert set(json.loads(request["messages"][1]["content"])) == {
            "packet_id", "claim_id", "question", "claim", "source"}
    save_new(HERE / "inputs.json", tasks)
    paths = [Path(__file__), HERE / "README.md", HERE / "test_glm_annotation.py", HERE / "inputs.json",
             CACHE / "inputs.json", CACHE / "checkpoint_v53.json", CACHE / "preflight.json",
             CACHE / "span_protocol_v2.py", CACHE.parent / "span_protocol.py"]
    pre = {
        "version": "glm_claim_entailment_pilot_v54_20260911",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": ENDPOINT, "model": MODEL, "max_calls": MAX_CALLS,
        "max_output_tokens_per_call": MAX_TOKENS,
        "input_rows": len(tasks), "sha256": {str(p): sha(p) for p in paths},
        "canonical_source_repository": "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_HOME}/ASA-ArknightStoryAgent",
        "execution_location": "local_cache_pending_canonical_sync",
        "human_gold": False, "training_ready": False, "prior_labels_sent_to_teacher": False,
        "production_changed": False, "automatic_retry": False,
        "composite_claim_policy": "All clauses must be supported; pilot does not generate atomic subclaims.",
    }
    save_new(HERE / "preflight.json", pre)
    print(json.dumps({"prepared_claims": len(tasks), "preflight_sha256": sha(HERE / "preflight.json"),
                      "teacher_labels_not_human_gold": True, "remote_sync_pending": True}), flush=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def verify():
    pre = load(HERE / "preflight.json")
    assert all(sha(Path(p)) == h for p, h in pre["sha256"].items())
    assert pre["endpoint"] == ENDPOINT and pre["model"] == MODEL
    return pre


def annotation(raw, item):
    # API receives complete messages without the local model's brace prefill.
    value = protocol.strict_object(raw)
    packet = protocol.from_dict(item["packet"])
    protocol.validate_verdict(value, packet, item["id"])
    refs = [dict(span_id=s.span_id, evidence_id=s.evidence_id, start=s.start, end=s.end,
                 block_sha256=s.block_sha256, quote=s.text)
            for s in protocol.select_spans(packet, value["span_ids"], required=False)]
    return {
        "judgment": value, "source_refs": refs, "teacher": MODEL,
        "label_status": "machine_annotation_pending_independent_review",
        "human_gold": False, "training_ready": False,
    }


def safe_http_error(error, secret):
    # Do not persist message/body: providers may echo credentials there.
    body = error.read(65536)
    try:
        data = json.loads(body)
        code = str(data.get("error", {}).get("code", ""))
    except (ValueError, TypeError, AttributeError):
        code = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", code) or secret in code:
        code = None
    return {"error_type": "HTTPError", "http_status": error.code, "provider_error_code": code,
            "response_body_not_saved_for_credential_safety": True}


def run():
    pre = verify()
    assert not (HERE / "attempts.jsonl").exists()
    assert sys.stdin.isatty(), "Use a TTY for non-echoed credential entry"
    # Refuse getpass fallback: it could echo a credential in an unsuitable PTY.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        secret = getpass.getpass("GLM credential (not echoed or stored): ").strip()
    assert secret and not any(c.isspace() for c in secret)
    context = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context), NoRedirect())
    tasks = load(HERE / "inputs.json")
    records, labels = [], []
    with (HERE / "attempts.jsonl").open("x", encoding="utf-8") as journal:
        for item in tasks:
            start = time.monotonic()
            record = {
                "claim_id": item["id"], "request_sha256": item["request_sha256"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "error",
            }
            response = None
            try:
                request = urllib.request.Request(
                    ENDPOINT, data=json.dumps(item["request_body"], ensure_ascii=False).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + secret},
                    method="POST")
                with opener.open(request, timeout=120) as stream:
                    record["http_status"] = stream.status
                    response = json.loads(stream.read())
                # This preserves model content/usage, not request headers.
                record["response"] = redact(response, secret)
                assert response["model"] == MODEL, "returned_model_mismatch"
                assert len(response["choices"]) == 1
                choice = response["choices"][0]
                assert choice["finish_reason"] == "stop", "non_stop_completion_preserved"
                raw = choice["message"]["content"]
                result = annotation(raw, item)
                record.update(status="ok", annotation=result)
                labels.append({"claim_id": item["id"], "claim": item["claim"],
                               "question": item["question"], **result})
            except urllib.error.HTTPError as error:
                record.update(safe_http_error(error, secret))
            except Exception as error:
                record["error_type"] = type(error).__name__
                # Only deterministic, key-free validation errors are retained.
                if response is not None:
                    record["validation_error"] = redact(str(error), secret)[:500]
                else:
                    record["connection_error_details_withheld"] = True
            record["seconds"] = time.monotonic() - start
            record = redact(record, secret)
            text = json.dumps(record, ensure_ascii=False)
            assert secret not in text
            journal.write(text + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            records.append(record)
            print(json.dumps({k: record[k] for k in (
                "claim_id", "status", "http_status", "provider_error_code", "error_type")
                if k in record}, ensure_ascii=False), flush=True)
            # Do not spend on the rest until the first real annotation passes.
            # Any later failure also stops this pilot; never retry implicitly.
            if record["status"] != "ok":
                break
    secret = None
    verify()
    save_new(HERE / "labels_candidate.json", labels)
    summary = {
        "preflight_sha256": sha(HERE / "preflight.json"),
        "attempts": len(records), "successful_annotations": len(labels),
        "not_attempted": MAX_CALLS - len(records),
        "http200_responses": sum(r.get("http_status") == 200 for r in records),
        "status": "pilot_completed_pending_review" if len(labels) == MAX_CALLS else "pilot_stopped_on_failure",
        "credential_persisted": False, "human_gold": False, "training_ready": False,
        "remote_sync_pending": True, "production_changed": False,
    }
    save_new(HERE / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1]]()
