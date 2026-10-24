"""Frozen slot-coverage rubric on every direct fixed79 answer, with exact-input reuse."""
from collections import Counter
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_answer_slots_v1 as slots


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_key(case):
    return hashlib.sha256(json.dumps({
        "question": case["question"], "shown_answer": case["shown_answer"],
    }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def request(case):
    return {"model": "glm-5.3", "temperature": 0, "max_tokens": 16384, "messages": slots.messages(case)}


def main():
    source = HERE / "canonical_fixed79_validation_v1.json"
    previous_path = HERE / "answer_slot_coverage7_v1.json"
    source_rows = json.loads(source.read_text())["results"]
    rows = [r for r in source_rows if r["status"] == "ok" and r["grounded"]["next_action"] == "answer_directly"]
    assert len(source_rows) == 79 and len(rows) == 39
    groups = {}
    for row in rows:
        case = {"question": row["question"], "shown_answer": row["grounded"]["answer"]}
        key = input_key(case)
        if key not in groups:
            groups[key] = {**case, "id": "slot_input_" + key, "parent_ids": []}
        groups[key]["parent_ids"].append(row["id"])

    previous = json.loads(previous_path.read_text())
    assert previous["version"] == slots.VERSION and previous["rubric"] == slots.RUBRIC
    assert previous["driver_sha256"] == sha(HERE / "probe_answer_slots_v1.py")
    old_results = {r["id"]: r for r in previous["results"]}
    reusable = {}
    for old_case in previous["cases"]:
        result = old_results[old_case["id"]]
        if result["status"] != "ok":
            continue
        assert result["request_without_credentials"] == request(old_case)
        assert result["response_model"] == "glm-5.3"
        slots.validate(result["judgement"], old_case)
        reusable[input_key(old_case)] = result
    cached, pending = {}, []
    for key, case in groups.items():
        if key in reusable:
            cached[key] = {
                **reusable[key], "id": case["id"], "reused": True,
                "reused_source_id": reusable[key]["id"], "reused_source_sha256": sha(previous_path),
            }
        else:
            pending.append(case)

    output = HERE / "answer_slot_coverage_direct39_v1.json"
    journal = output.with_suffix(".progress.jsonl")
    preflight = HERE / "answer_slot_coverage_direct39_preflight_v1.json"
    if any(path.exists() for path in (output, journal, preflight)):
        raise ValueError("Existing artifact; never duplicate or overwrite")
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    if pending and not key:
        raise ValueError("Provider key unavailable")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": slots.VERSION, "evaluation_only": True, "training_ready": False,
        "source_analysis_sha256": sha(source), "driver_sha256": sha(Path(__file__)),
        "rubric_script_sha256": sha(HERE / "probe_answer_slots_v1.py"),
        "reused_source_sha256": sha(previous_path), "states": 39, "unique_inputs": len(groups),
        "reused_successful_inputs": len(cached), "new_api_requests": len(pending),
        "groups": list(groups.values()),
        "limitations": [
            "All 39 direct states from fixed79, not independent questions or new E2E generations.",
            "Only exact question+shown-answer inputs with identical rubric/model/request are reused.",
            "No visible evidence, old label, reference answer, action, or source-review notes is sent.",
            "Coverage is not factual correctness. Errors remain in the 39-state denominator.",
            "No runtime gate, training labels, weights, default model or production answers are changed.",
        ],
    }
    with preflight.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: manifest[k] for k in ("states", "unique_inputs", "reused_successful_inputs", "new_api_requests")}), flush=True)
    slots.transport.messages = slots.messages
    slots.transport.validate = slots.validate
    results = list(cached.values())
    with journal.open("x", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(slots.transport.call, case, key) for case in pending]
            for future in concurrent.futures.as_completed(futures):
                result = {**future.result(), "reused": False}
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({"completed_unique": len(results), "total_unique": len(groups),
                                  "id": result["id"], "status": result["status"], "error": result.get("error")}), flush=True)
    by_group_id = {r["id"]: r for r in results}
    assert len(by_group_id) == len(groups)
    per_state = {}
    for case in groups.values():
        group_result = by_group_id[case["id"]]
        for parent in case["parent_ids"]:
            per_state[parent] = {
                **group_result, "id": parent, "judgment_group_id": case["id"],
                "source_case_question": case["question"], "source_case_shown_answer": case["shown_answer"],
            }
    assert set(per_state) == {r["id"] for r in rows}
    report = {
        **manifest, "preflight_sha256": sha(preflight),
        "results": [per_state[r["id"]] for r in rows], "unique_results": results,
        "state_statuses": dict(Counter(r["status"] for r in per_state.values())),
        "state_coverage_counts": dict(Counter((r.get("judgement") or {}).get("coverage", "audit_error") for r in per_state.values())),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("state_statuses", "state_coverage_counts")}))


if __name__ == "__main__":
    main()
