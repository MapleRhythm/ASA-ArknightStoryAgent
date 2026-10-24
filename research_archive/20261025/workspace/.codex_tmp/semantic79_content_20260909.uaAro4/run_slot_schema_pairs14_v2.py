"""Fresh v1 retry control versus one-clause clarification on all seven errors."""
from collections import Counter
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_answer_slots_v1 as old
import probe_answer_slots_v2 as new

CONDITIONS = {"retry_v1": old, "schema_clarified_v2": new}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def messages(case):
    return CONDITIONS[case["condition"]].messages(case)


def main():
    source = HERE / "answer_slot_coverage_direct39_v1.json"
    original = json.loads(source.read_text())
    assert original["states"] == 39 and len(original["results"]) == 39
    errors = [r for r in original["results"] if r["status"] == "error"]
    assert len(errors) == 7
    cases, pairs = [], []
    for index, row in enumerate(errors):
        pair = []
        order = ["retry_v1", "schema_clarified_v2"]
        if index % 2:
            order.reverse()
        for condition in order:
            case = {
                "id": row["id"] + "::" + condition,
                "parent_id": row["id"], "condition": condition,
                "question": row["source_case_question"], "shown_answer": row["source_case_shown_answer"],
            }
            pair.append(case)
            cases.append(case)
        assert messages(pair[0])[1] == messages(pair[1])[1]
        pairs.append(pair)
    assert len(cases) == 14 and len({c["id"] for c in cases}) == 14
    output = HERE / "answer_slot_schema_pairs14_v2.json"
    preflight = HERE / "answer_slot_schema_pairs14_preflight_v2.json"
    journal = HERE / "answer_slot_schema_pairs14_v2.progress.jsonl"
    assert not any(p.exists() for p in (output, preflight, journal))
    suites = ET.parse(HERE / "answer_slot_schema_tests_v2.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 12
    assert all(all(int(s.attrib.get(k, 0)) == 0 for k in ("errors", "failures", "skipped")) for s in suites)
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    assert key, "Provider credential unavailable"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "answer_slot_schema_pairs14_v2", "evaluation_only": True, "training_ready": False,
        "source_sha256": sha(source), "source_states": 39, "source_failed_states": 7,
        "driver_sha256": sha(Path(__file__)),
        "rubric_sha256": {name: sha(HERE / f"probe_answer_slots_{version}.py") for name, version in (("retry_v1", "v1"), ("schema_clarified_v2", "v2"))},
        "transport_sha256": sha(HERE.parent / "probe_claim_entailment_v1.py"),
        "cases": cases, "old_clause": new.OLD_CLAUSE, "new_clause": new.NEW_CLAUSE,
        "protocol": [
            "All seven failed historical states, no cherry-picking only schema-missing failures.",
            "One fresh request per condition, sequential within each pair; two pairs at most concurrently.",
            "Four v1-first pairs, three v2-first pairs. No retries beyond these predeclared requests.",
            "Only one rubric clause changes; validator and output version remain identical.",
            "Only question and unchanged shown answer are sent; no source, desired answer or old judgement.",
            "Provider sampling is temperature zero but determinism/identical backend seeds are not assumed.",
            "Keep failures in each seven-state denominator; preserve original39 and do not overwrite its labels.",
            "No runtime gate, model update, training-label promotion or factual-accuracy claim.",
        ],
    }
    with preflight.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    old.transport.messages = messages
    old.transport.validate = old.validate
    results = []
    lock = threading.Lock()
    with journal.open("x", encoding="utf-8") as stream:
        def run_pair(pair):
            for case in pair:
                result = old.transport.call(case, key)
                result.update(parent_id=case["parent_id"], condition=case["condition"])
                with lock:
                    results.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(json.dumps({"completed": len(results), "total": 14, "id": result["id"], "status": result["status"], "error": result.get("error")}), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run_pair, pairs))
    assert sha(source) == manifest["source_sha256"]
    by_id = {r["id"]: r for r in results}
    assert len(by_id) == 14 and set(by_id) == {c["id"] for c in cases}
    for case in cases:
        expected = {"model": "glm-5.3", "temperature": 0, "max_tokens": 16384, "messages": messages(case)}
        assert by_id[case["id"]]["request_without_credentials"] == expected
    summary = {}
    for condition in CONDITIONS:
        group = [r for r in results if r["condition"] == condition]
        assert len(group) == 7
        summary[condition] = {
            "states": 7, "statuses": dict(Counter(r["status"] for r in group)),
            "coverage": dict(Counter((r.get("judgement") or {}).get("coverage", "audit_error") for r in group)),
        }
    report = {**manifest, "preflight_sha256": sha(preflight), "results": [by_id[c["id"]] for c in cases], "summary": summary}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
