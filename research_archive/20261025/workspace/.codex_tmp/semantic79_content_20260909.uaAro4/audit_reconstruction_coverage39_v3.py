"""Blind coverage comparison of parsed versus actually shown direct answers."""
from collections import Counter
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_answer_slots_v1 as owner
import probe_answer_slots_v3 as rubric


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(analysis):
    rows = [r for r in analysis["results"] if r["status"] == "ok" and r["grounded"]["next_action"] == "answer_directly"]
    assert len({r["id"] for r in rows}) == len(rows)
    groups, pairs = {}, []
    for row in rows:
        pair = {"parent_id": row["id"], "question": row["question"], "changed": row["parsed"]["answer"] != row["grounded"]["answer"]}
        for side, answer in (("parsed", row["parsed"]["answer"]), ("shown", row["grounded"]["answer"])):
            payload = {"question": row["question"], "shown_answer": answer}
            assert isinstance(answer, str) and answer.strip()
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            group_id = "coverage_input_" + digest
            if group_id not in groups:
                groups[group_id] = {"id": group_id, **payload, "members": []}
            groups[group_id]["members"].append({"parent_id": row["id"], "side": side})
            pair[side + "_group_id"] = group_id
        pairs.append(pair)
    return rows, list(groups.values()), pairs


def main():
    source = HERE / "canonical_fixed79_validation_v1.json"
    analysis = json.loads(source.read_text())
    rows, cases, pairs = prepare(analysis)
    assert len(analysis["results"]) == 79 and len(rows) == 39
    assert sum(p["changed"] for p in pairs) == 34
    output = HERE / "reconstruction_coverage39_v3.json"
    preflight = HERE / "reconstruction_coverage39_preflight_v3.json"
    journal = HERE / "reconstruction_coverage39_v3.progress.jsonl"
    assert not any(p.exists() for p in (output, preflight, journal))
    suites = ET.parse(HERE / "reconstruction_coverage_tests_v3.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 5
    assert all(all(int(s.attrib.get(k, 0)) == 0 for k in ("errors", "failures", "skipped")) for s in suites)
    scope_path = HERE / "slot_cause_scope40_v3.json"
    scope = json.loads(scope_path.read_text())
    assert scope["summary"]["cause_scope_v3"]["cases"] == 20
    # This is a protocol smoke check, not certification of the evaluator.
    assert scope["summary"]["cause_scope_v3"]["statuses"] == {"ok": 20}
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    assert key, "Provider credential unavailable"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "reconstruction_coverage39_v3", "evaluation_only": True, "training_ready": False,
        "source_sha256": sha(source), "driver_sha256": sha(Path(__file__)),
        "rubric_sha256": sha(HERE / "probe_answer_slots_v3.py"),
        "rubric_dependency_hashes": {name: sha(HERE / name) for name in ("probe_answer_slots_v1.py", "probe_answer_slots_v2.py")},
        "transport_sha256": sha(HERE.parent / "probe_claim_entailment_v1.py"),
        "cause_scope_smoke_sha256": sha(scope_path),
        "all_source_states": 79, "direct_states": 39, "changed_answers": 34,
        "unchanged_answers": 5, "unique_provider_inputs": len(cases),
        "cases": cases, "pairs": pairs,
        "protocol": [
            "All 39 shown direct states from fixed79, not selected only for visibly degraded answers.",
            "Compare the saved parsed/pre-validation answer with the actual shown answer; neither is generated anew or rewritten here.",
            "Source parse/grounding failures and non-direct states are out of this comparison, not erased from the parent79 benchmark.",
            "Only question+candidate answer is sent; no evidence, previous judge, source review, side, grounding warnings or expected label.",
            "Exact identical question+answer inputs are evaluated once and reused within this run, including five unchanged pairs.",
            "One request per unique input, maximum two concurrent, no retry; all errors retained.",
            "Version3 coverage rubric passed a synthetic protocol smoke, not human correctness certification.",
            "Coverage differences do not establish factual improvement; raw answers may add unsupported content, and shown facts may also be wrong.",
            "No production rewrite bypass, model/config change, label replacement, training or new retrieval.",
        ],
    }
    with preflight.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: manifest[k] for k in ("all_source_states", "direct_states", "changed_answers", "unchanged_answers", "unique_provider_inputs")}), flush=True)
    owner.transport.messages = rubric.messages
    owner.transport.validate = rubric.validate
    results = []
    # Deterministic hash ordering avoids knowingly putting every pre-validation
    # answer first; no treatment information is sent to the judge.
    ordered = sorted(cases, key=lambda c: c["id"])
    with journal.open("x", encoding="utf-8") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(owner.transport.call, case, key) for case in ordered]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({"completed": len(results), "total": len(cases), "id": result["id"], "status": result["status"], "error": result.get("error")}), flush=True)
    by_id = {r["id"]: r for r in results}
    assert len(by_id) == len(cases) and set(by_id) == {c["id"] for c in cases}
    assert sha(source) == manifest["source_sha256"]
    paired = []
    for pair in pairs:
        item = dict(pair)
        for side in ("parsed", "shown"):
            result = by_id[pair[side + "_group_id"]]
            item[side + "_status"] = result["status"]
            item[side + "_coverage"] = (result.get("judgement") or {}).get("coverage", "audit_error")
        paired.append(item)
    summary = {
        "unique_statuses": dict(Counter(r["status"] for r in results)),
        "parsed_state_coverage": dict(Counter(p["parsed_coverage"] for p in paired)),
        "shown_state_coverage": dict(Counter(p["shown_coverage"] for p in paired)),
        "paired_transitions": dict(Counter(p["parsed_coverage"] + "->" + p["shown_coverage"] for p in paired)),
    }
    report = {**manifest, "preflight_sha256": sha(preflight), "results": [by_id[c["id"]] for c in cases], "paired_results": paired, "summary": summary}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
