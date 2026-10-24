"""Two frozen coverage rubrics, twenty independent synthetic minimal controls."""
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
import probe_answer_slots_v1 as transport_owner
import probe_answer_slots_v2 as v2
import probe_answer_slots_v3 as v3

CONDITIONS = {"schema_v2": v2, "cause_scope_v3": v3}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def messages(case):
    return CONDITIONS[case["condition"]].messages(case)


def main():
    fixture_path = HERE / "slot_cause_scope_synthetic20_v1.json"
    fixture = json.loads(fixture_path.read_text())
    assert fixture["evaluation_only"] and not fixture["training_ready"]
    assert len(fixture["cases"]) == 20
    output = HERE / "slot_cause_scope40_v3.json"
    preflight = HERE / "slot_cause_scope40_preflight_v3.json"
    journal = HERE / "slot_cause_scope40_v3.progress.jsonl"
    assert not any(p.exists() for p in (output, preflight, journal))
    suites = ET.parse(HERE / "slot_cause_scope_tests_v3.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 4
    assert all(all(int(s.attrib.get(k, 0)) == 0 for k in ("errors", "failures", "skipped")) for s in suites)
    pairs, cases = [], []
    for index, row in enumerate(fixture["cases"]):
        pair = []
        order = list(CONDITIONS)
        if index % 2:
            order.reverse()
        for condition in order:
            case = {
                "id": row["id"] + "::" + condition, "parent_id": row["id"],
                "condition": condition, "question": row["question"], "shown_answer": row["shown_answer"],
            }
            pair.append(case)
            cases.append(case)
        assert messages(pair[0])[1] == messages(pair[1])[1]
        pairs.append(pair)
    assert len(cases) == 40 and len({c["id"] for c in cases}) == 40
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    assert key, "Provider credential unavailable"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "slot_cause_scope40_v3", "evaluation_only": True, "training_ready": False,
        "fixture_sha256": sha(fixture_path), "driver_sha256": sha(Path(__file__)),
        "rubric_script_sha256": {name: sha(HERE / f"probe_answer_slots_{ver}.py") for name, ver in (("schema_v2", "v2"), ("cause_scope_v3", "v3"))},
        "transport_sha256": sha(HERE.parent / "probe_claim_entailment_v1.py"),
        "cases": cases,
        "protocol": [
            "Twenty assistant-authored synthetic cases frozen before any provider call; not human story gold.",
            "Immediate instruction/rule/condition causes, same-subject tautologies, explicit deeper-purpose questions, identity controls and two-part coverage.",
            "Question and shown answer only; no expected label, rationale, story evidence, review or condition identifier is sent.",
            "The sole v3 rubric addition clarifies causal scope; unchanged strict schema, validator and model parameters.",
            "Ten v2-first and ten v3-first pairs. Sequential within pairs, at most two concurrent pairs.",
            "One request per case/condition, no retry, no selection on outcomes.",
            "All 20 inputs per condition remain in denominator; schema errors count as unmatched oracle.",
            "Keep raw model responses. Do not generalize synthetic contract agreement to ASA answer accuracy.",
            "No production gate, data relabel, model/config change or promotion based on this set.",
        ],
    }
    with preflight.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    transport_owner.transport.messages = messages
    transport_owner.transport.validate = v2.validate
    results = []
    lock = threading.Lock()
    with journal.open("x", encoding="utf-8") as stream:
        def run_pair(pair):
            for case in pair:
                result = transport_owner.transport.call(case, key)
                result.update(parent_id=case["parent_id"], condition=case["condition"])
                with lock:
                    results.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(json.dumps({"completed": len(results), "total": 40, "id": result["id"], "status": result["status"], "error": result.get("error")}), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run_pair, pairs))
    by_id = {r["id"]: r for r in results}
    assert len(by_id) == 40 and set(by_id) == {c["id"] for c in cases}
    assert sha(fixture_path) == manifest["fixture_sha256"]
    expected = {c["id"]: c["expected_coverage"] for c in fixture["cases"]}
    summary = {}
    for condition in CONDITIONS:
        group = [r for r in results if r["condition"] == condition]
        assert len(group) == 20
        summary[condition] = {
            "cases": 20, "statuses": dict(Counter(r["status"] for r in group)),
            "coverage": dict(Counter((r.get("judgement") or {}).get("coverage", "audit_error") for r in group)),
            "synthetic_contract_matches_including_errors": sum(r["status"] == "ok" and r["judgement"]["coverage"] == expected[r["parent_id"]] for r in group),
            "mismatches": [
                {"parent_id": r["parent_id"], "expected": expected[r["parent_id"]], "observed": (r.get("judgement") or {}).get("coverage", "audit_error")}
                for r in group if r["status"] != "ok" or r["judgement"]["coverage"] != expected[r["parent_id"]]
            ],
        }
    report = {**manifest, "preflight_sha256": sha(preflight), "results": [by_id[c["id"]] for c in cases], "summary": summary}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
