"""Blind, deduplicated coverage and source audits; retain all116 parent outcomes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from build_fact_coverage_pairs_v1 import HERE, RUN, load, sha

SEM = RUN / "semantic79_content_v2.fPkkhM"
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(SEM))
import probe_claim_entailment_v1 as transport
import probe_answer_slots_v3 as coverage
import answer_semantics_content_v2 as semantics


def audit_messages(case):
    return coverage.messages(case) if case["audit_kind"] == "coverage" else semantics.messages(case)


def audit_validate(value, case):
    return coverage.validate(value, case) if case["audit_kind"] == "coverage" else semantics.validate(value, case)


def prepare(analysis):
    rows = analysis["results"]
    assert len({row["id"] for row in rows}) == len(rows)
    groups, states = {}, []
    for row in rows:
        state = {"id": row["id"], "renderer": row["renderer"], "question": row["question"]}
        if row["status"] != "ok":
            state["state"] = "pipeline_error"
        elif row["grounded"]["next_action"] != "answer_directly":
            state["state"] = "no_answer_action"
            state["action"] = row["grounded"]["next_action"]
        else:
            state["state"] = "direct_answer"
            assert row["grounded"]["answer"].strip()
            for kind in ("coverage", "semantics"):
                case = {
                    "audit_kind": kind, "question": row["question"],
                    "shown_answer": row["grounded"]["answer"],
                    "evidence_brief": row["evidence_brief"],
                    "grounded": {
                        "next_action": row["grounded"]["next_action"],
                        "answer": row["grounded"]["answer"],
                    },
                    "current_round": row["current_round"],
                    "max_retrieval_rounds": row["max_retrieval_rounds"],
                }
                # Group on the actual provider-visible payload, not row ID.
                group_id = kind + "_" + transport.digest(audit_messages(case))
                if group_id not in groups:
                    groups[group_id] = {"id": group_id, **case, "members": []}
                groups[group_id]["members"].append(row["id"])
                state[kind + "_group_id"] = group_id
        states.append(state)
    return list(groups.values()), states


def summarize(states, results):
    by_id = {result["id"]: result for result in results}
    labeled = []
    for state in states:
        item = dict(state)
        for kind, label in (("coverage", "coverage"), ("semantics", "verdict")):
            group_id = state.get(kind + "_group_id")
            if group_id is None:
                item[kind + "_label"] = state["state"]
            else:
                result = by_id[group_id]
                item[kind + "_label"] = (
                    result["judgement"][label] if result["status"] == "ok" else "audit_error"
                )
        labeled.append(item)
    summary = {}
    for renderer in sorted({r["renderer"] for r in states}):
        subset = [r for r in labeled if r["renderer"] == renderer]
        summary[renderer] = {
            "states": len(subset),
            "coverage": dict(Counter(r["coverage_label"] for r in subset)),
            "semantics": dict(Counter(r["semantics_label"] for r in subset)),
        }
    return labeled, summary


def main():
    source = HERE / "validation_v1.json"
    destination = HERE / "blind_audits_v1.json"
    journal = HERE / "blind_audits_v1.progress.jsonl"
    preflight = HERE / "blind_audits_preflight_v1.json"
    assert not any(p.exists() for p in (destination, journal, preflight))
    analysis = load(source)
    assert len(analysis["results"]) == 116
    suites = ET.parse(HERE / "audit_tests_v1.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 5
    assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    cases, states = prepare(analysis)
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    assert key, "Provider credential unavailable"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "judge_type": "model_not_human",
        "source_sha256": sha(source), "matrix_sha256": sha(HERE / "fact_coverage_pairs116_v1.matrix.json"),
        "driver_sha256": sha(Path(__file__)),
        "dependencies_sha256": {
            str(p): sha(p) for p in (
                RUN / "probe_claim_entailment_v1.py", RUN / "answer_semantics_content_v2.py",
                SEM / "probe_answer_slots_v1.py", SEM / "probe_answer_slots_v2.py",
                SEM / "probe_answer_slots_v3.py",
            )
        },
        "cases": cases, "states": states,
        "unique_provider_inputs": len(cases),
        "audit_kinds": dict(Counter(c["audit_kind"] for c in cases)),
        "protocol": [
            "Every116 parent state remains in summaries, including no-answer actions and pipeline errors.",
            "No API cost for nondirect states: deterministic state labels are not presented as model reviews.",
            "Deduplicate only identical actual provider-visible messages within an audit kind.",
            "Coverage uses frozen v3 question+answer rubric; semantics uses frozen full-evidence content-v2 rubric.",
            "Neither grader sees variant, old answer, alternative answer, old verdict, gold or source-review corrections.",
            "One request per unique input, two concurrent requests total, no retry or malformed-output repair.",
            "The two model opinions are not independent human gold or a production promotion gate.",
            "Known content-v2 cause-scope/identity false-positive risks require independent source review.",
            "Do not claim E2E answer accuracy, RAG recall or generalization from this development experiment.",
        ],
    }
    with preflight.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        "states": len(states), "state_counts": dict(Counter(s["state"] for s in states)),
        "unique_requests": len(cases), "audit_kinds": manifest["audit_kinds"],
    }), flush=True)
    transport.messages = audit_messages
    transport.validate = audit_validate
    results = []
    with journal.open("x", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(transport.call, case, key)
                for case in sorted(cases, key=lambda c: transport.digest(audit_messages(c)))
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({
                    "completed": len(results), "total": len(cases),
                    "id": result["id"], "status": result["status"], "error": result.get("error"),
                }), flush=True)
    by_id = {result["id"]: result for result in results}
    assert len(by_id) == len(cases) and set(by_id) == {case["id"] for case in cases}
    assert sha(source) == manifest["source_sha256"]
    labeled, summary = summarize(states, results)
    report = {
        **manifest, "preflight_sha256": sha(preflight),
        "results": [by_id[case["id"]] for case in cases],
        "labeled_states": labeled, "summary": summary,
        "unique_statuses": dict(Counter(r["status"] for r in results)),
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"unique_statuses": report["unique_statuses"], "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
