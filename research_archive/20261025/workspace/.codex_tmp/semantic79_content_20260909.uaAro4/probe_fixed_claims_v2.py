"""Diagnostic only: full cited blocks for all facts in seven selected states."""
from collections import Counter
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN))
import probe_claim_entailment_v1 as probe

sys.path.insert(0, "${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt

SELECTED_PREFIXES = {
    "0b3f0002117b06cb__", "relabel-3c4ca0be007bfbbb7dfde508::",
    "relabel-92de73ea55deebeb48f5d07c::", "relabel-f4ec283e29fcc994d667b6b1::",
    "relabel-ffc93a494808c61623543ea9::", "relabel-23ebdc399dbe63646b934d15::",
    "2711ab940eb8a3b6__",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = HERE / "canonical_fixed79_validation_v1.json"
    cases = []
    rows = json.loads(source.read_text())["results"]
    selected = [r for r in rows if any(r["id"].startswith(p) for p in SELECTED_PREFIXES)]
    assert len(selected) == 7
    for row in selected:
        assert row["status"] == "ok" and row["grounded"]["next_action"] == "answer_directly"
        blocks = evidence_id_text_map_from_prompt(row["evidence_brief"])
        for index, fact in enumerate(row["grounded"]["supported_facts"], 1):
            ids = list(dict.fromkeys(ref["evidence_id"] for ref in fact["evidence_refs"]))
            assert ids and all(eid in blocks for eid in ids)
            cases.append({
                "id": f'{row["id"]}::fact_{index}', "parent_id": row["id"], "fact_index": index,
                "question": row["question"], "claim": fact["fact"],
                "original_refs": fact["evidence_refs"],
                "cited_blocks": {eid: blocks[eid] for eid in ids},
            })
    assert len(cases) == 13
    output = HERE / "cited_claim_probe13_v2.json"
    journal = output.with_suffix(".progress.jsonl")
    manifest_path = HERE / "cited_claim_probe13_preflight_v2.json"
    if any(path.exists() for path in (output, journal, manifest_path)):
        raise ValueError("Existing artifacts; never repeat or overwrite")
    key = os.environ.get("BIGMODEL_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    if not key:
        raise ValueError("Provider key unavailable")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "source_analysis_sha256": sha(source), "driver_sha256": sha(Path(__file__)),
        "probe_script_sha256": sha(RUN / "probe_claim_entailment_v1.py"),
        "case_count": len(cases), "state_count": len(selected),
        "cases": cases,
        "limitations": [
            "Selected development diagnostics, including known disagreements, not independent accuracy data.",
            "All facts from each chosen state included, not only disputed facts; Nasti is an interpretive control.",
            "Only question, unchanged fact and full cited blocks are sent, never gold, prior judge or review notes.",
            "Fact entailment is distinct from shown-answer relevance/completeness; the Yato case tests this separation.",
            "No automatic semantic gate, answer rewrite, label promotion, training or deployment.",
        ],
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    results = []
    with journal.open("x", encoding="utf-8") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(probe.call, case, key) for case in cases]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({
                    "completed": len(results), "total": len(cases), "id": result["id"],
                    "status": result["status"], "error": result.get("error"),
                }), flush=True)
    by_id = {r["id"]: r for r in results}
    assert len(by_id) == len(cases)
    report = {
        **manifest, "preflight_sha256": sha(manifest_path),
        "results": [by_id[c["id"]] for c in cases],
        "statuses": dict(Counter(r["status"] for r in results)),
        "relations": dict(Counter((r.get("judgement") or {}).get("relation", "audit_error") for r in results)),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"statuses": report["statuses"], "relations": report["relations"]}))


if __name__ == "__main__":
    main()
