"""Read saved normalized traces; never invent pre-normalization planner output."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
RESULTS = Path("${REMOTE_ASA_ROOT}/rank_v8_coverage/runs/e2e_no_minirag_fixed58_20260909.v23v3/results.jsonl")
SRC = ROOT / "release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / "trace_provenance_v1.json"
    assert not output.exists()
    trace_source = SRC / "pipeline/trace.py"
    generation_source = SRC / "generation/hypothesis_generation.py"
    trace_text = trace_source.read_text()
    generation_text = generation_source.read_text()
    assert '"hypothesis": asdict(hypothesis)' in trace_text
    assert "return normalize_hypothesis_payload(" in generation_text
    original_sha = sha(RESULTS)
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    assert len(rows) == 58
    observations = []
    for index, row in enumerate(rows):
        steps = []
        for trace in row.get("retrieval_trace", []):
            steps.append({
                "round": trace["round"], "keys": sorted(trace),
                "hypothesis": trace.get("hypothesis"),
                "follow_up_hypothesis": trace.get("follow_up_hypothesis"),
            })
        observations.append({
            "row_index": index, "question": row["question"],
            "top_level_keys": sorted(row), "steps": steps,
            "top_level_raw_output_present": any(
                key in row for key in ("raw_output", "raw_hypothesis", "hypothesis_raw_output")
            ),
            "trace_raw_output_present": any(
                key in trace for trace in row.get("retrieval_trace", [])
                for key in ("raw_output", "raw_hypothesis", "hypothesis_raw_output")
            ),
        })
    assert not any(r["top_level_raw_output_present"] or r["trace_raw_output_present"] for r in observations)
    sample = next(r for r in observations if "小大厨" in r["question"])
    assert sample["steps"]
    assert "出煌" in sample["steps"][0]["hypothesis"]["entities"]
    assert sha(RESULTS) == original_sha
    report = {
        "version": "saved_planner_trace_provenance_v36",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "gpu_used": False, "training_ready": False,
        "results_path": str(RESULTS), "results_sha256": original_sha,
        "questions": len(rows), "trace_steps": sum(len(r["steps"]) for r in observations),
        "basis_sha256": {str(p): sha(p) for p in (trace_source, generation_source, Path(__file__))},
        "finding": "Saved hypothesis traces are dataclass snapshots returned after normalization or heuristic fallback, not raw model payloads. The inspected top-level and trace keys do not store raw planner output.",
        "limits": [
            "This audit covers the saved results JSONL and current trace-generation source, not every historical log on the server.",
            "No inference that any particular saved entity originated in the model rather than normalization.",
            "Do not subtract heuristic entities from saved snapshots and call the result the original model payload.",
            "Fresh planner capture or explicitly synthetic payloads are required for a valid normalization intervention."
        ],
        "observations": observations,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print({key: report[key] for key in ("questions", "trace_steps", "finding")})


if __name__ == "__main__":
    main()
