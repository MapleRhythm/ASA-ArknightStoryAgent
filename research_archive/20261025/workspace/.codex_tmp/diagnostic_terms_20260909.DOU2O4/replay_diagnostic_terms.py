"""Paired query-state and grounding replays of a frozen diagnostic-cleaning patch."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
CANONICAL = ROOT / "release/ASA-ArknightStoryAgent/src"
ORIGINAL = HERE / "original/src"
CANDIDATE = HERE / "candidate/src"
CASES = [
    ("fixed79", "fixed79.current_quote.matrix.json", "fixed79_configured_lora_ctx12288", 79),
    ("strict9", "after.matrix.json", "strict9_vllm_context_guard_ctx12288_stable_idle", 9),
    ("paired68", "same_evidence_retry_paired68_v1.matrix.json", "same_evidence_retry_paired68_ctx12288_v1", 68),
    ("focus18", "question_focus_paired18_v1.matrix.json", "question_focus_paired18_ctx12288_v1", 18),
    ("quote_sft9", "quote80_sft_adapter_probe9_v1.matrix.json", "quote80_sft_adapter_probe9_ctx12288_v1", 9),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def code_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))}


original_hashes, candidate_hashes = code_hashes(ORIGINAL), code_hashes(CANDIDATE)
canonical_hashes = code_hashes(CANONICAL / "asa_arknight_story_agent")
assert {"asa_arknight_story_agent/" + k: v for k, v in canonical_hashes.items()} == original_hashes
changed = {p for p in candidate_hashes if candidate_hashes[p] != original_hashes.get(p)}
prefix = "asa_arknight_story_agent/inference/"
assert changed == {
    prefix + "payload/hypothesis_payload.py", prefix + "planning/retrieval_term_cleaning.py",
    prefix + "planning/follow_up_hypothesis.py", prefix + "planning/follow_up_query_generation.py",
    prefix + "retrieval/query_rendering.py",
}
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "evaluation_only": True, "production_changed": False, "training_ready": False,
    "canonical_source_hashes": original_hashes, "candidate_source_hashes": candidate_hashes,
    "changed_source_files": sorted(changed), "comparisons": [],
}
for label, source in (("baseline", ORIGINAL), ("candidate", CANDIDATE)):
    subprocess.run([
        sys.executable, str(HERE / "probe_diagnostic_term_states.py"),
        "--source-root", str(source), "--output", str(HERE / f"{label}_query_states_v2.json"),
    ], check=True)
old_states = load(HERE / "baseline_query_states_v2.json")
new_states = load(HERE / "candidate_query_states_v2.json")
assert old_states["source_files"] == new_states["source_files"]
old = {r["id"]: r for r in old_states["results"]}
new = {r["id"]: r for r in new_states["results"]}
assert old.keys() == new.keys()
state_changes = []
for key, before in old.items():
    after = new[key]
    assert before["hypothesis"] == after["hypothesis"]
    changed_fields = [field for field in (
        "follow_up", "merged", "direct_query", "follow_up_queries",
        "merged_query", "normalized", "normalization_error",
    ) if before.get(field) != after.get(field)]
    if changed_fields:
        state_changes.append({
            "id": key, "source_kind": before["source_kind"], "question": before["question"],
            "changed_fields": changed_fields,
            "removed_entities": sorted(set(before["merged"]["entities"]) - set(after["merged"]["entities"])),
            "removed_keywords": sorted(set(before["merged"]["keywords"]) - set(after["merged"]["keywords"])),
            "new_normalization_error": after.get("normalization_error") if not before.get("normalization_error") else None,
        })
report["state_replay"] = {
    "states": len(old), "training_input_stats": old_states["training_input_stats"],
    "source_files": old_states["source_files"], "changes": state_changes,
    "changes_by_source": dict(Counter(r["source_kind"] for r in state_changes)),
    "all_removed_terms": sorted({term for row in state_changes
                                for term in row["removed_entities"] + row["removed_keywords"]}),
    "new_normalization_errors": [r["id"] for r in state_changes if r["new_normalization_error"]],
}
print(json.dumps({k: v for k, v in report["state_replay"].items()
                  if k not in {"changes", "source_files"}}, ensure_ascii=False), flush=True)
for name, matrix_name, folder, count in CASES:
    matrix, predictions = RUN / matrix_name, RUN / folder / "predictions.json"
    results = {}
    for label, source in (("baseline", ORIGINAL), ("candidate", CANDIDATE)):
        output = HERE / f"{label}_{name}_v2.json"
        subprocess.run([
            sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
            "--validator", "after", "--source-root", str(source),
            "--matrix", str(matrix), "--predictions", str(predictions), "--output", str(output),
        ], check=True, capture_output=True, text=True)
        results[label] = load(output)
    old = {r["id"]: r for r in results["baseline"]["results"]}
    new = {r["id"]: r for r in results["candidate"]["results"]}
    assert old.keys() == new.keys() and len(old) == count
    assert all(r["status"] == "ok" for r in new.values())
    assert all(old[k]["raw_output"] == new[k]["raw_output"] for k in old)
    item = {
        "name": name, "rows": count, "matrix": str(matrix), "predictions": str(predictions),
        "matrix_sha256": sha(matrix), "predictions_sha256": sha(predictions),
        "action_changed_ids": [k for k in old if old[k]["grounded"]["next_action"] != new[k]["grounded"]["next_action"]],
        "answer_changed_ids": [k for k in old if old[k]["grounded"]["answer"] != new[k]["grounded"]["answer"]],
        "parsed_changed_ids": [k for k in old if old[k]["parsed"] != new[k]["parsed"]],
        "grounded_changed_ids": [k for k in old if old[k]["grounded"] != new[k]["grounded"]],
        "after_summary": results["candidate"]["summary"],
    }
    report["comparisons"].append(item)
    print(json.dumps(item, ensure_ascii=False), flush=True)
assert code_hashes(ORIGINAL) == original_hashes and code_hashes(CANDIDATE) == candidate_hashes
assert code_hashes(CANONICAL / "asa_arknight_story_agent") == canonical_hashes
report["limitations"] = [
    "Query-state changes are not retrieval recall or answer-accuracy measurements.",
    "183 grounding rows are repeated development states; no new retrieval/generation or training.",
    "Only known internal diagnostic patterns are removed; this is not a general contamination detector.",
]
with (HERE / "diagnostic_terms_replay_comparison_v2.json").open("x", encoding="utf-8") as output:
    json.dump(report, output, ensure_ascii=False, indent=2)
