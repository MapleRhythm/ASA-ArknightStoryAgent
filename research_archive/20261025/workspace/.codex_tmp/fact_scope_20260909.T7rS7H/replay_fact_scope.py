"""Paired CPU-only revalidation of frozen outputs with canonical/candidate code."""
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
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted((root / "asa_arknight_story_agent").rglob("*.py"))
    }


before_hashes, candidate_hashes = code_hashes(CANONICAL), code_hashes(CANDIDATE)
assert before_hashes.keys() == candidate_hashes.keys()
changed = {name for name in before_hashes if before_hashes[name] != candidate_hashes[name]}
assert changed == {
    "asa_arknight_story_agent/inference/grounding/grounded_fact_answers.py",
    "asa_arknight_story_agent/inference/grounding/quote_validation.py",
}
for name in changed:
    assert before_hashes[name] == sha(HERE / "original" / Path(name).name)
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "evaluation_only": True, "training_ready": False, "production_changed": False,
    "new_generation": False, "fresh_retrieval": False,
    "canonical_source_hashes": before_hashes, "candidate_source_hashes": candidate_hashes,
    "changed_source_files": sorted(changed), "comparisons": [],
}
for name, matrix_name, folder, count in CASES:
    matrix, predictions = RUN / matrix_name, RUN / folder / "predictions.json"
    source_hashes = {"matrix": sha(matrix), "predictions": sha(predictions)}
    reports = {}
    for label, source in (("baseline", CANONICAL), ("candidate", CANDIDATE)):
        output = HERE / f"{label}_{name}_v1.json"
        command = [
            sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
            "--validator", "after", "--source-root", str(source),
            "--matrix", str(matrix), "--predictions", str(predictions), "--output", str(output),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        reports[label] = load(output)
        assert reports[label]["matrix_sha256"] == source_hashes["matrix"]
        assert reports[label]["predictions_sha256"] == source_hashes["predictions"]
    old = {row["id"]: row for row in reports["baseline"]["results"]}
    new = {row["id"]: row for row in reports["candidate"]["results"]}
    assert len(old) == len(new) == count and old.keys() == new.keys()
    changes = []
    for key in old:
        a, b = old[key], new[key]
        assert a["status"] == b["status"] == "ok", (name, key)
        assert a["parsed"] == b["parsed"] and a["raw_output"] == b["raw_output"]
        if b["grounded"]["next_action"] == "answer_directly":
            assert a["grounded"]["next_action"] == "answer_directly"
        if a["grounded"] != b["grounded"]:
            changes.append({
                "id": key, "question": a["question"], "before": a["grounded"], "after": b["grounded"],
                "parsed": a["parsed"], "visible_evidence": a["evidence_brief"],
            })
    item = {
        "name": name, "rows": count, "matrix": str(matrix), "predictions": str(predictions),
        "source_sha256": source_hashes,
        "action_changed_ids": [
            key for key in old if old[key]["grounded"]["next_action"] != new[key]["grounded"]["next_action"]
        ],
        "answer_changed_ids": [
            key for key in old if old[key]["grounded"]["answer"] != new[key]["grounded"]["answer"]
        ],
        "grounded_changes": changes, "before_summary": reports["baseline"]["summary"],
        "after_summary": reports["candidate"]["summary"],
        "baseline_report_sha256": sha(HERE / f"baseline_{name}_v1.json"),
        "candidate_report_sha256": sha(HERE / f"candidate_{name}_v1.json"),
    }
    report["comparisons"].append(item)
    print(json.dumps({
        "case": name, "rows": count, "action_changed": item["action_changed_ids"],
        "answer_changed": item["answer_changed_ids"], "grounded_changes": len(changes),
        "after_summary": item["after_summary"],
    }, ensure_ascii=False), flush=True)
    assert {"matrix": sha(matrix), "predictions": sha(predictions)} == source_hashes
assert code_hashes(CANONICAL) == before_hashes
assert code_hashes(CANDIDATE) == candidate_hashes
report["limitations"] = [
    "These 183 rows include repeated development evidence states, not 183 independent held-out questions.",
    "No answer accuracy or retrieval recall is measured by this mechanical replay.",
    "A scoped keyword check is not a semantic entailment or entity-role validator.",
]
with (HERE / "fact_scope_replay_comparison_v1.json").open("x", encoding="utf-8") as output:
    json.dump(report, output, ensure_ascii=False, indent=2)
