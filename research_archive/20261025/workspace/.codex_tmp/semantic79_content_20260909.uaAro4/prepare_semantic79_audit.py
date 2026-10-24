"""Freeze full fixed79 baseline under current canonical code, without generation."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SOURCE = ROOT / "release/ASA-ArknightStoryAgent/src"
matrix = RUN / "fixed79.current_quote.matrix.json"
predictions = RUN / "fixed79_configured_lora_ctx12288/predictions.json"
output = HERE / "canonical_fixed79_validation_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


applied = load(RUN / "diagnostic_terms_patch_v1.5xrKw6/applied_verified.json")
replay = load(RUN / "diagnostic_terms_patch_v1.5xrKw6/diagnostic_terms_replay_comparison_v2.json")
assert applied["canonical_tests"]["tests"] == 214
for relative, digest in replay["candidate_source_hashes"].items():
    assert sha(SOURCE / relative) == digest, relative
subprocess.run([
    sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
    "--validator", "after", "--source-root", str(SOURCE),
    "--matrix", str(matrix), "--predictions", str(predictions), "--output", str(output),
], check=True)
analysis = load(output)
rows = analysis["results"]
assert len(rows) == len({r["id"] for r in rows}) == 79
assert all(r["status"] == "ok" for r in rows)
prior = load(RUN / "diagnostic_terms_patch_v1.5xrKw6/candidate_fixed79_v2.json")
assert [r["grounded"] for r in rows] == [r["grounded"] for r in prior["results"]]
questions = {re.sub(r"\s+", "", r["question"]).strip("？?。！!，,；;：:\"'") for r in rows}
assert len(questions) == 58
actions = dict(Counter(r["grounded"]["next_action"] for r in rows))
assert actions == {"answer_directly": 39, "retrieve_more": 31, "abstain": 9}
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "evaluation_only": True, "training_ready": False, "new_generation": False,
    "new_retrieval": False, "states": 79, "distinct_questions": 58, "actions": actions,
    "matrix_sha256": sha(matrix), "predictions_sha256": sha(predictions),
    "analysis_sha256": sha(output), "canonical_source_hashes": replay["candidate_source_hashes"],
    "audit_protocol": "asa_visible_answer_semantics_v2",
    "audit_implementation": {
        name: sha(RUN / name) for name in (
            "judge_visible_answers_content_v3.py", "answer_semantics_content_v2.py",
            "judge_visible_answers.py", "analyze_canonical_grounding_v4.py",
        )
    },
    "requested_model": "glm-5.3", "workers": 2, "max_tokens": 16384,
    "limitations": [
        "79 frozen evidence states from 58 questions; not 79 independent fresh E2E tests.",
        "Model audit labels are provisional and must be reviewed before any training use.",
        "The entire denominator is retained, including unanswered states and audit failures.",
    ],
}
with (HERE / "preflight_manifest.json").open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, ensure_ascii=False, indent=2)
print(json.dumps({"states": 79, "questions": 58, "actions": actions}, ensure_ascii=False))
