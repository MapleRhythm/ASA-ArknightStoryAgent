"""Freeze the corrected inference-reference candidate, without modifying release."""
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


run = Path(__file__).resolve().parent
source = run / "quote_integrity_candidate_v1/src"
canonical = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
patch_files = run / "inferred_integrity_patch_files_v2"
candidate = run / "inferred_integrity_candidate_v2"
target = candidate / "src"
target.mkdir(parents=True)
prefix = Path("asa_arknight_story_agent/inference")
replacements = {
    prefix / "payload/utils.py": patch_files / "utils.py",
    prefix / "grounding/quote_validation.py": patch_files / "quote_validation.py",
    prefix / "grounding/quote_checks.py": patch_files / "quote_checks.py",
    prefix / "grounding/grounded_fact_answers.py": patch_files / "grounded_fact_answers.py",
}
records = []
for path in sorted(source.rglob("*")):
    relative = path.relative_to(source)
    if "__pycache__" in relative.parts or path.suffix == ".pyc" or not path.is_file():
        continue
    assert not path.is_symlink()
    old = path.read_bytes()
    assert (canonical / relative).read_bytes() == old, f"Concurrent canonical edit: {relative}"
    new = replacements[relative].read_bytes() if relative in replacements else old
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        output.write(new)
    records.append({
        "path": str(relative), "before_sha256": digest(old), "after_sha256": digest(new),
        "changed": old != new,
    })
assert {r["path"] for r in records if r["changed"]} == {str(p) for p in replacements}
report = {
    "evaluation_only": True, "production_changed": False, "version": 2,
    "source_files": records, "changed_paths": [str(p) for p in replacements],
    "v1_regression_prevented": "Supported facts cannot borrow inference-only quote terms.",
    "limitations": [
        "Citation integrity does not prove semantic entailment or logical validity of inference.",
        "Uncited inferences are rejected; valid cited inferences retain an explicit inference label.",
        "No prompt, model checkpoint, production source or runtime configuration is changed.",
    ],
}
with (candidate / "manifest.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"files_verified": len(records), "changed_paths": report["changed_paths"],
                  "production_changed": False}))
