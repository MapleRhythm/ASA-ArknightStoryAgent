"""Prepare the smallest quote-preservation patch after full replay checks."""
import difflib
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
candidate = run / "quote_integrity_candidate_v1"
manifest = read(candidate / "manifest.json")
release = Path("release/ASA-ArknightStoryAgent")
for info in manifest["source_files"]:
    path = repo / release / "src" / info["path"]
    assert sha(path.read_bytes()) == info["before_sha256"], f"Concurrent source edit: {path}"
suites = list(ET.parse(run / "quote_integrity_candidate_tests_v1.xml").getroot().iter("testsuite"))
assert sum(int(s.get("tests", 0)) for s in suites) == 106
assert all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0 for s in suites)
fixed = run / "fixed79_configured_lora_ctx12288"
before = read(fixed / "validation_after_v2.json")
after = read(fixed / "validation_quote_integrity_candidate_v1.json")
assert before["predictions_sha256"] == after["predictions_sha256"]
assert before["matrix_sha256"] == after["matrix_sha256"]
old = {r["id"]: r for r in before["results"]}
new = {r["id"]: r for r in after["results"]}
assert len(old) == len(new) == 79 and set(old) == set(new)
changed = []
action_changes = []
for key in old:
    if old[key]["grounded"] != new[key]["grounded"]:
        changed.append(key)
    if old[key]["grounded"]["next_action"] != new[key]["grounded"]["next_action"]:
        action_changes.append(key)
expected_bad = "relabel-4835313d2c74e78706d2b4e3::current_quote_fixed79"
expected_already_rejected = "576f60a60f04580d__teacher_state_grounded_action_sft::current_quote_fixed79"
assert set(changed) == {expected_bad, expected_already_rejected}, changed
assert action_changes == [expected_bad], action_changes
assert old[expected_bad]["grounded"]["next_action"] == "answer_directly"
assert new[expected_bad]["grounded"]["next_action"] == "retrieve_more"
assert "supported_fact_1_ref_1_quote_over_80" in new[expected_bad]["grounded"]["grounding_warnings"]
assert all(old[k]["grounded"]["answer"] == new[k]["grounded"]["answer"] for k in old if k != expected_bad)
context = run / "strict9_vllm_context_guard_ctx12288_stable_idle"
old_context = read(context / "validation_after_v2.json")["results"]
new_context = read(context / "validation_quote_integrity_candidate_v1.json")["results"]
assert [r["id"] for r in old_context] == [r["id"] for r in new_context]
assert all(a["grounded"] == b["grounded"] for a, b in zip(old_context, new_context))
folder = run / "canonical_quote_integrity_patch_v1"
folder.mkdir()
utils_rel = release / "src/asa_arknight_story_agent/inference/payload/utils.py"
test_rel = release / "tests/test_quote_payload_integrity.py"
assert not (repo / test_rel).exists()
content = {
    utils_rel: (run / "payload_utils.quote_integrity.py").read_bytes(),
    test_rel: (run / "test_quote_payload_integrity.py").read_bytes(),
}
patch, records = [], []
for relative, new_bytes in content.items():
    path = repo / relative
    old_bytes = path.read_bytes() if path.exists() else b""
    if path.exists():
        backup = folder / "before" / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as output:
            output.write(old_bytes)
    patch.extend(difflib.unified_diff(
        old_bytes.decode().splitlines(keepends=True), new_bytes.decode().splitlines(keepends=True),
        fromfile=f"a/{relative}" if path.exists() else "/dev/null", tofile=f"b/{relative}",
    ))
    records.append({
        "path": str(relative), "existed_before": path.exists(),
        "before_sha256": sha(old_bytes) if path.exists() else None, "after_sha256": sha(new_bytes),
    })
patch_bytes = "".join(patch).encode()
with (folder / "change.patch").open("xb") as output:
    output.write(patch_bytes)
report = {
    "canonical_host": "${REMOTE_HOST}", "applied": False,
    "patch_sha256": sha(patch_bytes), "changes": records,
    "candidate_tests_passed": 106, "fixed79_action_changed_ids": action_changes,
    "fixed79_grounded_result_changed_ids": changed, "strict9_all_grounded_results_unchanged": True,
    "frozen_generation_changed": False, "generation_accuracy_improvement_claimed": False,
    "known_bad_case_source_review": {
        "id": expected_bad,
        "answer": old[expected_bad]["grounded"]["answer"],
        "emitted_quote_length": 101, "previous_parser_quote_length": 80,
        "review": "原文说莱欧斯更着急、齐尔查克缩短路程，不能推出因此不急着赶路；实际因果断言与源文相反。",
    },
    "limitations": [
        "Only proves preservation of emitted quotes and enforcement of existing length policy.",
        "Suppression of one source-reviewed error in a frozen replay is not more questions answered correctly.",
        "Other semantic errors and uncited inferred_facts remain separate unresolved risks.",
        "An honest but overlong quote will also be rejected; same-evidence regeneration remains necessary.",
    ],
}
with (folder / "manifest.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "patch": str(folder / "change.patch"), "verified_tests": 106,
    "fixed79_action_changes": action_changes, "strict9_all_grounded_results_unchanged": True,
}))
