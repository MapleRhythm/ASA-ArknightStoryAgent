"""Prepare a backed-up, narrowly scoped patch after exact replay comparisons."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def load(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
release = Path("release/ASA-ArknightStoryAgent")
candidate = run / "inferred_integrity_candidate_v2"
spec = load(candidate / "manifest.json")
comparison = load(run / "inferred_integrity_v2_comparison.json")
assert comparison["candidate_tests"] == 115 and comparison["v1_cross_pool_regression_reproduced"]
expected_changed = "relabel-60cb10813e52f206ff5c105e::current_quote_fixed79"
fixed, strict, paired = comparison["comparisons"]
assert fixed["action_changed_ids"] == fixed["answer_changed_ids"] == [expected_changed]
assert strict["grounded_changes"] == []
assert paired["action_changed_ids"] == paired["answer_changed_ids"] == []
assert all(not item["v1_v2_grounded_difference_ids"] for item in comparison["comparisons"])
tests = run / "inferred_integrity_candidate_tests_v2.xml"
suites = list(ET.parse(tests).getroot().iter("testsuite"))
assert sum(int(s.get("tests", 0)) for s in suites) == 115
assert all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0 for s in suites)
for info in spec["source_files"]:
    assert digest((repo / release / "src" / info["path"]).read_bytes()) == info["before_sha256"], info["path"]
    assert digest((candidate / "src" / info["path"]).read_bytes()) == info["after_sha256"], info["path"]
frozen = load(run / "manifest.json")
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert head == frozen["git_head"]
folder = run / "canonical_inferred_integrity_patch_v2"
folder.mkdir()
content = {
    release / "src" / path: (candidate / "src" / path).read_bytes()
    for path in spec["changed_paths"]
}
test_rel = release / "tests/test_inferred_quote_integrity.py"
assert not (repo / test_rel).exists()
content[test_rel] = (run / "inferred_integrity_patch_files_v2/test_inferred_quote_integrity.py").read_bytes()
patch, changes = [], []
for relative, new in content.items():
    path = repo / relative
    exists = path.exists()
    before = path.read_bytes() if exists else b""
    if exists:
        backup = folder / "before" / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as output:
            output.write(before)
    patch.extend(difflib.unified_diff(
        before.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True),
        fromfile=f"a/{relative}" if exists else "/dev/null", tofile=f"b/{relative}",
    ))
    changes.append({
        "path": str(relative), "existed_before": exists,
        "before_sha256": digest(before) if exists else None, "after_sha256": digest(new),
    })
patch_bytes = "".join(patch).encode()
with (folder / "change.patch").open("xb") as output:
    output.write(patch_bytes)
report = {
    "canonical_host": "${REMOTE_HOST}", "head": head, "applied": False,
    "patch_sha256": digest(patch_bytes), "changes": changes, "candidate_tests_passed": 115,
    "comparison_sha256": digest((run / "inferred_integrity_v2_comparison.json").read_bytes()),
    "fixed79_action_changed_ids": [expected_changed],
    "strict9_grounded_results_unchanged": True, "paired68_actions_and_answers_unchanged": True,
    "v1_candidate_not_deployed": True, "inference_only_quotes_not_shared_with_supported_facts": True,
    "generation_accuracy_improvement_claimed": False,
    "limitations": [
        "Checks normalized inference references; the existing compactor still limits reference counts.",
        "Matching citations and explicit inference labels do not prove semantic entailment.",
        "All replay outcomes are frozen evidence states, not fresh end-to-end final answers.",
    ],
}
with (folder / "manifest.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
subprocess.run(["git", "-C", str(repo), "apply", "--check", str(folder / "change.patch")], check=True)
print(json.dumps({"patch": str(folder / "change.patch"), "files": len(changes),
                  "candidate_tests_passed": 115, "applied": False}))
