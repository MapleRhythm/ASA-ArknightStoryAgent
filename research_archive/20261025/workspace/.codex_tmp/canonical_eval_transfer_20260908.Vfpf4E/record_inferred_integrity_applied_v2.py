"""Verify the latest hash per path across sequential, overlapping scoped patches."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
history, effective = [], {}
for name in (
    "canonical_context_guard_patch_v1",
    "canonical_quote_integrity_patch_v1",
    "canonical_inferred_integrity_patch_v2",
):
    folder = run / name
    spec = json.loads((folder / "manifest.json").read_text())
    assert sha(folder / "change.patch") == spec["patch_sha256"]
    for change in spec["changes"]:
        path = change["path"]
        if path in effective:
            assert change["before_sha256"] == effective[path]["after_sha256"], path
        if change["existed_before"]:
            assert sha(folder / "before" / path) == change["before_sha256"], path
        history.append({"patch": name, **change})
        effective[path] = change
for path, record in effective.items():
    assert sha(repo / path) == record["after_sha256"], path
assert len(effective) == 9
frozen = json.loads((run / "manifest.json").read_text())
expected = {str(repo / path): record["after_sha256"] for path, record in effective.items()}
checked, changed = [], []
for relative, item in frozen["source_files"].items():
    if not relative.startswith("after/"):
        continue
    actual = sha(Path(item["origin"]))
    assert actual == expected.get(item["origin"], item["sha256"]), item["origin"]
    checked.append(item["origin"])
    if actual != item["sha256"]:
        changed.append(item["origin"])
assert len(checked) == 208 and len(changed) == 6
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert head == frozen["git_head"]
subprocess.run(["git", "-C", str(repo), "diff", "--check"], check=True)
xml_path = run / "canonical_release_tests_inferred_integrity_v2.xml"
suites = list(ET.parse(xml_path).getroot().iter("testsuite"))
assert sum(int(s.get("tests", 0)) for s in suites) == 115
assert all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0 for s in suites)
report = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "canonical_host": "${REMOTE_HOST}", "head": head, "applied": True,
    "test_count": 115, "test_receipt_sha256": sha(xml_path),
    "cumulative_unique_changed_files": len(effective),
    "effective_changes": list(effective.values()), "patch_history": history,
    "frozen_source_files_rechecked": len(checked), "changed_frozen_source_files": changed,
    "service_restarted": False, "committed": False, "pushed": False, "training_started": False,
}
with (run / "canonical_inferred_integrity_patch_v2/applied_verified.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("effective_changes", "patch_history")}))
