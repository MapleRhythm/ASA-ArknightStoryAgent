"""Verify both scoped patches and the 106-test canonical release checkpoint."""
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
records = []
for name in ("canonical_context_guard_patch_v1", "canonical_quote_integrity_patch_v1"):
    folder = run / name
    spec = json.loads((folder / "manifest.json").read_text())
    assert sha(folder / "change.patch") == spec["patch_sha256"]
    for change in spec["changes"]:
        assert sha(repo / change["path"]) == change["after_sha256"], change["path"]
        if change["existed_before"]:
            assert sha(folder / "before" / change["path"]) == change["before_sha256"]
        records.append(change)
assert len({r["path"] for r in records}) == len(records) == 5
manifest = json.loads((run / "manifest.json").read_text())
expected = {str(repo / r["path"]): r["after_sha256"] for r in records}
checked, changed = [], []
for relative, item in manifest["source_files"].items():
    if not relative.startswith("after/"):
        continue
    checksum = sha(Path(item["origin"]))
    assert checksum == expected.get(item["origin"], item["sha256"]), item["origin"]
    checked.append(item["origin"])
    if checksum != item["sha256"]:
        changed.append(item["origin"])
assert len(checked) == 208 and len(changed) == 3
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert head == manifest["git_head"]
subprocess.run(["git", "-C", str(repo), "diff", "--check"], check=True)
xml_path = run / "canonical_release_tests_quote_integrity_v1.xml"
suites = list(ET.parse(xml_path).getroot().iter("testsuite"))
assert sum(int(s.get("tests", 0)) for s in suites) == 106
assert all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0 for s in suites)
report = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "canonical_host": "${REMOTE_HOST}", "head": head, "applied": True,
    "cumulative_changed_files_this_phase": records,
    "frozen_source_files_rechecked": len(checked), "changed_frozen_source_files": changed,
    "test_count": 106, "test_receipt_sha256": sha(xml_path),
    "service_restarted": False, "committed": False, "pushed": False, "training_started": False,
}
with (run / "canonical_quote_integrity_patch_v1/applied_verified.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "cumulative_changed_files_this_phase"}))
