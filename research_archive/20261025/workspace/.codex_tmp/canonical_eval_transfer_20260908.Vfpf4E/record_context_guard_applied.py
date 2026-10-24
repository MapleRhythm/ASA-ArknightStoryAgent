"""Verify exactly the prepared patch landed; retain prior frozen snapshots."""
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
patch_dir = run / "canonical_context_guard_patch_v1"
prepared = json.loads((patch_dir / "manifest.json").read_text())
assert sha(patch_dir / "change.patch") == prepared["patch_sha256"]
for change in prepared["changes"]:
    assert sha(repo / change["path"]) == change["after_sha256"], change["path"]
    if change["existed_before"]:
        assert sha(patch_dir / "before" / change["path"]) == change["before_sha256"]
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert head == prepared["head"]
diff = subprocess.run(["git", "-C", str(repo), "diff", "--check"], capture_output=True, text=True)
assert diff.returncode == 0, diff.stdout + diff.stderr
manifest = json.loads((run / "manifest.json").read_text())
changes = {str(repo / r["path"]): r for r in prepared["changes"]}
verified = []
for relative, info in manifest["source_files"].items():
    if relative.startswith("after/"):
        expected = changes.get(info["origin"], {}).get("after_sha256", info["sha256"])
        assert sha(Path(info["origin"])) == expected, info["origin"]
        verified.append({
            "path": info["origin"], "after_sha256": expected,
            "changed_from_frozen": expected != info["sha256"],
        })
xml_path = run / "canonical_release_tests_context_guard_v1.xml"
suites = list(ET.parse(xml_path).getroot().iter("testsuite"))
tests = sum(int(s.get("tests", 0)) for s in suites)
assert tests == 98
assert all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0 for s in suites)
report = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "canonical_host": "${REMOTE_HOST}", "head": head, "applied": True,
    "service_restarted": False, "committed": False, "pushed": False, "training_started": False,
    "patch_sha256": prepared["patch_sha256"], "changes": prepared["changes"],
    "frozen_sources_rechecked": len(verified),
    "expected_modified_frozen_files": sum(r["changed_from_frozen"] for r in verified),
    "tests_passed": tests, "test_receipt_sha256": sha(xml_path),
    "verified_sources": verified,
    "scope": "Capacity integrity/observability only; no claim of semantic accuracy improvement.",
}
with (patch_dir / "applied_verified.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in {"verified_sources", "changes"}}))
