"""Verify installed scope against the exact source used in the paired replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
RELEASE = ROOT / "release/ASA-ArknightStoryAgent"
SOURCE = RELEASE / "src"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode()


def counts(path):
    suites = ET.parse(path).getroot().findall("testsuite")
    return {k: sum(int(s.attrib.get(k, 0)) for s in suites)
            for k in ("tests", "failures", "errors", "skipped")}


replay = load(HERE / "fact_scope_replay_comparison_v1.json")
stage = sys.argv[1]
assert git("rev-parse", "HEAD").strip() == "0d594a3c9fd641fd8d7f798c8ca6b07db6c7020b"
assert all(not r["action_changed_ids"] and not r["answer_changed_ids"] for r in replay["comparisons"])
if stage == "before":
    assert counts(HERE / "candidate_fact_scope_tests_v1.xml") == {
        "tests": 128, "failures": 0, "errors": 0, "skipped": 0,
    }
    for relative, digest in replay["canonical_source_hashes"].items():
        assert sha(SOURCE / relative) == digest, relative
    for relative, digest in replay["candidate_source_hashes"].items():
        assert sha(HERE / "candidate/src" / relative) == digest, relative
    assert not (RELEASE / "tests/test_fact_scoped_quotes.py").exists()
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "head": git("rev-parse", "HEAD").strip(),
        "replay_sha256": sha(HERE / "fact_scope_replay_comparison_v1.json"),
        "test_source_sha256": sha(HERE / "test_fact_scoped_quotes.py"),
        "prior_dirty_sha256": {
            path: sha(ROOT / path) for path in git("diff", "--name-only").splitlines()
            if (ROOT / path).is_file()
        },
    }
    target = HERE / "preapply_manifest.json"
elif stage == "after":
    before = load(HERE / "preapply_manifest.json")
    for relative, digest in replay["candidate_source_hashes"].items():
        assert sha(SOURCE / relative) == digest, relative
    changed = {"release/ASA-ArknightStoryAgent/src/" + p for p in replay["changed_source_files"]}
    for relative, digest in before["prior_dirty_sha256"].items():
        if relative not in changed:
            assert sha(ROOT / relative) == digest, relative
    assert sha(RELEASE / "tests/test_fact_scoped_quotes.py") == before["test_source_sha256"]
    result = counts(HERE / "canonical_fact_scope_tests_v1.xml")
    assert result == {"tests": 186, "failures": 0, "errors": 0, "skipped": 0}
    for case in replay["comparisons"]:
        assert sha(Path(case["matrix"])) == case["source_sha256"]["matrix"]
        assert sha(Path(case["predictions"])) == case["source_sha256"]["predictions"]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "head": git("rev-parse", "HEAD").strip(), "canonical_tests": result,
        "modified_source_files": replay["changed_source_files"],
        "new_test": "release/ASA-ArknightStoryAgent/tests/test_fact_scoped_quotes.py",
        "source_files_checked": len(replay["candidate_source_hashes"]),
        "other_dirty_content_unchanged": True,
        "frozen_eval_inputs_unchanged": True,
        "action_or_answer_changes_in_183_replay_states": 0,
        "services_restarted": False, "training_started": False,
        "model_changed": False, "training_ready": False,
    }
    target = HERE / "applied_verified.json"
else:
    raise ValueError(stage)
with target.open("x", encoding="utf-8") as output:
    json.dump(payload, output, ensure_ascii=False, indent=2)
print(json.dumps({"manifest": str(target), "stage": stage,
                  "tests": payload.get("canonical_tests")}))
