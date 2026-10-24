"""Record guarded installation against the frozen canonical query-state replay."""
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
    return {key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")}


replay = load(HERE / "diagnostic_terms_replay_comparison_v2.json")
assert git("rev-parse", "HEAD").strip() == "0d594a3c9fd641fd8d7f798c8ca6b07db6c7020b"
assert all(not c["action_changed_ids"] and not c["answer_changed_ids"] for c in replay["comparisons"])
assert not replay["state_replay"]["new_normalization_errors"]
stage = sys.argv[1]
if stage == "before":
    assert counts(HERE / "candidate_diagnostic_terms_tests_v2.xml") == {
        "tests": 156, "failures": 0, "errors": 0, "skipped": 0,
    }
    for relative, digest in replay["canonical_source_hashes"].items():
        assert sha(SOURCE / relative) == digest, relative
    for relative, digest in replay["candidate_source_hashes"].items():
        assert sha(HERE / "candidate/src" / relative) == digest, relative
        if relative not in replay["canonical_source_hashes"]:
            assert not (SOURCE / relative).exists(), relative
    assert not (RELEASE / "tests/test_retrieval_diagnostic_terms.py").exists()
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "head": git("rev-parse", "HEAD").strip(),
        "replay_sha256": sha(HERE / "diagnostic_terms_replay_comparison_v2.json"),
        "test_sha256": sha(HERE / "test_retrieval_diagnostic_terms.py"),
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
    assert sha(RELEASE / "tests/test_retrieval_diagnostic_terms.py") == before["test_sha256"]
    result = counts(HERE / "canonical_diagnostic_terms_tests_v1.xml")
    assert result == {"tests": 214, "failures": 0, "errors": 0, "skipped": 0}
    for path, digest in replay["state_replay"]["source_files"].items():
        assert sha(Path(path)) == digest, path
    for case in replay["comparisons"]:
        assert sha(Path(case["matrix"])) == case["matrix_sha256"]
        assert sha(Path(case["predictions"])) == case["predictions_sha256"]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "head": git("rev-parse", "HEAD").strip(), "canonical_tests": result,
        "changed_source_files": replay["changed_source_files"],
        "new_test": "release/ASA-ArknightStoryAgent/tests/test_retrieval_diagnostic_terms.py",
        "source_files_checked": len(replay["candidate_source_hashes"]),
        "other_dirty_content_unchanged": True, "frozen_inputs_unchanged": True,
        "historical_hypothesis_states_changed": replay["state_replay"]["changes_by_source"].get("historical_training_input"),
        "action_or_answer_changes_in_183_replay_states": 0,
        "services_restarted": False, "model_changed": False, "training_started": False,
        "training_ready": False,
    }
    target = HERE / "applied_verified.json"
else:
    raise ValueError(stage)
with target.open("x", encoding="utf-8") as output:
    json.dump(payload, output, ensure_ascii=False, indent=2)
print(json.dumps({"manifest": str(target), "stage": stage, "tests": payload.get("canonical_tests")}))
