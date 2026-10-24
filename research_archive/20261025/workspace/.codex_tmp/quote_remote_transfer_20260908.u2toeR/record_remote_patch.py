"""Record a server-based patch, preserving and checking unrelated source edits."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
RELEASE = REPO / "release/ASA-ArknightStoryAgent"
SOURCE = RELEASE / "src/asa_arknight_story_agent"
RELATIVE_FILES = (
    "evidence/rendering.py",
    "generation/minimal_conclusion_prompt.py",
    "generation/direct_answer_prompt.py",
    "generation/conclusion_prompt_rendering.py",
    "generation/answer_conclusion_generation.py",
    "generation/direct_answer_generation.py",
    "grounding/quote_validation.py",
    "grounding/evidence_id_validation.py",
    "grounding/quote_checks.py",
    "grounding/validation.py",
)
TEST_FILES = (
    "test_quote_complete_evidence_rendering.py",
    "test_grounding_visible_evidence.py",
    "test_conclusion_sampling_failures.py",
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    paths = set()
    for folder in (SOURCE, RELEASE / "configs", RELEASE / "tests"):
        paths.update(p for p in folder.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix in {".py", ".json", ".yaml", ".yml"})
    for name in git("diff", "--name-only", "-z").decode().split("\0"):
        path = REPO / name
        if name and path.is_file():
            paths.add(path)
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "check-before", "after"))
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    before_path = run / "before.json"
    source_paths = [str((SOURCE / "inference" / name).relative_to(REPO)) for name in RELATIVE_FILES]
    test_paths = [str((RELEASE / "tests" / name).relative_to(REPO)) for name in TEST_FILES]
    common = {
        "run_id": run.name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "canonical_host": "${REMOTE_HOST}",
        "repository": str(REPO),
        "release_root": str(RELEASE),
        "canonical_working_root": str(run.parent.parent),
        "branch": git("branch", "--show-current").decode().strip(),
        "head": git("rev-parse", "HEAD").decode().strip(),
        "git_status": git("status", "--porcelain=v1").decode(),
        "sha256": inventory(),
        "patch_sha256": sha(run / "quote_complete_and_visible_grounding.patch"),
        "roles_sha256": sha(run.parent.parent / "REPOSITORY_ROLES.md"),
        "recording_script_sha256": sha(Path(__file__).resolve()),
    }
    if args.mode == "before":
        if before_path.exists():
            raise SystemExit("refusing to overwrite baseline")
        dirty = set(git("diff", "--name-only", "-z").decode().split("\0"))
        if dirty.intersection(source_paths) or any((REPO / p).exists() for p in test_paths):
            raise SystemExit("patch target overlaps pre-existing edits or tests")
        subprocess.run(
            ["tar", "--exclude=__pycache__", "-czf", str(run / "before.tar.gz"),
             "-C", str(RELEASE), "src/asa_arknight_story_agent", "configs", "tests"],
            check=True,
        )
        common["backup_sha256"] = sha(run / "before.tar.gz")
        common["allowed_source_changes"] = source_paths
        common["allowed_test_additions"] = test_paths
        before_path.write_text(json.dumps(common, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"stage": "before", "head": common["head"],
                          "protected_files": len(common["sha256"]),
                          "manifest": str(before_path)}))
        return
    before = json.loads(before_path.read_text())
    if common["head"] != before["head"] or common["branch"] != before["branch"]:
        raise SystemExit("repository branch or HEAD changed")
    if common["patch_sha256"] != before["patch_sha256"]:
        raise SystemExit("patch changed after baseline")
    initial_dirty = {
        line[3:] for line in before["git_status"].splitlines()
        if line[:2] != "??"
    }
    if initial_dirty.intersection(source_paths):
        raise SystemExit("actual patch targets overlap pre-existing edits")
    old, new = before["sha256"], common["sha256"]
    changed = {p for p in old if old[p] != new.get(p)}
    added = new.keys() - old.keys()
    if args.mode == "check-before":
        if changed or added or common["git_status"] != before["git_status"]:
            raise SystemExit("remote source changed after baseline")
        print("Baseline hashes and git status still match; patch may be checked/applied.")
        return
    if changed != set(source_paths) or added != set(test_paths):
        raise SystemExit(f"unexpected change set: changed={sorted(changed)}, added={sorted(added)}")
    common.update({
        "changed_files": sorted(changed),
        "added_files": sorted(added),
        "unchanged_protected_files": len(old) - len(changed),
        "unrelated_source_config_and_existing_edits_preserved": True,
        "before_manifest_sha256": sha(before_path),
        "deployed_service_restarted": False,
        "answer_accuracy_evaluated": False,
        "allowed_source_changes": source_paths,
        "baseline_allowlist_corrected": before["allowed_source_changes"] != source_paths,
        "baseline_allowlist_note": (
            "The original before.json omitted inference/ in its descriptive target list. "
            "Its complete file hashes and git status remain unchanged and authoritative. "
            "This verification uses the actual ten inference paths and independently "
            "checks they did not overlap the recorded pre-existing edits."
        ),
    })
    after_path = run / "after.json"
    if after_path.exists():
        raise SystemExit("refusing to overwrite after manifest")
    after_path.write_text(json.dumps(common, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "stage": "after", "changed_files": len(changed), "added_tests": len(added),
        "unchanged_protected_files": common["unchanged_protected_files"],
        "manifest": str(after_path),
    }))


if __name__ == "__main__":
    main()
