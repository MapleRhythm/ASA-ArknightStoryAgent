"""248-only token-rejection delta, checked before and after with169 CPU tests."""
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SOURCE = ROOT / "release/ASA-ArknightStoryAgent/src"
DEPS = ROOT / ".canonical_sync_20260908.lxnhWO/cpu_test_deps"
MODULE = "asa_arknight_story_agent.inference.grounding.token_checks"
RELATIVE = "asa_arknight_story_agent/inference/grounding/token_checks.py"
TARGETS = {
    "release/ASA-ArknightStoryAgent/src/" + RELATIVE: "token_checks.py",
    "release/ASA-ArknightStoryAgent/tests/test_token_rejection_display.py": "test_token_rejection_display.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def snapshot():
    return {
        "runtime": {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")},
        "dataset_scripts": {
            name: sha(ROOT / "scripts" / name)
            for name in (
                "build_soda_api_verifier_dataset.py", "clean_soda_api_verifier_dataset.py",
                "merge_soda_api_verifier_shards.py", "soda_dataset_integrity.py",
            )
        },
    }


def canonical_manifest():
    pre_path = HERE / "token_rejection_preapply_v1.json"
    pre, applied = load(pre_path), load(HERE / "token_rejection_applied_verified_v1.json")
    assert applied["preapply_sha256"] == sha(pre_path)
    assert applied["canonical_tests"] == {"tests": 325, "failures": 0, "errors": 0, "skipped": 0}
    assert applied["applied_sha256"] == pre["candidate_sha256"]
    assert set(pre["candidate_sha256"]) == set(TARGETS)
    assert all(sha(HERE / name) == pre["candidate_sha256"][path] for path, name in TARGETS.items())
    return pre


def run_tests(candidate):
    output = HERE / ("candidate_tests169_v1.xml" if candidate else "mirror_tests169_v1.xml")
    assert not output.exists() and DEPS.is_dir()
    sys.path[:0] = [str(DEPS), str(SOURCE)]
    if candidate:
        parent = importlib.import_module("asa_arknight_story_agent.inference.grounding")
        spec = importlib.util.spec_from_file_location(MODULE, HERE / "token_checks.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE] = module
        spec.loader.exec_module(module)
        setattr(parent, "token_checks", module)
    else:
        importlib.import_module(MODULE)
    expected = HERE / "token_checks.py" if candidate else SOURCE / RELATIVE
    assert Path(sys.modules[MODULE].__file__).resolve() == expected.resolve()
    paths = [
        ROOT / "tests" / name for name in (
            "test_soda_cleaner_integrity.py", "test_soda_followup_integrity.py",
            "test_soda_verifier_contract.py", "test_soda_verifier_merge_integrity.py",
        )
    ]
    paths += [
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_empty_normalized_quote.py",
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_empty_quote_diagnostic_terms.py",
        HERE / "test_token_rejection_display.py" if candidate else
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_token_rejection_display.py",
    ]
    assert all(path.is_file() for path in paths)
    import pytest
    code = pytest.main([
        "-q", "-p", "no:cacheprovider", "--tb=short",
        *(str(path) for path in paths), "--junitxml=" + str(output),
    ])
    suites = ET.parse(output).getroot().findall("testsuite")
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    assert code == 0 and counts == {"tests": 169, "failures": 0, "errors": 0, "skipped": 0}
    assert Path(sys.modules[MODULE].__file__).resolve() == expected.resolve()
    return {
        "counts": counts, "xml_sha256": sha(output), "module_origin": str(expected),
        "python_version": sys.version, "pytest_version": pytest.__version__,
        "dependencies_path": str(DEPS), "cpu_only": True,
    }


def main():
    mode = sys.argv[1]
    assert mode in {"candidate", "apply", "verify"}
    canonical = canonical_manifest()
    if mode == "candidate":
        assert not (HERE / "candidate_verified_v1.json").exists()
        assert sha(SOURCE / RELATIVE) == canonical["old_target_sha256"]
        assert not (ROOT / "release/ASA-ArknightStoryAgent/tests/test_token_rejection_display.py").exists()
        before = snapshot()
        tests = run_tests(True)
        assert snapshot() == before
        save(HERE / "candidate_verified_v1.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_applied_receipt_sha256": sha(HERE / "token_rejection_applied_verified_v1.json"),
            "mirror_before": before, "candidate_test": tests,
            "target_sha256": canonical["candidate_sha256"], "training_ready": False,
        })
    elif mode == "apply":
        assert not (HERE / "mirror_preapply_v1.json").exists()
        candidate = load(HERE / "candidate_verified_v1.json")
        assert snapshot() == candidate["mirror_before"]
        assert not (ROOT / "release/ASA-ArknightStoryAgent/tests/test_token_rejection_display.py").exists()
        backup = HERE / "token_checks.original.py"
        assert not backup.exists()
        shutil.copyfile(SOURCE / RELATIVE, backup)
        assert sha(backup) == canonical["old_target_sha256"]
        save(HERE / "mirror_preapply_v1.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mirror_before": snapshot(), "target_sha256": canonical["candidate_sha256"],
            "backup_path": str(backup), "backup_sha256": sha(backup),
        })
        for path, name in TARGETS.items():
            target = ROOT / path
            if "/src/" in path:
                assert sha(target) == canonical["old_target_sha256"]
            else:
                assert not target.exists()
            shutil.copyfile(HERE / name, target)
            assert sha(target) == canonical["candidate_sha256"][path]
    else:
        assert not (HERE / "mirror_applied_verified_v1.json").exists()
        pre = load(HERE / "mirror_preapply_v1.json")
        assert all(sha(ROOT / path) == value for path, value in pre["target_sha256"].items())
        assert sha(Path(pre["backup_path"])) == pre["backup_sha256"]
        expected = pre["mirror_before"]
        expected["runtime"][RELATIVE] = canonical["candidate_sha256"]["release/ASA-ArknightStoryAgent/src/" + RELATIVE]
        assert snapshot() == expected
        tests = run_tests(False)
        assert snapshot() == expected
        save(HERE / "mirror_applied_verified_v1.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_applied_receipt_sha256": sha(HERE / "token_rejection_applied_verified_v1.json"),
            "applied_sha256": pre["target_sha256"], "postapply_test": tests,
            "other_runtime_and_dataset_scripts_unchanged": True,
            "backup_path": pre["backup_path"], "backup_sha256": pre["backup_sha256"],
            "training_ready": False, "installed_packages": False,
            "models_or_data_modified_by_this_delta": False, "services_restarted": False,
            "limits": ["169 CPU tests, not certification of the full GPU training environment."],
        })
    print(json.dumps({"stage": mode, "completed": True, "training_ready": False}))


if __name__ == "__main__":
    main()
