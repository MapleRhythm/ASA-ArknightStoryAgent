"""248-only guarded delta: preserve original bytes, run192 CPU tests twice."""
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
MODULE = "asa_arknight_story_agent.inference.grounding.quote_validation"
RELATIVE = "asa_arknight_story_agent/inference/grounding/quote_validation.py"
TARGETS = {
    "release/ASA-ArknightStoryAgent/src/" + RELATIVE: "question_echo_quote_validation.py",
    "release/ASA-ArknightStoryAgent/tests/test_question_echo_grounding.py": "test_question_echo_grounding.py",
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
            for name in ("build_soda_api_verifier_dataset.py", "clean_soda_api_verifier_dataset.py",
                         "merge_soda_api_verifier_shards.py", "soda_dataset_integrity.py")
        },
    }


def canonical():
    prepath = HERE / "question_echo_preapply_v1.json"
    pre, applied = load(prepath), load(HERE / "question_echo_applied_verified_v1.json")
    assert applied["preapply_sha256"] == sha(prepath)
    assert applied["canonical_tests"] == {"tests": 348, "failures": 0, "errors": 0, "skipped": 0}
    assert applied["applied_sha256"] == pre["candidate_sha256"]
    assert set(TARGETS) == set(pre["candidate_sha256"])
    assert all(sha(HERE / name) == pre["candidate_sha256"][path] for path, name in TARGETS.items())
    return pre


def run_tests(candidate):
    xml = HERE / ("candidate_tests192_v1.xml" if candidate else "mirror_tests192_v1.xml")
    assert not xml.exists() and DEPS.is_dir()
    sys.path[:0] = [str(DEPS), str(SOURCE)]
    if candidate:
        parent = importlib.import_module("asa_arknight_story_agent.inference.grounding")
        spec = importlib.util.spec_from_file_location(MODULE, HERE / "question_echo_quote_validation.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE] = module
        spec.loader.exec_module(module)
        setattr(parent, "quote_validation", module)
    else:
        importlib.import_module(MODULE)
    expected = HERE / "question_echo_quote_validation.py" if candidate else SOURCE / RELATIVE
    assert Path(sys.modules[MODULE].__file__).resolve() == expected.resolve()
    files = [ROOT / "tests" / name for name in (
        "test_soda_cleaner_integrity.py", "test_soda_followup_integrity.py",
        "test_soda_verifier_contract.py", "test_soda_verifier_merge_integrity.py",
    )]
    files += [ROOT / "release/ASA-ArknightStoryAgent/tests" / name for name in (
        "test_empty_normalized_quote.py", "test_empty_quote_diagnostic_terms.py",
        "test_token_rejection_display.py",
    )]
    files.append(HERE / "test_question_echo_grounding.py" if candidate else
                 ROOT / "release/ASA-ArknightStoryAgent/tests/test_question_echo_grounding.py")
    assert all(p.is_file() for p in files)
    import pytest
    code = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short",
                       *(str(p) for p in files), "--junitxml=" + str(xml)])
    suites = ET.parse(xml).getroot().findall("testsuite")
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
              for k in ("tests", "failures", "errors", "skipped")}
    assert code == 0 and totals == {"tests": 192, "failures": 0, "errors": 0, "skipped": 0}
    assert Path(sys.modules[MODULE].__file__).resolve() == expected.resolve()
    return {"counts": totals, "xml_sha256": sha(xml), "module_origin": str(expected),
            "python_version": sys.version, "pytest_version": pytest.__version__, "cpu_only": True}


def main():
    mode = sys.argv[1]
    assert mode in {"candidate", "apply", "verify"}
    pre = canonical()
    expected_before = {"runtime": pre["canonical_source_hashes"],
                       "dataset_scripts": pre["dataset_scripts_sha256"]}
    if mode == "candidate":
        assert snapshot() == expected_before
        assert not (ROOT / "release/ASA-ArknightStoryAgent/tests/test_question_echo_grounding.py").exists()
        tests = run_tests(True)
        assert snapshot() == expected_before
        save(HERE / "candidate_verified_v1.json", {
            "mirror_before": expected_before, "candidate_test": tests,
            "canonical_applied_receipt_sha256": sha(HERE / "question_echo_applied_verified_v1.json"),
            "training_ready": False,
        })
    elif mode == "apply":
        assert load(HERE / "candidate_verified_v1.json")["mirror_before"] == expected_before
        assert snapshot() == expected_before
        assert not (ROOT / "release/ASA-ArknightStoryAgent/tests/test_question_echo_grounding.py").exists()
        backup = HERE / "quote_validation.original.py"
        assert not backup.exists()
        shutil.copyfile(SOURCE / RELATIVE, backup)
        assert sha(backup) == pre["old_target_sha256"]
        save(HERE / "mirror_preapply_v1.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mirror_before": expected_before, "target_sha256": pre["candidate_sha256"],
            "backup_path": str(backup), "backup_sha256": sha(backup),
        })
        for target, name in TARGETS.items():
            path = ROOT / target
            assert not path.is_symlink()
            if "/src/" in target:
                assert sha(path) == pre["old_target_sha256"]
            else:
                assert not path.exists()
            shutil.copyfile(HERE / name, path)
            assert sha(path) == pre["candidate_sha256"][target]
    else:
        applied_pre = load(HERE / "mirror_preapply_v1.json")
        expected = {"runtime": pre["candidate_source_hashes"],
                    "dataset_scripts": pre["dataset_scripts_sha256"]}
        assert snapshot() == expected
        assert sha(Path(applied_pre["backup_path"])) == applied_pre["backup_sha256"]
        assert all(sha(ROOT / p) == h for p, h in pre["candidate_sha256"].items())
        tests = run_tests(False)
        assert snapshot() == expected
        save(HERE / "mirror_applied_verified_v1.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_applied_receipt_sha256": sha(HERE / "question_echo_applied_verified_v1.json"),
            "applied_sha256": pre["candidate_sha256"], "postapply_test": tests,
            "full_runtime_python_matches241": True, "four_dataset_scripts_match241": True,
            "backup_path": applied_pre["backup_path"], "backup_sha256": applied_pre["backup_sha256"],
            "training_ready": False, "installed_packages": False, "services_restarted": False,
            "models_or_data_modified": False,
            "limitations": ["192 CPU subset tests, not full GPU training environment certification."],
        })
    print(json.dumps({"stage": mode, "completed": True, "training_ready": False}), flush=True)


if __name__ == "__main__":
    main()
