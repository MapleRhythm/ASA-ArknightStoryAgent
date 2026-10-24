"""Fresh-process canonical regression checks after the guarded one-file edit."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
RELEASE = ROOT / "release/ASA-ArknightStoryAgent"
HERE = Path(__file__).resolve().parent
REL_MODULE = "asa_arknight_story_agent/inference/grounding/grounded_fact_answers.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = HERE / "installed_verification_v1.json"
    assert not receipt.exists()
    prior = json.loads((HERE / "isolated_replay_v1.json").read_text())
    expected = dict(prior["source_sha256"])
    module = RELEASE / "src" / REL_MODULE
    expected[str(module)] = sha(HERE / "grounded_fact_answers.py")
    assert all(sha(Path(p)) == value for p, value in expected.items())
    assert sha(RELEASE / "tests/test_grounded_fact_layout_regression.py") == sha(
        HERE / "test_grounded_fact_layout_regression.py")
    configs = [
        path for folder in (ROOT / "configs", ROOT / "api-mode", RELEASE / "configs")
        for path in sorted(folder.glob("*.json"))
        if isinstance(json.loads(path.read_text()).get("retrieval"), dict)
    ]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(RELEASE / "src"),
        ASA_NO_MINIRAG_CONFIG=str(RELEASE / "configs/runtime_gpu_reranker_qwen35_4b.json"),
        ASA_NO_MINIRAG_CONFIGS=os.pathsep.join(map(str, configs)),
        ASA_NO_MINIRAG_RETRIEVERS=os.pathsep.join(map(str, (
            ROOT / "src/goldenglow/retrieval/hybrid.py",
            RELEASE / "src/asa_arknight_story_agent/retrieval/hybrid.py",
        ))),
        ASA_NO_MINIRAG_RUNNER=str(RELEASE / "scripts/run_cpu_inference.py"),
        ASA_ROOT_SRC=str(ROOT / "src"),
        ASA_RUN_CPU_INFERENCE_SCRIPT=str(RELEASE / "scripts/run_cpu_inference.py"),
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "gpu_used": False,
        "only_expected_existing_release_source_changed": True,
        "baseline_source_sha256": prior["source_sha256"][str(module)],
        "installed_source_sha256": sha(module),
        "backup": str(HERE / "baseline_grounded_fact_answers.py"),
        "config_sha256": {str(p): sha(p) for p in configs},
        "script_sha256": sha(Path(__file__)), "tests": {},
        "semantic_accuracy_gain_claimed": False,
        "model_pointer_or_services_changed": False,
    }
    for name, cwd in (("release", RELEASE), ("root", ROOT)):
        xml, log = HERE / f"{name}_tests_v1.xml", HERE / f"{name}_tests_v1.log"
        assert not xml.exists() and not log.exists()
        test_env = dict(env)
        test_env["PYTHONPATH"] = str(cwd / "src")
        command = [sys.executable, "-m", "pytest", "-q", "tests", f"--junitxml={xml}"]
        with log.open("x") as stream:
            status = subprocess.run(command, cwd=cwd, env=test_env,
                                    stdout=stream, stderr=subprocess.STDOUT).returncode
        suites = ET.parse(xml).getroot()
        counts = {key: sum(int(s.get(key, 0)) for s in suites.iter("testsuite"))
                  for key in ("tests", "failures", "errors", "skipped")}
        report["tests"][name] = {
            "command": command, "cwd": str(cwd), "exit_code": status, "counts": counts,
            "xml_sha256": sha(xml), "log_sha256": sha(log),
        }
        print(json.dumps({name: report["tests"][name]}, ensure_ascii=False), flush=True)
    assert all(sha(Path(p)) == value for p, value in expected.items())
    assert report["config_sha256"] == {str(p): sha(p) for p in configs}
    report["all_passed"] = all(
        test["exit_code"] == 0 and not any(test["counts"][key] for key in ("failures", "errors", "skipped"))
        for test in report["tests"].values()
    )
    with receipt.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
