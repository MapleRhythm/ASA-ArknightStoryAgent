"""Reproducible CPU verification; never load models or rewrite runtime files."""
from __future__ import annotations

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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt_path = HERE / "cpu_verification_v2.json"
    if receipt_path.exists():
        raise RuntimeError("verification receipt already exists")
    configs = [
        path for folder in [ROOT / "configs", ROOT / "api-mode", RELEASE / "configs"]
        for path in sorted(folder.glob("*.json"))
        if isinstance(json.loads(path.read_text()).get("retrieval"), dict)
    ]
    retrievers = [
        ROOT / "src/goldenglow/retrieval/hybrid.py",
        RELEASE / "src/asa_arknight_story_agent/retrieval/hybrid.py",
    ]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(RELEASE / "src"),
        ASA_NO_MINIRAG_CONFIG=str(RELEASE / "configs/runtime_gpu_reranker_qwen35_4b.json"),
        ASA_NO_MINIRAG_CONFIGS=os.pathsep.join(map(str, configs)),
        ASA_NO_MINIRAG_RETRIEVERS=os.pathsep.join(map(str, retrievers)),
        ASA_NO_MINIRAG_RUNNER=str(RELEASE / "scripts/run_cpu_inference.py"),
        ASA_ROOT_SRC=str(ROOT / "src"),
        ASA_RUN_CPU_INFERENCE_SCRIPT=str(RELEASE / "scripts/run_cpu_inference.py"),
    )
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__)), "gpu_used": False,
        "config_sha256": {str(path): sha(path) for path in configs},
        "tests": {},
    }
    # Separate processes prevent the root tests' sys.path changes from
    # contaminating imports in the release package.
    for name, directory in [("release", RELEASE), ("root", ROOT)]:
        xml = HERE / f"{name}_tests_v2.xml"
        log = HERE / f"{name}_tests_v2.log"
        if xml.exists() or log.exists():
            raise RuntimeError("test artifacts already exist")
        command = [sys.executable, "-m", "pytest", "-q", "tests", f"--junitxml={xml}"]
        test_env = dict(env)
        if name == "root":
            test_env["PYTHONPATH"] = str(ROOT / "src")
        with log.open("x") as stream:
            status = subprocess.run(command, cwd=directory, env=test_env,
                                    stdout=stream, stderr=subprocess.STDOUT).returncode
        suites = ET.parse(xml).getroot()
        counts = {
            key: sum(int(suite.get(key, 0)) for suite in suites.iter("testsuite"))
            for key in ("tests", "failures", "errors", "skipped")
        }
        result["tests"][name] = {
            "command": command, "cwd": str(directory), "exit_code": status,
            "counts": counts, "xml_sha256": sha(xml), "log_sha256": sha(log),
        }
        print(json.dumps({name: result["tests"][name]}, ensure_ascii=False), flush=True)
    result["all_passed"] = all(
        test["exit_code"] == 0 and not any(test["counts"][key] for key in ("failures", "errors", "skipped"))
        for test in result["tests"].values()
    )
    with receipt_path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
