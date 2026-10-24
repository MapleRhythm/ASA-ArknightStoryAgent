"""Baseline broad tests with the existing v33/no-MiniRAG environment contract.

Preserve the first collection-error attempt. Never skip tests or modify config.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import run_trace_retention_candidate as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared
ROOT = probe.probe.reader.ROOT
RELEASE = ROOT / "release/ASA-ArknightStoryAgent"


def main():
    pre = shared.load(HERE / "preflight.json")
    probe.verify(pre)
    receipt = HERE / "canonical_release_baseline_attempt02.json"
    xml = HERE / "canonical_release_baseline_tests_attempt02.xml"
    log = HERE / "canonical_release_baseline_tests_attempt02.log"
    assert not any(p.exists() for p in (receipt, xml, log))
    first = [HERE / "canonical_release_baseline_tests_v1.xml", HERE / "canonical_release_baseline_tests_v1.log"]
    assert "ASA_NO_MINIRAG_RETRIEVERS" in first[1].read_text()
    configs = [
        path for folder in (ROOT / "configs", ROOT / "api-mode", RELEASE / "configs")
        for path in sorted(folder.glob("*.json"))
        if isinstance(shared.load(path).get("retrieval"), dict)
    ]
    retrievers = [
        ROOT / "src/goldenglow/retrieval/hybrid.py",
        RELEASE / "src/asa_arknight_story_agent/retrieval/hybrid.py",
    ]
    runner = RELEASE / "scripts/run_cpu_inference.py"
    test_sources = sorted((RELEASE / "tests").rglob("*.py"))
    binding = {str(p): shared.sha(p) for p in [*configs, *retrievers, runner, *test_sources]}
    env = os.environ.copy()
    overrides = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join((str(RELEASE / "src"), str(ROOT / ".python_packages/train"))),
        "ASA_NO_MINIRAG_CONFIG": str(RELEASE / "configs/runtime_gpu_reranker_qwen35_4b.json"),
        "ASA_NO_MINIRAG_CONFIGS": os.pathsep.join(map(str, configs)),
        "ASA_NO_MINIRAG_RETRIEVERS": os.pathsep.join(map(str, retrievers)),
        "ASA_NO_MINIRAG_RUNNER": str(runner),
        "ASA_ROOT_SRC": str(ROOT / "src"),
        "ASA_RUN_CPU_INFERENCE_SCRIPT": str(runner),
    }
    env.update(overrides)
    command = [sys.executable, "-m", "pytest", "-q", "tests", f"--junitxml={xml}"]
    with log.open("x") as stream:
        status = subprocess.run(command, cwd=RELEASE, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    suites = list(ET.parse(xml).getroot().iter("testsuite"))
    counts = {k: sum(int(s.attrib.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    probe.verify(pre)
    assert all(shared.sha(Path(p)) == h for p, h in binding.items())
    shared.save(receipt, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "cwd": str(RELEASE), "environment_overrides": overrides,
        "exit_code": status, "counts": counts,
        "xml_sha256": shared.sha(xml), "log_sha256": shared.sha(log),
        "preserved_first_collection_error_attempt": {str(p): shared.sha(p) for p in first},
        "config_runner_retriever_tests_unchanged_sha256": binding,
        "script_sha256": shared.sha(Path(__file__)),
        "gpu_disabled": True, "production_changed": False, "tests_skipped_intentionally": False,
    })
    print({"release_baseline_attempt02": counts, "exit_code": status, "gpu_disabled": True}, flush=True)
    if status != 0 or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
