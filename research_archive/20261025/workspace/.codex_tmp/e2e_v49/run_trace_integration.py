"""Guarded one-source-file integration and fresh-process postapply checks."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
V45 = HERE.parent / "staged_round2_reader_20260910.v45"
V47 = HERE.parent / "fallback_trace_completeness_20260910.v47"
V48 = HERE.parent / "fallback_trace_retention_candidate_20260910.v48"
sys.path[:0] = [str(V45), str(V47)]
import run_second_round_reader as probe
import run_fallback_trace_audit as audit
shared = probe.shared
ROOT = probe.reader.ROOT
RELEASE = ROOT / "release/ASA-ArknightStoryAgent"
SOURCE = RELEASE / "src/asa_arknight_story_agent/inference/generation/answer_conclusion_generation.py"
TEST = RELEASE / "tests/test_final_direct_trace_retention.py"
FIELD = "final_direct_fallback_attempts"


def functions(path):
    return {n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def prepare():
    assert not (HERE / "preflight.json").exists() and not TEST.exists()
    old = shared.load(V45 / "preflight.json")
    probe.reader.verify(old)
    closed = shared.load(V48 / "checkpoint_v48.json")
    baseline = shared.load(V48 / "canonical_release_baseline_attempt02.json")
    assert closed["legacy_behavior_changes"] == 0 and closed["actual_state_pairs"] == 14
    assert closed["candidate_unit_tests"] == 20 and closed["baseline_release_tests"] == 306
    assert shared.sha(SOURCE) == closed["original_source_sha256"] == shared.sha(V48 / "answer_conclusion_generation_original.py")
    original = functions(V48 / "answer_conclusion_generation_original.py")
    proposed = functions(HERE / SOURCE.name)
    tested = functions(V48 / "answer_conclusion_generation_candidate.py")
    assert set(proposed) == set(original) | {"record_final_direct_attempt"}
    assert proposed["record_final_direct_attempt"] == tested["record_final_direct_attempt"]
    assert proposed["generate_conclusion_from_model"] == tested["generate_conclusion_from_model"]
    for name in original.keys() - {"generate_conclusion_from_model"}:
        assert proposed[name] == original[name]
    assert functions(HERE / TEST.name) == functions(V48 / "test_fallback_trace_candidate.py")
    legacy = {**old["sha256"], **baseline["config_runner_retriever_tests_unchanged_sha256"]}
    assert all(shared.sha(Path(p)) == h for p, h in legacy.items())
    paths = [
        Path(__file__), HERE / SOURCE.name, HERE / TEST.name, HERE / "README.md",
        V48 / "answer_conclusion_generation_original.py", V48 / "answer_conclusion_generation_candidate.py",
        V48 / "test_fallback_trace_candidate.py", V48 / "checkpoint_v48.json",
        V48 / "canonical_release_baseline_attempt02.json",
        V45 / "preflight.json", V45 / "generations.json", V45 / "primary_fallback_audit_v1.json",
        V45 / "checkpoint_v45.json", V45 / "run_second_round_reader.py",
        probe.V44 / "retrieval.json", V47 / "audit_v1.json", V47 / "run_fallback_trace_audit.py",
    ]
    shared.save(HERE / "preflight.json", {
        "version": "guarded_final_direct_trace_retention_integration_v49",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_sha256": legacy, "artifact_sha256": {str(p): shared.sha(p) for p in paths},
        "existing_source_target": str(SOURCE), "new_test_target": str(TEST),
        "baseline_source_sha256": shared.sha(SOURCE), "new_source_sha256": shared.sha(HERE / SOURCE.name),
        "new_test_sha256": shared.sha(HERE / TEST.name),
        "canonical_preimage": str(V48 / "answer_conclusion_generation_original.py"),
        "environment_overrides": baseline["environment_overrides"],
        "ast_unchanged_helpers_and_tested_candidate_equivalence": True,
        "protocol": [
            "One existing release source file and one absent regression-test target only; preserve all other dirty files.",
            "Only new append-only final_direct_fallback_attempts sidecar at final-direct fallback success/failure.",
            "No source copy is authorized unless every baseline source/model/config/runner/retriever/test hash still matches.",
            "Fresh process after application verifies installed source path, full326 release tests and14 actual cached states.",
            "Strict comparison strips only the new sidecar; all old trace fields, conclusions, final answers, prompts, calls and kwargs must match.",
            "Ten first-round terminals unchanged; two synthetic failure fixtures retain old output/stdout and add recorded error.",
            "No new LLM/GPU/retrieval, default model, validator, evidence binding, labels, training, service restart, commit or push.",
            "Runtime source patch does not imply an already-running service has hot-loaded it, and is not a semantic accuracy gain.",
        ],
        "production_changed_at_preflight": False, "training_ready": False,
    })
    print({"old_source": shared.sha(SOURCE), "new_source": shared.sha(HERE / SOURCE.name),
           "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


def verify(pre, installed):
    assert all(shared.sha(Path(p)) == h for p, h in pre["artifact_sha256"].items())
    for path, expected in pre["legacy_sha256"].items():
        if path == str(SOURCE) and installed:
            expected = pre["new_source_sha256"]
        assert shared.sha(Path(path)) == expected, path
    if installed:
        assert shared.sha(TEST) == pre["new_test_sha256"]
    else:
        assert not TEST.exists()


def guard():
    pre = shared.load(HERE / "preflight.json")
    assert not (HERE / "preapply_guard_v1.json").exists()
    verify(pre, installed=False)
    shared.save(HERE / "preapply_guard_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "existing_source_sha256": shared.sha(SOURCE), "new_test_absent": True,
        "all_unrelated_source_models_configs_tests_unchanged": True,
        "authorized_targets": [str(SOURCE), str(TEST)],
    })
    print("Exact pre-image and unrelated-file guards passed.", flush=True)


def projection(result, source):
    result = deepcopy(result)
    sidecars = result["retrieval_trace"][-1].pop(FIELD, [])
    assert all(FIELD not in step for step in result["retrieval_trace"][:-1])
    state = deepcopy(source["state_after_retrieval"])
    state["retrieval_trace"] = result["retrieval_trace"]
    state["final_answer"] = result["final_answer"]
    result["state_after_reader_sha256"] = probe.staged.object_digest(state)
    return result, sidecars


def check():
    pre = shared.load(HERE / "preflight.json")
    assert not (HERE / "installed_verification_v1.json").exists()
    verify(pre, installed=True)
    assert shared.load(HERE / "preapply_guard_v1.json")["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    xml, log = HERE / "release_tests_v1.xml", HERE / "release_tests_v1.log"
    assert not xml.exists() and not log.exists()
    env = os.environ.copy()
    env.update(pre["environment_overrides"])
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    command = [sys.executable, "-m", "pytest", "-q", "tests", f"--junitxml={xml}"]
    with log.open("x") as stream:
        status = subprocess.run(command, cwd=RELEASE, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    counts = {k: sum(int(s.attrib.get(k, "0")) for s in ET.parse(xml).getroot().iter("testsuite"))
              for k in ("tests", "failures", "errors", "skipped")}
    print({"installed_release_tests": counts, "exit_code": status}, flush=True)
    assert status == 0 and counts == {"tests": 326, "failures": 0, "errors": 0, "skipped": 0}
    old = shared.load(V45 / "preflight.json")
    saved = shared.load(V45 / "generations.json")
    helpers = probe.staged.imports()
    from asa_arknight_story_agent.inference.generation import answer_conclusion_generation as installed
    assert Path(installed.__file__).resolve() == SOURCE.resolve()
    assert installed.generate_conclusion_from_model.__code__.co_filename == str(SOURCE)
    assert hasattr(installed, "record_final_direct_attempt")
    frozen = probe.sources()
    primary = {(r["pair_id"], r["condition"]): r for r in shared.load(V45 / "primary_fallback_audit_v1.json")["rows"]}
    rows, carry = [], 0
    for item, result in zip(old["records"], saved["records"], strict=True):
        key = item["pair_id"], item["condition"]
        if item["terminal_after_round1"]:
            assert result["final_answer"] == item["carried_terminal_answer"]
            assert result["conclusion"] == item["carried_conclusion"] and not result["requests"]
            carry += 1
            continue
        generator = probe.ReplayGenerator(old["generator"], item, result)
        actual = probe.invoke(old, helpers, item, frozen[key], generator)
        projected, sidecars = projection(actual, frozen[key])
        assert all(result[k] == v for k, v in projected.items())
        assert generator.calls == len(result["requests"])
        if len(result["requests"]) == 2:
            assert len(sidecars) == 1
            assert sidecars[0]["primary_conclusions"] == [primary[key]["primary_conclusion"]]
            assert sidecars[0]["fallback"] == {"status": "ok", "conclusion": result["conclusion"]}
        else:
            assert not sidecars
        rows.append({
            "pair_id": key[0], "condition": key[1], "legacy_behavior_unchanged": True,
            "calls_replayed": generator.calls, "sidecars": sidecars,
            "projected_state_sha256": projected["state_after_reader_sha256"],
        })
    assert len(rows) == 14 and carry == 10
    key = "fixed58-010:seed0", "control"
    item = next(r for r in old["records"] if (r["pair_id"], r["condition"]) == key)
    result = next(r for r in saved["records"] if (r["pair_id"], r["condition"]) == key)
    old_synthetic = shared.load(V47 / "audit_v1.json")["synthetic_failure_protocol_cases"]
    synthetic = []
    for fail_primary, before in zip((False, True), old_synthetic, strict=True):
        generator = audit.FailingFallback(old["generator"], item, result, fail_primary)
        output = io.StringIO()
        with redirect_stdout(output):
            actual = probe.invoke(old, helpers, item, frozen[key], generator)
        projected, sidecars = projection(actual, frozen[key])
        assert probe.staged.object_digest(projected["retrieval_trace"]) == before["trace_sha256"]
        assert projected["conclusion"]["next_action"] == before["final_action"]
        assert projected["final_answer"] == before["final_answer"]
        assert output.getvalue() == before["captured_stdout"]
        assert generator.requests == before["requests"] and generator.calls == 2
        assert len(sidecars) == 1 and sidecars[0]["fallback"] == {
            "status": "error", "error": "ValueError: synthetic_final_direct_fallback_error_v47",
        }
        synthetic.append({"fixture": before["fixture"], "synthetic_only": True,
                          "legacy_behavior_and_stdout_unchanged": True, "sidecars": sidecars})
    verify(pre, installed=True)
    shared.save(HERE / "installed_verification_v1.json", {
        "version": pre["version"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "installed_source_sha256": shared.sha(SOURCE), "installed_test_sha256": shared.sha(TEST),
        "exact_expected_targets_only": True,
        "tests": {"exit_code": status, "counts": counts, "xml_sha256": shared.sha(xml), "log_sha256": shared.sha(log)},
        "actual_canonical_state_replays": 14, "terminal_carry_unchanged": carry,
        "legacy_behavior_changes": 0, "rows": rows, "synthetic_failure_protocol_cases": synthetic,
        "production_changed": True, "changed_targets": [str(SOURCE), str(TEST)],
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False, "training_ready": False,
        "services_restarted": False, "semantic_accuracy_gain_claimed": False,
        "old_experiment_raws_outputs_labels_unchanged": True,
        "important_manifest_note": "Historical v45/v48 source manifests retain the original hash. Their strict verify-current-source checks will now intentionally fail for this one patched source; do not edit historical manifests.",
    })
    print({"installed_tests": 326, "canonical_state_replays": 14, "terminal_carry": carry,
           "legacy_behavior_changes": 0, "new_llm_calls": 0, "production_changed": True}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "guard": guard, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_trace_integration.py prepare|guard|check")
    operations[sys.argv[1]]()
