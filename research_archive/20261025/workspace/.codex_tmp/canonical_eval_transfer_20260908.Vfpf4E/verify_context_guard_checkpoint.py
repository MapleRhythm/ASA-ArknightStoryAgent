"""Verify GPU receipts and frozen canonical bytes before a scoped patch.

This is an operational-capacity check, not a claim of improved answer quality.
All 79 historical states remain; exact GPU input IDs were checked at generation.
"""
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(run):
    folder = run / "strict9_vllm_context_guard_ctx12288_stable_idle"
    meta = read(folder / "metadata.json")
    rows = read(folder / "predictions.json")
    matrix = read(run / "after.matrix.json")
    inputs = {r["id"]: r for r in matrix["rows"]}
    candidate = run / "context_guard_candidate_bytes_v1/src"
    runner = candidate / "asa_arknight_story_agent/inference/model_runtime/vllm_runner.py"
    assert meta["status"] == "completed" and meta["completed_rows"] == len(rows) == 9
    assert len(inputs) == 9 and {r["id"] for r in rows} == set(inputs)
    assert meta["source_matrix_sha256"] == sha(run / "after.matrix.json")
    assert meta["predictions_sha256"] == sha(folder / "predictions.json")
    assert meta["runtime_runner_sha256"] == sha(runner) == sha(run / "vllm_runner.context_guard.py")
    assert meta["runtime"]["max_model_len"] == 12288
    assert meta["runtime"]["gpu_memory_utilization"] == 0.52
    assert meta["decoding"]["max_new_tokens"] == 2048
    assert meta["length_outputs"] == 0 and meta["input_truncated"] is False
    checks = []
    for row in rows:
        source = inputs[row["id"]]
        diagnostics = row["runtime_generation_diagnostics"]
        assert row["actual_prompt"] == source["runtime_prompt"]
        assert row["actual_prompt_sha256"] == source["runtime_prompt_sha256"]
        assert row["input_token_ids_match_untruncated_prompt"] is True
        assert row["reserved_output_over_context"] is False
        assert diagnostics["prompt_tokens"] == row["prompt_token_count"]
        assert diagnostics["generated_tokens"] == row["generated_token_count"]
        assert diagnostics["finish_reason"] == row["finish_reason"] == "stop"
        assert diagnostics["stop_reason"] == row["stop_reason"]
        assert diagnostics["requested_max_tokens"] == 2048
        assert diagnostics["max_model_len"] == 12288
        assert diagnostics["available_output_tokens"] == 12288 - row["prompt_token_count"]
        assert diagnostics["input_truncated"] is False
        assert diagnostics["output_budget_reduced"] is False
        assert diagnostics["status"] == "completed"
        checks.append({"id": row["id"], **diagnostics})
    fixed_path = run / "fixed79_configured_lora_ctx12288/validation_after_v2.json"
    fixed = read(fixed_path)["results"]
    assert len(fixed) == len({r["id"] for r in fixed}) == 79
    assert all(r["input_tokens"] + 3000 <= 12288 for r in fixed)
    manifest = read(run / "manifest.json")
    canonical = []
    for relative, info in manifest["source_files"].items():
        if not relative.startswith("after/"):
            continue
        path = Path(info["origin"])
        assert sha(path) == info["sha256"], f"Concurrent canonical edit: {path}"
        canonical.append(str(path))
    repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert head == manifest["git_head"]
    return {
        "version": "context_guard_promotion_precheck_v1",
        "operational_capacity_verified": True, "answer_quality_improvement_claimed": False,
        "production_service_restarted": False, "evaluation_only": True,
        "canonical_head": head, "unchanged_canonical_files": len(canonical),
        "runtime_candidate_sha256": sha(runner), "gpu_predictions_sha256": sha(folder / "predictions.json"),
        "fixed79_analysis_sha256": sha(fixed_path), "fixed79_max_prompt_tokens": max(r["input_tokens"] for r in fixed),
        "fixed79_states_fitting_12288_with_3000_output": len(fixed),
        "verified_gpu_rows": checks,
        "limitations": [
            "9 GPU records are repeated development diagnostics, not a holdout or full pipeline.",
            "Capacity increase does not establish semantic accuracy or solve output repetition.",
            "Arbitrary future prompts may still exceed context and must fail explicitly.",
            "Runtime diagnostics on the runner are not yet propagated into all pipeline traces.",
        ],
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = verify(root)
    with (root / "context_guard_promotion_precheck_v1.json").open("x") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"verified_gpu_rows", "limitations"}}))
