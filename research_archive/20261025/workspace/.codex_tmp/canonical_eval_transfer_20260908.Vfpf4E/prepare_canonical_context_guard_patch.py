"""Prepare, but do not apply, a 3-file patch after exact-byte/GPU checks.

Preexisting dirty files are preserved. Backup byte snapshots do not require
chmod/chown on the experiment mount. Git apply is the only canonical edit step.
"""
import difflib
import hashlib
import json
from pathlib import Path

from verify_context_guard_checkpoint import verify


def sha(data):
    return hashlib.sha256(data).hexdigest()


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
release = Path("release/ASA-ArknightStoryAgent")
precheck = verify(run)
backup = run / "canonical_context_guard_patch_v1"
backup.mkdir()
runner_rel = release / "src/asa_arknight_story_agent/inference/model_runtime/vllm_runner.py"
config_rel = release / "configs/runtime_gpu_reranker_qwen35_4b.json"
test_rel = release / "tests/test_vllm_context_budget.py"
old_config = (repo / config_rel).read_text()
assert old_config.count('"ctx_size": 10000') == 1
assert old_config.count('"max_model_len": 10000') == 1
new_config = old_config.replace('"ctx_size": 10000', '"ctx_size": 12288').replace(
    '"max_model_len": 10000', '"max_model_len": 12288'
)
before_cfg, after_cfg = json.loads(old_config), json.loads(new_config)
after_cfg["generator"]["ctx_size"] = before_cfg["generator"]["ctx_size"]
after_cfg["generator"]["vllm"]["max_model_len"] = before_cfg["generator"]["vllm"]["max_model_len"]
assert after_cfg == before_cfg
assert not (repo / test_rel).exists(), "Do not overwrite an existing test"
replacements = {
    runner_rel: (run / "vllm_runner.context_guard.py").read_bytes(),
    config_rel: new_config.encode(),
    test_rel: (run / "test_vllm_context_budget.py").read_bytes(),
}
changes, patch = [], []
for relative, content in replacements.items():
    path = repo / relative
    previous = path.read_bytes() if path.exists() else b""
    assert previous != content
    original_path = backup / "before" / relative
    original_path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with original_path.open("xb") as output:
            output.write(previous)
    patch.extend(difflib.unified_diff(
        previous.decode().splitlines(keepends=True), content.decode().splitlines(keepends=True),
        fromfile=f"a/{relative}" if path.exists() else "/dev/null",
        tofile=f"b/{relative}",
    ))
    changes.append({
        "path": str(relative), "existed_before": path.exists(),
        "before_sha256": sha(previous) if path.exists() else None,
        "after_sha256": sha(content),
    })
patch_data = "".join(patch).encode()
with (backup / "change.patch").open("xb") as output:
    output.write(patch_data)
with (backup / "manifest.json").open("x") as output:
    output.write(json.dumps({
        "canonical_host": "${REMOTE_HOST}", "head": precheck["canonical_head"],
        "applied": False, "precheck": precheck,
        "patch_sha256": sha(patch_data), "changes": changes,
        "promotion_scope": "Token capacity guard, diagnostics and 10000->12288 config only; not a semantic-quality promotion.",
    }, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"patch": str(backup / "change.patch"), "changes": changes}, ensure_ascii=False))
