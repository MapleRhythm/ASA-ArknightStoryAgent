"""CAS promotion of the reviewed sibling release; no service/public push."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
OLD = "releases/exx-20260915-v1"
NEW = "releases/exx-20260915-v4"
BACKUP = ROOT / "outputs/release_backups/exx-20260915-v4"
PYTHON = "${REMOTE_PYTHON}"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    review = json.loads(args.review.read_text())
    assert review["release"] == Path(NEW).name
    assert review["reviewed_qids"] == [13, 19]
    assert review["replay_target_first_candidates_accepted"] == [8, 13, 18]
    assert review["q13_strict_correct"] is True
    assert review["q19_failed_claims_not_exposed"] is True
    assert review["new_major_errors"] == 0
    results = ROOT / "outputs/experiments/exx_grounding_fix_final2_20260915_v4/results.jsonl"
    assert sha(results) == review["results_sha256"]
    assert os.readlink(ROOT / "current-release") == OLD
    assert not BACKUP.exists()
    subprocess.run([PYTHON, str(ROOT / NEW / "scripts/run_exx_official.py"), "--verify-only"], check=True)
    for alias in ("runtime_inference.json", "runtime_inference_gpu.json", "runtime_exx_official.json"):
        assert (ROOT / "configs" / alias).read_bytes() == (ROOT / NEW / "configs/runtime_exx_official.json").read_bytes()
    if not args.apply:
        print(json.dumps({"ready": True, "default_changed": False}))
        return
    doc = (
        "# Exx正式默认修复版本 — exx-20260915-v4\n\n"
        "当前默认：`current-release -> releases/exx-20260915-v4`。\n"
        "保持原Exx v55模型、检索配置和20条完整证据；修复窄范围转述词面误拒，"
        "失败候选和部分事实仅保留在诊断记录，不自动放行。默认关闭无效重复生成。"
        "不是独立语义验证器。\n\n"
        "定向回归见 `current-release/acceptance/REPORT.md`；"
        "不替代原20题成绩，不代表达到60%线上准确率。v2/v3试验未推广，结果保留。\n\n"
        "入口：`bash scripts/run_exx_official.sh \"问题\"`；"
        "核验：`python scripts/run_exx_official.py --verify-only`。\n\n"
        "回滚本次修复到v1：`python scripts/rollback_exx_20260915_v4.py`。\n"
        "所有历史模型和实验保留；没有公共push或独立API服务切换。\n"
    )
    replacements = {
        "docs/OFFICIAL_EXX_RELEASE_20260915.md": doc,
        "README.md": (
            "# 当前默认版本更新：exx-20260915-v4\n\n"
            "2026-09-15：Exx词面误拒/失败恢复修复已固定为默认。"
            "质量未认证达到60%。详见 `current-release/acceptance/REPORT.md` "
            "及 `docs/OFFICIAL_EXX_RELEASE_20260915.md`。\n\n"
            "以下为先前版本说明；如与上述当前版本冲突，以上述版本为准。\n\n"
            + (ROOT / "README.md").read_text()
        ),
    }
    BACKUP.mkdir(parents=True)
    record = {
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "old": OLD, "new": NEW, "backup": str(BACKUP),
        "review_sha256": sha(args.review), "changes": [],
        "public_push": False, "independent_service_switched": False,
    }
    for relative, content in replacements.items():
        path = ROOT / relative
        backup = BACKUP / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        before = sha(path)
        temporary = path.with_name(path.name + ".exx-v4-new")
        with temporary.open("x") as f:
            f.write(content)
        assert sha(path) == before, "Concurrent modification"
        os.replace(temporary, path)
        record["changes"].append({"path": relative, "before": before, "after": sha(path)})
    assert os.readlink(ROOT / "current-release") == OLD
    temporary_pointer = ROOT / "current-release.exx-v4-new"
    temporary_pointer.symlink_to(NEW)
    os.replace(temporary_pointer, ROOT / "current-release")
    with (BACKUP / "publication.json").open("x") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
