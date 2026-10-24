"""Freeze evaluation assets from 241; no inference or training promotion."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


REPO = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
RELEASE = REPO / "release/ASA-ArknightStoryAgent"
ROOT = Path("${REMOTE_ASA_ROOT}")
PATCH_RUN = ROOT / "rank_v8_coverage/experiments/quote_complete_remote_20260908.ESowxe"
SNAPSHOT = ROOT / "rank_v8_coverage/experiments/answer_quality_20260907_aa2545f/results/family_holdout_strict9.snapshots_v3.jsonl"
REVIEW = ROOT / "rank_v8_coverage/experiments/reader_evidence_p0_strict9_20260907_primaryreview_v2/input/strict9_reader_evidence_reviewed_20260907.jsonl"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    run = Path(__file__).resolve().parent
    if (run / "manifest.json").exists() or (run / "before").exists():
        raise SystemExit("refusing to overwrite run")
    observed = {}
    for name, source in (
        ("before", PATCH_RUN / "before_source"),
        ("after", RELEASE),
    ):
        (run / name).mkdir()
        for folder in ("src", "configs"):
            shutil.copytree(source / folder, run / name / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for p in sorted((run / name).rglob("*")):
            if p.is_file():
                origin = source / p.relative_to(run / name)
                digest = sha(p)
                if digest != sha(origin):
                    raise ValueError("source changed while freezing")
                observed[str(p.relative_to(run))] = {
                    "origin": str(origin), "sha256": digest,
                }
    rows = [json.loads(line) for line in SNAPSHOT.read_text().splitlines() if line.strip()]
    if len(rows) != 9 or len({row["id"] for row in rows}) != 9:
        raise ValueError("strict9 snapshot identity mismatch")
    # Explicitly remove gold, answers and review notes from generation input.
    selected = [{key: row[key] for key in ("id", "question", "question_family", "hypothesis")}
                | {"selected_evidence": row["stages"]["prompt"]} for row in rows]
    save(run / "selected_input.json", selected)
    shutil.copy2(REVIEW, run / "review.jsonl")
    gold_path = ROOT / "relevance_data_20260906/gold_val.json"
    pred_path = ROOT / "relevance_data_20260906/clean79.json"
    gold, pred = json.loads(gold_path.read_text()), json.loads(pred_path.read_text())
    by_id = {row["id"]: row for row in pred}
    fixed79 = [row for row in gold if row["task_type"] == "grounded_action_generation"]
    if len(fixed79) != 79 or {row["id"] for row in fixed79} != by_id.keys():
        raise ValueError("historical79 IDs differ from all gold answer tasks")
    for row in fixed79:
        other = by_id[row["id"]]
        if row["system"] != other["system"] or row["conversations"][0] != other["conversations"][0]:
            raise ValueError("historical79 prompt mismatch")
    save(run / "fixed79_gold.json", fixed79)
    cfg = json.loads((RELEASE / "configs/runtime_gpu_reranker_qwen35_4b.json").read_text())
    adapter = RELEASE / cfg["generator"]["vllm"]["lora_path"]
    base = RELEASE / cfg["generator"]["vllm"]["base_model_path"]
    assets = {}
    for role, folder in (("configured_base", base), ("configured_adapter", adapter)):
        assets[role] = {
            "configured_path": str(folder), "resolved_path": str(folder.resolve()),
            "exists": folder.is_dir(),
            "weight_files": [p.name for p in folder.glob("*.safetensors")],
        }
    manifest = {
        "run_id": run.name, "evaluation_only": True, "training_ready": False,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": os.uname().nodename, "canonical_host": "${REMOTE_HOST}",
        "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "-C", str(REPO), "status", "--short"], text=True),
        "source_files": observed,
        "snapshot": {"path": str(SNAPSHOT), "sha256": sha(SNAPSHOT)},
        "review": {"path": str(REVIEW), "sha256": sha(REVIEW)},
        "gold95": {"path": str(gold_path), "sha256": sha(gold_path)},
        "historical79": {"path": str(pred_path), "sha256": sha(pred_path)},
        "selected_input_sha256": sha(run / "selected_input.json"),
        "fixed79_gold_sha256": sha(run / "fixed79_gold.json"),
        "model_assets": assets, "builder_sha256": sha(__file__),
        "limitations": [
            "Frozen selected candidates, not fresh retrieval or production E2E.",
            "Strict9 is a development set; reviews are prior model review, not human gold.",
            "79 answer rows include repeated questions and historical train overlap.",
            "Before/after use actual 241 source, not a local project checkout.",
        ],
    }
    save(run / "manifest.json", manifest)
    print(json.dumps({
        "run_id": run.name, "source_files": len(observed), "strict9": len(rows),
        "fixed79": len(fixed79), "model_assets": assets,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
