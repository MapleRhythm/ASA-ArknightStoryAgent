"""Audit predetermined complete prefixes while generation continues.

Batching is only for latency. All 68 generation records must be present before
any combined summary. Successful audits are reused by exact request hash.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time


def completed_journal_rows(text):
    # A non-newline-terminated final record may still be in-flight.
    return [json.loads(line) for line in text.splitlines(keepends=True)
            if line.endswith("\n") and line.strip()]


def check_prefix(rows, matrix_rows):
    if len(rows) > len(matrix_rows):
        raise ValueError("too many generated rows")
    for actual, expected in zip(rows, matrix_rows):
        if actual["id"] != expected["id"] or actual["actual_prompt"] != expected["runtime_prompt"]:
            raise ValueError("prediction prefix differs from frozen matrix")


def write_new(path, data):
    with path.open("x") as output:
        output.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = Path(__file__).resolve().parent
    py = "${REMOTE_HOME}/miniconda3/envs/train/bin/python"
    matrix_path = run / "same_evidence_retry_paired68_v1.matrix.json"
    matrix = json.loads(matrix_path.read_text())
    source_rows = matrix["rows"]
    assert len(source_rows) == 68
    folder = run / "same_evidence_retry_paired68_ctx12288_v1"
    audit_root = run / "same_evidence_retry_paired68_audit_v1"
    audit_root.mkdir()
    journal = folder / "predictions.progress.jsonl"
    deadline = time.monotonic() + 5400
    semantics = []
    # Fixed, declared cut points; no selection by outcomes or model labels.
    for start, end in ((0, 16), (16, 32), (32, 48), (48, 68)):
        while True:
            predictions = completed_journal_rows(journal.read_text()) if journal.exists() else []
            check_prefix(predictions, source_rows)
            if len(predictions) >= end:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"generation prefix incomplete: {len(predictions)}/{end}")
            time.sleep(10)
        batch = audit_root / f"rows_{start + 1:02}_{end:02}"
        batch.mkdir()
        subset = copy.deepcopy(matrix)
        subset["rows"] = subset["rows"][start:end]
        subset["batch_only_not_full_summary"] = True
        subset["parent_matrix_sha256"] = sha(matrix_path)
        write_new(batch / "matrix.json", subset)
        write_new(batch / "predictions.json", predictions[start:end])
        subprocess.run([
            py, str(run / "analyze_canonical_quote_v2.py"),
            "--validator", "after", "--matrix", str(batch / "matrix.json"),
            "--predictions", str(batch / "predictions.json"), "--output", str(batch / "analysis.json"),
        ], check=True)
        semantic = batch / "semantic.json"
        print(json.dumps({"status": "auditing_fixed_batch", "start": start + 1, "end": end}), flush=True)
        result = subprocess.run([
            py, str(run / "judge_visible_answers_v2.py"),
            "--analysis", str(batch / "analysis.json"), "--output", str(semantic), "--workers", "2",
        ])
        if result.returncode not in (0, 1) or not semantic.exists():
            raise RuntimeError(f"judge process failed without a retained result: {result.returncode}")
        semantics.append(semantic)
    while not (folder / "sampling_seed_override.json").exists() or not (folder / "metadata.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("generation completion receipt is missing")
        time.sleep(5)
    meta = json.loads((folder / "metadata.json").read_text())
    seeds = json.loads((folder / "sampling_seed_override.json").read_text())
    predictions = json.loads((folder / "predictions.json").read_text())
    check_prefix(predictions, source_rows)
    assert len(predictions) == meta["completed_rows"] == 68
    assert meta["status"] == "completed" and seeds["completed"]
    assert meta["predictions_sha256"] == seeds["predictions_sha256"] == sha(folder / "predictions.json")
    assert [r["id"] for r in seeds["rows"]] == [r["id"] for r in source_rows]
    assert [r["seed"] for r in seeds["rows"]] == [r["sampling_seed"] for r in source_rows]
    analysis_path = folder / "validation_after_v2.json"
    subprocess.run([
        py, str(run / "analyze_canonical_quote_v2.py"), "--validator", "after",
        "--matrix", str(matrix_path), "--predictions", str(folder / "predictions.json"),
        "--output", str(analysis_path),
    ], check=True)
    command = [
        py, str(run / "judge_visible_answers_v2.py"),
        "--analysis", str(analysis_path), "--output", str(folder / "semantic68_reuse_retry1.json"),
        "--workers", "2",
    ]
    for source in semantics:
        command.extend(["--reuse", str(source)])
    result = subprocess.run(command)
    if result.returncode not in (0, 1):
        raise RuntimeError("full-cohort audit reuse failed")
    subprocess.run([
        py, str(run / "summarize_canonical_audit.py"), "--analysis", str(analysis_path),
        "--semantic", str(folder / "semantic68_reuse_retry1.json"),
        "--output", str(folder / "semantic_summary68_retry1.json"),
    ], check=True)
    write_new(audit_root / "completed.json", {
        "all_generation_records": 68, "semantic_returncode": result.returncode,
        "combined_summary_sha256": sha(folder / "semantic_summary68_retry1.json"),
        "batches": [{"path": str(p), "sha256": sha(p)} for p in semantics],
        "no_failed_record_dropped": True, "evaluation_only": True, "training_ready": False,
    })
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
