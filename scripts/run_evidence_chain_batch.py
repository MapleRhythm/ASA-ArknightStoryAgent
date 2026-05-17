#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from goldenglow.config import STORY_ROOT

import evidence_chain_dataset as ecd


def natural_key(path: Path) -> list[Any]:
    return ecd.natural_key(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ecd.write_jsonl(path, records)


def collect_activity_dirs(activities_root: Path) -> list[Path]:
    return sorted(
        [path for path in activities_root.iterdir() if path.is_dir()],
        key=natural_key,
    )


def collect_story_files(activity_dir: Path, glob_pattern: str) -> list[Path]:
    files = sorted(activity_dir.rglob(glob_pattern), key=natural_key)
    return [
        path
        for path in files
        if path.is_file() and path.suffix == ".txt"
    ]


def parse_activity_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def estimate_file_chars(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return path.stat().st_size


def split_story_files_by_chars(files: list[Path], max_chars: int) -> list[list[Path]]:
    if max_chars <= 0:
        return [files]
    groups: list[list[Path]] = []
    current: list[Path] = []
    current_chars = 0
    for path in files:
        file_chars = estimate_file_chars(path)
        if current and current_chars + file_chars > max_chars:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(path)
        current_chars += file_chars
    if current:
        groups.append(current)
    return groups


def render_prompt_for_files(
    files: list[Path],
    *,
    min_questions: int,
    max_questions: int,
) -> tuple[str, int]:
    source_text = ecd.render_story_source(files)
    prompt = ecd.PROMPT_TEMPLATE.format(
        min_questions=min_questions,
        max_questions=max_questions,
        source_text=source_text,
    )
    return prompt, len(source_text)


def save_prompt(prompt_dir: Path, prompt_id: str, files: list[Path], prompt: str, source_chars: int) -> dict[str, Any]:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / f"{prompt_id}.prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    source_files = [
        path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else path.as_posix()
        for path in files
    ]
    return {
        "prompt_id": prompt_id,
        "prompt_file": prompt_file.relative_to(PROJECT_ROOT).as_posix() if prompt_file.is_relative_to(PROJECT_ROOT) else prompt_file.as_posix(),
        "source_files": source_files,
        "source_chars": source_chars,
        "prompt_chars": len(prompt),
    }


def call_api_for_prompt(
    prompt_file: Path,
    output_json: Path,
    *,
    api_base: str,
    api_key_env: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    response_format_json: bool,
    quiet: bool,
) -> int:
    args = SimpleNamespace(
        prompt_file=prompt_file,
        output_json=output_json,
        api_base=api_base,
        endpoint_path="/v1/chat/completions",
        api_key_env=api_key_env,
        model=model,
        model_env="MINIMAX_MODEL",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format_json=response_format_json,
        quiet=quiet,
    )
    return ecd.command_call_api(args)


def recover_response_if_needed(output_json: Path) -> bool:
    response_text = output_json.with_suffix(output_json.suffix + ".response.txt")
    if not response_text.exists() or output_json.exists():
        return False
    try:
        args = SimpleNamespace(input=response_text, output_json=output_json)
        ecd.command_recover_json(args)
        return True
    except Exception as exc:
        print(f"[batch] recover failed for {response_text}: {exc}", file=sys.stderr, flush=True)
        return False


def export_outputs(annotation_files: list[Path], output_dir: Path, *, allow_errors: bool) -> int:
    args = SimpleNamespace(
        inputs=[path.as_posix() for path in annotation_files],
        output_dir=output_dir,
        allow_errors=allow_errors,
    )
    return ecd.command_export(args)


def print_progress(
    *,
    completed_jobs: int,
    total_jobs: int,
    accepted_samples: int,
    target_samples: int,
    ok_jobs: int,
    skipped_jobs: int,
    invalid_jobs: int,
    failed_jobs: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    jobs_per_min = completed_jobs / elapsed * 60.0
    samples_per_min = accepted_samples / elapsed * 60.0
    remaining_samples = max(target_samples - accepted_samples, 0)
    eta_seconds = remaining_samples / max(samples_per_min / 60.0, 1e-9) if remaining_samples else 0.0
    print(
        "[progress] "
        f"jobs={completed_jobs}/{total_jobs} "
        f"samples={accepted_samples}/{target_samples} "
        f"ok={ok_jobs} skipped={skipped_jobs} invalid={invalid_jobs} failed={failed_jobs} "
        f"jobs/min={jobs_per_min:.2f} samples/min={samples_per_min:.2f} "
        f"eta={eta_seconds/60.0:.1f}m",
        file=sys.stderr,
        flush=True,
    )


def run_api_job(job: dict[str, Any]) -> dict[str, Any]:
    prompt_id = str(job["prompt_id"])
    annotation_file = Path(job["annotation_file"])
    prompt_file = Path(job["prompt_file"])
    if job.get("resume") and annotation_file.exists():
        return {
            "prompt_id": prompt_id,
            "status": "skipped_existing",
            "annotation_file": annotation_file.as_posix(),
            "valid_samples": 0,
            "warnings": 0,
            "errors": 0,
        }
    try:
        call_api_for_prompt(
            prompt_file,
            annotation_file,
            api_base=str(job["api_base"]),
            api_key_env=str(job["api_key_env"]),
            model=str(job["model"]),
            max_tokens=int(job["max_tokens"]),
            temperature=float(job["temperature"]),
            timeout=float(job["timeout"]),
            response_format_json=bool(job["response_format_json"]),
            quiet=bool(job["quiet_api"]),
        )
    except Exception as exc:
        print(f"[batch] API failed for {prompt_id}: {exc}", file=sys.stderr, flush=True)
        recover_response_if_needed(annotation_file)
        if not annotation_file.exists():
            return {
                "prompt_id": prompt_id,
                "status": "failed",
                "annotation_file": annotation_file.as_posix(),
                "error": str(exc),
                "valid_samples": 0,
                "warnings": 0,
                "errors": 1,
            }

    try:
        normalized, issues = ecd.validate_and_normalize_payload(
            ecd.load_json_payload(annotation_file),
            source_name=annotation_file.name,
        )
        return {
            "prompt_id": prompt_id,
            "status": "ok",
            "annotation_file": annotation_file.as_posix(),
            "valid_samples": len(normalized["rerank_dataset"]),
            "warnings": sum(1 for issue in issues if issue.level == "warning"),
            "errors": sum(1 for issue in issues if issue.level == "error"),
        }
    except Exception as exc:
        return {
            "prompt_id": prompt_id,
            "status": "invalid",
            "annotation_file": annotation_file.as_posix(),
            "error": str(exc),
            "valid_samples": 0,
            "warnings": 0,
            "errors": 1,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly generate evidence-chain reranker data from activity stories.")
    parser.add_argument("--target-samples", type=int, default=1000)
    parser.add_argument("--max-prompts", type=int, default=90)
    parser.add_argument("--sample-activities", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--glob", default="*.txt")
    parser.add_argument("--activities-root", type=Path, default=STORY_ROOT / "activities")
    parser.add_argument(
        "--priority-activities",
        default="act10mini,act17side,act34side",
        help="Comma-separated activity ids to try before random sampling; useful for known hard reasoning/reveal cases.",
    )
    parser.add_argument("--prompt-root", type=Path, default=Path("outputs/evidence_chain_prompts/batch_v2_answerability"))
    parser.add_argument("--annotation-root", type=Path, default=Path("data/processed/evidence_chain_annotations/batch_v2_answerability"))
    parser.add_argument("--dataset-output-dir", type=Path, default=Path("data/processed/evidence_chain_reranker/batch_v2_answerability"))
    parser.add_argument("--min-questions", type=int, default=10)
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--max-source-chars", type=int, default=0, help="Maximum rendered source chars per story folder. 0 disables the limit.")
    parser.add_argument(
        "--chunk-source-chars",
        type=int,
        default=70000,
        help="Split large activities into multiple prompts by approximate raw source chars. 0 disables splitting.",
    )
    parser.add_argument("--max-files-per-prompt", type=int, default=0, help="Deprecated compatibility option; folder prompts are never split.")
    parser.add_argument("--min-files-per-prompt", type=int, default=0, help="Deprecated compatibility option; folder prompts are never split.")
    parser.add_argument("--min-source-chars", type=int, default=10000, help="Skip complete story folders with fewer rendered source chars. 0 disables the filter.")
    parser.add_argument("--api-base", default=os.environ.get("MINIMAX_API_BASE", "https://api.svips.org"))
    default_api_key_env = "MINIMAX_API_KEY" if os.environ.get("MINIMAX_API_KEY") else "SVIPS_API_KEY"
    parser.add_argument("--api-key-env", default=default_api_key_env)
    parser.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"))
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent API requests. 1 keeps the old sequential behavior.")
    parser.add_argument("--no-response-format-json", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-export-errors", action="store_true", default=True)
    parser.add_argument("--strict-export", action="store_true", help="Fail the batch if export validation reports errors.")
    parser.add_argument("--quiet-api", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not args.dry_run and not api_key:
        raise RuntimeError(f"Missing API key. Set {args.api_key_env} before running, or pass --api-key-env.")
    if not args.dry_run and not args.model:
        raise RuntimeError("Missing model. Pass --model or set MINIMAX_MODEL.")

    random.seed(args.seed)
    activities_root = args.activities_root if args.activities_root.is_absolute() else PROJECT_ROOT / args.activities_root
    activity_dirs = collect_activity_dirs(activities_root)
    activity_dir_by_name = {path.name: path for path in activity_dirs}
    priority_dirs = [
        activity_dir_by_name[name]
        for name in parse_activity_list(args.priority_activities)
        if name in activity_dir_by_name
    ]
    priority_names = {path.name for path in priority_dirs}
    remaining_dirs = [path for path in activity_dirs if path.name not in priority_names]
    random.shuffle(remaining_dirs)
    activity_dirs = priority_dirs + remaining_dirs

    prompt_jobs: list[dict[str, Any]] = []
    for activity_dir in activity_dirs:
        files = collect_story_files(activity_dir, args.glob)
        if not files:
            continue
        activity_id = activity_dir.name
        file_groups = split_story_files_by_chars(files, args.chunk_source_chars)
        for group_index, file_group in enumerate(file_groups, start=1):
            prompt_id = activity_id if len(file_groups) == 1 else f"{activity_id}_part{group_index:02d}"
            prompt_jobs.append(
                {
                    "activity_id": activity_id,
                    "prompt_id": prompt_id,
                    "files": file_group,
                }
            )
        if args.sample_activities and len(prompt_jobs) >= args.sample_activities:
            break

    if not prompt_jobs:
        raise RuntimeError("No prompt jobs were built from the selected activities.")

    prompt_root = args.prompt_root if args.prompt_root.is_absolute() else PROJECT_ROOT / args.prompt_root
    annotation_root = args.annotation_root if args.annotation_root.is_absolute() else PROJECT_ROOT / args.annotation_root
    dataset_output_dir = args.dataset_output_dir if args.dataset_output_dir.is_absolute() else PROJECT_ROOT / args.dataset_output_dir
    prompt_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)

    plan_records: list[dict[str, Any]] = []
    annotation_files: list[Path] = []
    api_jobs: list[dict[str, Any]] = []
    accepted_samples = 0
    completed_jobs = 0
    ok_jobs = 0
    skipped_jobs = 0
    invalid_jobs = 0
    failed_jobs = 0
    skipped_oversize = 0
    skipped_too_small = 0
    started_at = time.monotonic()
    for job_index, job in enumerate(prompt_jobs, start=1):
        if args.max_prompts and len(plan_records) >= args.max_prompts:
            break
        activity_id = job["activity_id"]
        prompt_id = job.get("prompt_id") or activity_id
        prompt_dir = prompt_root / prompt_id
        annotation_file = annotation_root / f"{prompt_id}.json"
        prompt_file = prompt_dir / f"{prompt_id}.prompt.txt"

        prompt, source_chars = render_prompt_for_files(
            job["files"],
            min_questions=args.min_questions,
            max_questions=args.max_questions,
        )
        if args.min_source_chars and source_chars < args.min_source_chars:
            skipped_too_small += 1
            print(
                f"[batch] skip small {prompt_id}: source_chars={source_chars} < min_source_chars={args.min_source_chars}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if args.max_source_chars and source_chars > args.max_source_chars:
            skipped_oversize += 1
            print(
                f"[batch] skip oversize {prompt_id}: source_chars={source_chars} > max_source_chars={args.max_source_chars}",
                file=sys.stderr,
                flush=True,
            )
            continue
        prompt_record = save_prompt(prompt_dir, prompt_id, job["files"], prompt, source_chars)
        plan_records.append(prompt_record)

        print(
            f"[batch] {job_index}/{len(prompt_jobs)} prompt={prompt_id} source_chars={source_chars} files={len(job['files'])}",
            file=sys.stderr,
            flush=True,
        )
        if args.dry_run:
            continue

        if args.parallel > 1:
            api_jobs.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_file": prompt_file,
                    "annotation_file": annotation_file,
                    "api_base": args.api_base,
                    "api_key_env": args.api_key_env,
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "timeout": args.timeout,
                    "response_format_json": not args.no_response_format_json,
                    "quiet_api": args.quiet_api,
                    "resume": args.resume,
                }
            )
        else:
            result = run_api_job(
                {
                    "prompt_id": prompt_id,
                    "prompt_file": prompt_file,
                    "annotation_file": annotation_file,
                    "api_base": args.api_base,
                    "api_key_env": args.api_key_env,
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "timeout": args.timeout,
                    "response_format_json": not args.no_response_format_json,
                    "quiet_api": args.quiet_api,
                    "resume": args.resume,
                }
            )
            if annotation_file.exists():
                annotation_files.append(annotation_file)
            accepted_samples += int(result.get("valid_samples") or 0)
            if result["status"] == "skipped_existing":
                skipped_jobs += 1
                print(f"[batch] skip existing annotation {annotation_file}", file=sys.stderr, flush=True)
            elif result["status"] == "ok":
                ok_jobs += 1
                print(
                    f"[batch] accepted_samples+={result['valid_samples']} total={accepted_samples} warnings={result['warnings']} errors={result['errors']}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                if result["status"] == "invalid":
                    invalid_jobs += 1
                else:
                    failed_jobs += 1
                print(f"[batch] job {prompt_id} status={result['status']} error={result.get('error', '')}", file=sys.stderr, flush=True)
            completed_jobs += 1
            print_progress(
                completed_jobs=completed_jobs,
                total_jobs=max(len(plan_records), 1),
                accepted_samples=accepted_samples,
                target_samples=args.target_samples,
                ok_jobs=ok_jobs,
                skipped_jobs=skipped_jobs,
                invalid_jobs=invalid_jobs,
                failed_jobs=failed_jobs,
                started_at=started_at,
            )

        if args.parallel <= 1 and accepted_samples >= args.target_samples:
            print(f"[batch] target samples reached: {accepted_samples}", file=sys.stderr, flush=True)
            break
        if args.parallel <= 1 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if api_jobs:
        max_workers = max(1, args.parallel)
        print(f"[batch] running {len(api_jobs)} API jobs with parallel={max_workers}", file=sys.stderr, flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_api_job, job) for job in api_jobs]
            for future in as_completed(futures):
                result = future.result()
                annotation_file = Path(result["annotation_file"])
                if annotation_file.exists():
                    annotation_files.append(annotation_file)
                accepted_samples += int(result.get("valid_samples") or 0)
                if result["status"] == "ok":
                    ok_jobs += 1
                    print(
                        f"[batch] done {result['prompt_id']} samples+={result['valid_samples']} total={accepted_samples} warnings={result['warnings']} errors={result['errors']}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif result["status"] == "skipped_existing":
                    skipped_jobs += 1
                    print(f"[batch] skip existing annotation {annotation_file}", file=sys.stderr, flush=True)
                else:
                    if result["status"] == "invalid":
                        invalid_jobs += 1
                    else:
                        failed_jobs += 1
                    print(f"[batch] done {result['prompt_id']} status={result['status']} error={result.get('error', '')}", file=sys.stderr, flush=True)
                completed_jobs += 1
                print_progress(
                    completed_jobs=completed_jobs,
                    total_jobs=max(len(api_jobs), 1),
                    accepted_samples=accepted_samples,
                    target_samples=args.target_samples,
                    ok_jobs=ok_jobs,
                    skipped_jobs=skipped_jobs,
                    invalid_jobs=invalid_jobs,
                    failed_jobs=failed_jobs,
                    started_at=started_at,
                )
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    write_jsonl(prompt_root / "batch_plan.jsonl", plan_records)
    if args.dry_run:
        summary = {
            "dry_run": True,
            "prompt_jobs": len(plan_records),
            "selected_activities": len(plan_records),
            "skipped_oversize": skipped_oversize,
            "skipped_too_small": skipped_too_small,
            "estimated_samples_min": len(plan_records) * args.min_questions,
            "estimated_samples_max": len(plan_records) * args.max_questions,
            "batch_plan": str(prompt_root / "batch_plan.jsonl"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if annotation_files:
        export_code = export_outputs(annotation_files, dataset_output_dir, allow_errors=args.allow_export_errors and not args.strict_export)
        if export_code != 0:
            raise RuntimeError(
                "Export failed. Re-run with --allow-export-errors to write usable records while keeping validation issues."
            )

    elapsed = time.monotonic() - started_at
    summary = {
        "dry_run": False,
        "prompt_jobs": len(plan_records),
        "annotations": len(annotation_files),
        "skipped_oversize": skipped_oversize,
        "skipped_too_small": skipped_too_small,
        "accepted_samples_before_export": accepted_samples,
        "target_samples": args.target_samples,
        "prompt_root": str(prompt_root),
        "annotation_root": str(annotation_root),
        "dataset_output_dir": str(dataset_output_dir),
        "elapsed_seconds": round(elapsed, 2),
    }
    (dataset_output_dir / "batch_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (dataset_output_dir / "batch_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
