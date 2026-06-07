#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_INPUT = PROJECT_ROOT / "data/processed/llama_factory/student_rollout_quote80_sft_v1_200_merged/raw_pairs.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/llama_factory/student_rollout_quote80_sft_v1_200_teacher_replay_v1"

ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}


def load_api_mode_module() -> Any:
    module_path = PROJECT_ROOT / "api-mode/run_api_inference.py"
    spec = importlib.util.spec_from_file_location("goldenglow_api_mode_runner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load API mode runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def chatml(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system.strip()}\n<|im_end|>\n"
        f"<|im_start|>user\n{user.strip()}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "kto_tag": "kto_tag",
            },
            "tags": ROLE_TAGS,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def make_audit_record(
    *,
    record_id: str,
    task_type: str,
    system: str,
    user: str,
    response: str,
    kto_tag: bool,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "task_type": task_type,
        "bucket": "soda_blackbox",
        "system": system,
        "tools": "[]",
        "kto_tag": bool(kto_tag),
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": response},
        ],
        "meta": meta,
    }


def load_done_prompt_keys(output_dir: Path) -> set[str]:
    done: set[str] = set()
    for path in (output_dir / "raw_pairs_teacher.jsonl", output_dir / "failed.jsonl"):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            prompt_key = str(row.get("prompt_key") or "")
            if prompt_key:
                done.add(prompt_key)
    return done


def build_teacher(args: argparse.Namespace, output_dir: Path) -> Any:
    api_mode = load_api_mode_module()
    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --api-key.")
    api_key = api_key.strip()
    api_mode.validate_api_key(api_key, args.api_key_env)
    return api_mode.OpenAICompatibleAPIRunner(
        api_base_url=args.api_base_url,
        api_key=api_key,
        api_key_env=args.api_key_env,
        model=args.api_model,
        timeout=args.api_timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        response_format_json=not args.no_json_response_format,
        request_log_dir=output_dir / "api_request_logs" if args.save_api_request_logs else None,
    )


def replay_one(row: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    teacher = build_teacher(args, output_dir)
    prompt = chatml(str(row.get("system") or ""), str(row.get("user_prompt") or ""))
    started = time.perf_counter()
    teacher_raw = teacher.generate(
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=1.0,
    )
    elapsed = round(time.perf_counter() - started, 3)
    return {
        **row,
        "teacher_output_raw": teacher_raw,
        "teacher_output": teacher_raw,
        "teacher_valid": bool(str(teacher_raw or "").strip()),
        "teacher_elapsed_sec": elapsed,
        "dry_run": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay API teacher on existing student dry-run raw_pairs and build audit_records.")
    parser.add_argument("--input-raw-pairs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-name", default="student_rollout_quote80_sft_v1_200_teacher_replay_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-type", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default="deepseek-v4-flash")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--no-json-response-format", action="store_true")
    parser.add_argument("--save-api-request-logs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input_raw_pairs if args.input_raw_pairs.is_absolute() else PROJECT_ROOT / args.input_raw_pairs
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)
    if args.task_type:
        allowed = set(args.task_type)
        rows = [row for row in rows if str(row.get("task_type") or "") in allowed]
    if args.skip_existing:
        done = load_done_prompt_keys(output_dir)
        rows = [row for row in rows if str(row.get("prompt_key") or "") not in done]
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        print(json.dumps({"status": "nothing_to_do", "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
        return 0

    raw_path = output_dir / "raw_pairs_teacher.jsonl"
    audit_path = output_dir / "audit_records.jsonl"
    fail_path = output_dir / "failed.jsonl"
    stats = {
        "input_raw_pairs": str(input_path),
        "output_dir": str(output_dir),
        "requested_rows": len(rows),
        "completed": 0,
        "failed": 0,
        "audit_records": 0,
        "api_model": args.api_model,
        "workers": args.workers,
    }

    def handle_done(row: dict[str, Any]) -> None:
        append_jsonl(raw_path, row)
        prompt_key = str(row.get("prompt_key") or "")
        task_type = str(row.get("task_type") or "")
        system = str(row.get("system") or "")
        user = str(row.get("user_prompt") or "")
        base_meta = {
            "soda_mode": "semi_online_blackbox_replay_from_existing_student_raw_pairs",
            "question_key": str(row.get("question_key") or ""),
            "prompt_key": prompt_key,
            "task_type": task_type,
            "student_valid": row.get("student_valid"),
            "student_elapsed_sec": row.get("student_elapsed_sec"),
            "source": {
                "input_raw_pairs": str(input_path),
                "question": str(row.get("question") or ""),
            },
        }
        teacher_text = str(row.get("teacher_output") or "")
        student_text = str(row.get("student_output") or "")
        append_jsonl(
            audit_path,
            make_audit_record(
                record_id=f"{prompt_key}-teacher-pos",
                task_type=task_type,
                system=system,
                user=user,
                response=teacher_text,
                kto_tag=True,
                meta={**base_meta, "preference_role": "teacher_positive"},
            ),
        )
        stats["audit_records"] += 1
        if student_text and student_text != teacher_text:
            append_jsonl(
                audit_path,
                make_audit_record(
                    record_id=f"{prompt_key}-student-neg",
                    task_type=task_type,
                    system=system,
                    user=user,
                    response=student_text,
                    kto_tag=False,
                    meta={**base_meta, "preference_role": "student_negative"},
                ),
            )
            stats["audit_records"] += 1

    if args.workers <= 1:
        for index, row in enumerate(rows, start=1):
            try:
                done = replay_one(row, args, output_dir)
                handle_done(done)
                stats["completed"] += 1
                print(json.dumps({"progress": f"{index}/{len(rows)}", "prompt_key": row.get("prompt_key"), "ok": True}, ensure_ascii=False), flush=True)
            except Exception as exc:
                stats["failed"] += 1
                append_jsonl(
                    fail_path,
                    {
                        "prompt_key": row.get("prompt_key"),
                        "question_key": row.get("question_key"),
                        "question": row.get("question"),
                        "task_type": row.get("task_type"),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                print(json.dumps({"progress": f"{index}/{len(rows)}", "prompt_key": row.get("prompt_key"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {executor.submit(replay_one, row, args, output_dir): row for row in rows}
            for index, future in enumerate(as_completed(future_map), start=1):
                row = future_map[future]
                try:
                    done = future.result()
                    handle_done(done)
                    stats["completed"] += 1
                    ok_payload = {"progress": f"{index}/{len(rows)}", "prompt_key": row.get("prompt_key"), "ok": True}
                    print(json.dumps(ok_payload, ensure_ascii=False), flush=True)
                except Exception as exc:
                    stats["failed"] += 1
                    append_jsonl(
                        fail_path,
                        {
                            "prompt_key": row.get("prompt_key"),
                            "question_key": row.get("question_key"),
                            "question": row.get("question"),
                            "task_type": row.get("task_type"),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    print(json.dumps({"progress": f"{index}/{len(rows)}", "prompt_key": row.get("prompt_key"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)

    # These are mainly for compatibility with LLaMA Factory dataset directories.
    write_json(output_dir / "train.json", [])
    write_json(output_dir / "val.json", [])
    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name))
    write_json(output_dir / "build_summary.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0 if stats["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
