#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.data.sft_teacher import (  # noqa: E402
    ACCEPTED_TASK_TYPES,
    TeacherApiConfig,
    build_request_record,
    build_teacher_prompts,
    call_teacher_api,
    dedupe_samples,
    load_generation_config,
    load_story_documents,
    normalize_task_mix,
    normalize_task_type,
    parse_teacher_json,
    sample_evidence_documents,
    sample_worldbuilding_topic,
    split_samples,
    validate_and_normalize_samples,
    weighted_choice,
)
from goldenglow.config import EMBEDDING_MODEL_DIR, QueryConfig, RERANKER_MODEL_DIR  # noqa: E402


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


REQUEST_FILE_RE = re.compile(r"req-(\d+)\.json$")


def next_request_index(prompt_dir: Path, existing_request_records: list[dict]) -> int:
    max_index = -1
    for path in prompt_dir.glob("req-*.json"):
        match = REQUEST_FILE_RE.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    for record in existing_request_records:
        request_id = record.get("request_id")
        if isinstance(request_id, str) and request_id.startswith("req-"):
            try:
                max_index = max(max_index, int(request_id.split("-")[1]))
            except (IndexError, ValueError):
                continue
    return max_index + 1


def compute_target_counts(target_total: int, task_mix: dict[str, float]) -> dict[str, int]:
    total_weight = sum(task_mix.values())
    targets = {
        task_type: int(target_total * (weight / total_weight))
        for task_type, weight in task_mix.items()
    }
    assigned = sum(targets.values())
    remainders = sorted(
        (
            (
                target_total * (weight / total_weight) - targets[task_type],
                task_type,
            )
            for task_type, weight in task_mix.items()
        ),
        reverse=True,
    )
    for _, task_type in remainders[: max(0, target_total - assigned)]:
        targets[task_type] += 1
    return targets


def choose_task_type(
    rng: random.Random,
    task_mix: dict[str, float],
    target_counts: dict[str, int],
    current_counts: Counter,
) -> str:
    remaining = {
        task_type: max(0, target_counts[task_type] - current_counts.get(task_type, 0))
        for task_type in task_mix
    }
    deficit_weights = {task_type: count for task_type, count in remaining.items() if count > 0}
    if deficit_weights:
        return weighted_choice(rng, deficit_weights)
    return weighted_choice(rng, task_mix)


def build_worldbuilding_request_queue(
    topics: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    queue = [dict(topic) for topic in topics]
    return queue


def build_retrieval_seed_query(document: dict, *, max_chars: int) -> str:
    parts: list[str] = []
    for key in ("activity_name", "story_name", "stage_code", "avg_tag"):
        value = document.get(key)
        if value:
            parts.append(str(value))

    speakers = []
    for segment in document.get("segments") or []:
        speaker = segment.get("speaker") if isinstance(segment, dict) else None
        if speaker and speaker not in speakers:
            speakers.append(speaker)
        if len(speakers) >= 4:
            break
    if speakers:
        parts.append(" ".join(speakers))

    clean_text = str(document.get("clean_text") or "")
    if clean_text:
        parts.append(clean_text[:max_chars])

    return "\n".join(part for part in parts if part).strip()


def load_retriever(*, device: str, use_reranker: bool):
    from goldenglow.retrieval.hybrid import ArknightsHybridRetriever

    return ArknightsHybridRetriever.from_paths(
        embedding_model_path=EMBEDDING_MODEL_DIR,
        reranker_model_path=RERANKER_MODEL_DIR if use_reranker else None,
        device=device,
    )


def retrieve_evidence_documents(
    retriever,
    *,
    query: str,
    top_k: int,
) -> list[dict]:
    results = retriever.search(
        query,
        config=QueryConfig(
            dense_top_k=max(40, top_k * 8),
            sparse_top_k=max(40, top_k * 8),
            fusion_top_k=max(30, top_k * 6),
            rerank_top_k=top_k,
        ),
    )
    return [item["document"] for item in results]


def quotas_satisfied(current_counts: Counter, target_counts: dict[str, int]) -> bool:
    return all(
        current_counts.get(task_type, 0) >= target
        for task_type, target in target_counts.items()
    )


def select_balanced_subset(
    samples: list[dict],
    *,
    target_total: int,
    target_counts: dict[str, int],
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    pool = list(samples)
    rng.shuffle(pool)

    selected: list[dict] = []
    selected_counts: Counter = Counter()
    remainder: list[dict] = []

    for sample in pool:
        task_type = sample["task_type"]
        if selected_counts[task_type] < target_counts.get(task_type, 0):
            selected.append(sample)
            selected_counts[task_type] += 1
        else:
            remainder.append(sample)

    for sample in remainder:
        if len(selected) >= target_total:
            break
        selected.append(sample)

    return selected[:target_total]


def save_category_splits(
    output_dir: Path,
    samples: list[dict],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    categories = ("style", "knowledge", "tool")
    for category in categories:
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        category_records = [
            record
            for record in samples
            if (record.get("bucket") or record.get("meta", {}).get("category")) == category
        ]
        category_splits = split_samples(
            category_records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        save_jsonl(category_dir / "all.jsonl", category_records)
        save_jsonl(category_dir / "train.jsonl", category_splits["train"])
        save_jsonl(category_dir / "val.jsonl", category_splits["val"])
        save_jsonl(category_dir / "test.jsonl", category_splits["test"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grounded SFT data by calling a teacher-model API."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sft_teacher_generation.json",
    )
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument(
        "--only-task-type",
        type=str,
        choices=sorted(ACCEPTED_TASK_TYPES),
        default=None,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--evidence-mode",
        type=str,
        choices=("retrieval", "random"),
        default=None,
    )
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_generation_config(args.config)
    dataset_cfg = config["dataset"]
    source_cfg = config["source"]
    task_mix = normalize_task_mix(config["task_mix"])
    worldbuilding_cfg = config.get("worldbuilding") or {}

    output_dir = PROJECT_ROOT / dataset_cfg["output_dir"]
    raw_dir = output_dir / "raw"
    prompt_dir = output_dir / "prompts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    existing_request_records = load_jsonl_if_exists(raw_dir / "requests.jsonl")
    existing_samples = [] if args.dry_run else load_jsonl_if_exists(output_dir / "all.jsonl")
    request_offset = next_request_index(prompt_dir, existing_request_records)

    target_total = args.target_total or int(dataset_cfg["target_total"])
    max_requests = args.max_requests or int(dataset_cfg["max_requests"])
    samples_per_request = int(dataset_cfg["samples_per_request"])
    seed = int(dataset_cfg["seed"])
    evidence_mode = args.evidence_mode or source_cfg.get("evidence_mode", "retrieval")
    retrieval_top_k = int(
        source_cfg.get(
            "retrieval_top_k",
            source_cfg.get("max_evidence_docs_per_request", 3),
        )
    )
    seed_query_max_chars = int(source_cfg.get("seed_query_max_chars", 260))

    rng = random.Random(seed)
    documents = load_story_documents(PROJECT_ROOT / source_cfg["documents_path"])
    if not documents:
        raise RuntimeError("No source documents available for teacher-data generation.")
    worldbuilding_topics = worldbuilding_cfg.get("topics") or []
    worldbuilding_queue = build_worldbuilding_request_queue(worldbuilding_topics, rng)

    teacher_api = TeacherApiConfig(
        api_type=config["teacher_api"]["api_type"],
        base_url=config["teacher_api"]["base_url"],
        model=config["teacher_api"]["model"],
        api_key_env=config["teacher_api"]["api_key_env"],
        timeout_seconds=int(config["teacher_api"].get("timeout_seconds", 120)),
        temperature=float(config["teacher_api"].get("temperature", 0.8)),
        max_output_tokens=int(config["teacher_api"].get("max_output_tokens", 4000)),
        json_mode=bool(config["teacher_api"].get("json_mode", True)),
        extra_headers=config["teacher_api"].get("extra_headers") or {},
    )

    if not os.environ.get(teacher_api.api_key_env):
        raise SystemExit(
            f"Missing teacher API key env var: {teacher_api.api_key_env}\n"
            f"Please export it first, for example:\n"
            f"  export {teacher_api.api_key_env}=<your_key>"
        )

    all_samples: list[dict] = []
    request_records: list[dict] = []
    target_counts = compute_target_counts(target_total, task_mix)
    retriever = None
    progress = tqdm(
        total=max_requests,
        desc="Generating SFT prompts" if args.dry_run else "Generating SFT samples",
        unit="req",
    )

    for request_index in range(max_requests):
        request_started = time.time()
        retrieval_seconds = 0.0
        api_seconds = 0.0
        accepted_samples = 0
        parsed_ok = False
        request_error: str | None = None
        current_counts = Counter(sample["task_type"] for sample in all_samples)
        if (
            not args.only_task_type
            and len(all_samples) >= target_total
            and quotas_satisfied(current_counts, target_counts)
        ):
            break

        if args.only_task_type:
            task_type = normalize_task_type(args.only_task_type)
        elif worldbuilding_queue:
            task_type = "worldbuilding_qa"
        else:
            task_type = choose_task_type(rng, task_mix, target_counts, current_counts)
        evidence_docs: list[dict] = []
        retrieval_query: str | None = None
        retrieval_seed_doc_id: str | None = None
        worldbuilding_topic: dict[str, Any] | None = None
        if task_type == "worldbuilding_qa":
            if worldbuilding_queue:
                worldbuilding_topic = worldbuilding_queue.pop(0)
            else:
                worldbuilding_topic = sample_worldbuilding_topic(worldbuilding_topics, rng)
        else:
            if evidence_mode == "retrieval":
                seed_doc = rng.choice(documents)
                retrieval_seed_doc_id = seed_doc.get("id")
                retrieval_query = build_retrieval_seed_query(
                    seed_doc,
                    max_chars=seed_query_max_chars,
                )
                if not retrieval_query:
                    retrieval_query = str(seed_doc.get("search_text") or seed_doc.get("clean_text") or "")
                if retriever is None:
                    retriever = load_retriever(
                        device=args.device,
                        use_reranker=not args.no_reranker,
                    )
                retrieval_started = time.time()
                evidence_docs = retrieve_evidence_documents(
                    retriever,
                    query=retrieval_query,
                    top_k=retrieval_top_k,
                )
                retrieval_seconds = time.time() - retrieval_started
                if not evidence_docs:
                    evidence_docs = [seed_doc]
            else:
                evidence_docs = sample_evidence_documents(
                    documents,
                    rng,
                    int(source_cfg["max_evidence_docs_per_request"]),
                )
        request_id = f"req-{request_offset + request_index:05d}"
        system_prompt, user_prompt = build_teacher_prompts(
            task_type=task_type,
            evidence_docs=evidence_docs,
            worldbuilding_topic=worldbuilding_topic,
            samples_per_request=samples_per_request,
        )

        prompt_record = {
            "request_id": request_id,
            "task_type": task_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "evidence_doc_ids": [doc["id"] for doc in evidence_docs],
            "evidence_mode": evidence_mode if task_type != "worldbuilding_qa" else "topic",
            "retrieval_query": retrieval_query,
            "retrieval_seed_doc_id": retrieval_seed_doc_id,
            "worldbuilding_topic": worldbuilding_topic,
        }
        (prompt_dir / f"{request_id}.json").write_text(
            json.dumps(prompt_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.dry_run:
            request_records.append(
                build_request_record(
                    request_id=request_id,
                    task_type=task_type,
                    evidence_docs=evidence_docs,
                    worldbuilding_topic=worldbuilding_topic,
                    evidence_mode=evidence_mode if task_type != "worldbuilding_qa" else "topic",
                    retrieval_query=retrieval_query,
                    retrieval_seed_doc_id=retrieval_seed_doc_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    raw_text=None,
                    parsed_ok=False,
                    accepted_samples=0,
                    latency_seconds=0.0,
                    error=None,
                )
            )
            progress.update(1)
            progress.set_postfix(
                task=task_type,
                prompts=len(request_records),
                evidence=len(evidence_docs),
                retrieval=f"{retrieval_seconds:.2f}s",
            )
            continue

        started = time.time()
        raw_text = None
        try:
            raw_text, _raw_payload = call_teacher_api(
                teacher_api,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            api_seconds = time.time() - started
            parsed = parse_teacher_json(raw_text)
            normalized = validate_and_normalize_samples(
                parsed,
                expected_task_type=task_type,
                evidence_docs=evidence_docs,
                worldbuilding_topic=worldbuilding_topic,
                request_id=request_id,
            )
            all_samples.extend(normalized)
            accepted_samples = len(normalized)
            parsed_ok = True
            latency = time.time() - started
            request_records.append(
                build_request_record(
                    request_id=request_id,
                    task_type=task_type,
                    evidence_docs=evidence_docs,
                    worldbuilding_topic=worldbuilding_topic,
                    evidence_mode=evidence_mode if task_type != "worldbuilding_qa" else "topic",
                    retrieval_query=retrieval_query,
                    retrieval_seed_doc_id=retrieval_seed_doc_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    raw_text=raw_text,
                    parsed_ok=True,
                    accepted_samples=accepted_samples,
                    latency_seconds=latency,
                )
            )
        except Exception as exc:  # noqa: BLE001
            request_error = str(exc)
            latency = time.time() - started
            request_records.append(
                build_request_record(
                    request_id=request_id,
                    task_type=task_type,
                    evidence_docs=evidence_docs,
                    worldbuilding_topic=worldbuilding_topic,
                    evidence_mode=evidence_mode if task_type != "worldbuilding_qa" else "topic",
                    retrieval_query=retrieval_query,
                    retrieval_seed_doc_id=retrieval_seed_doc_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    raw_text=raw_text,
                    parsed_ok=False,
                    accepted_samples=0,
                    latency_seconds=latency,
                    error=request_error,
                )
            )
        progress.update(1)
        progress.set_postfix(
            task=task_type,
            ok=parsed_ok,
            accepted=f"{len(all_samples)}/{target_total}",
            last=accepted_samples,
            evidence=len(evidence_docs),
            retrieval=f"{retrieval_seconds:.2f}s",
            api=f"{api_seconds:.2f}s",
            elapsed=f"{time.time() - request_started:.2f}s",
            failed=sum(1 for item in request_records if not item["parsed_ok"]),
        )

    progress.close()

    if args.dry_run:
        save_jsonl(raw_dir / "requests.dryrun.jsonl", request_records)
        summary = {
            "dry_run": True,
            "generated_prompts": len(request_records),
            "request_id_start": request_offset,
            "request_id_end": request_offset + max(0, len(request_records) - 1),
            "output_dir": str(output_dir),
            "task_type": args.only_task_type,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    save_jsonl(raw_dir / "requests.jsonl", existing_request_records + request_records)

    run_unique_samples = dedupe_samples(all_samples)
    if args.only_task_type:
        run_selected_samples = run_unique_samples[:target_total]
    else:
        run_selected_samples = select_balanced_subset(
            run_unique_samples,
            target_total=target_total,
            target_counts=target_counts,
            seed=seed,
        )
    unique_samples = dedupe_samples(existing_samples + run_selected_samples)
    splits = split_samples(
        unique_samples,
        train_ratio=float(dataset_cfg["train_ratio"]),
        val_ratio=float(dataset_cfg["val_ratio"]),
        seed=seed,
    )

    save_jsonl(output_dir / "all.jsonl", unique_samples)
    save_jsonl(output_dir / "train.jsonl", splits["train"])
    save_jsonl(output_dir / "val.jsonl", splits["val"])
    save_jsonl(output_dir / "test.jsonl", splits["test"])
    save_category_splits(
        output_dir,
        unique_samples,
        train_ratio=float(dataset_cfg["train_ratio"]),
        val_ratio=float(dataset_cfg["val_ratio"]),
        seed=seed,
    )

    task_counter = Counter(sample["task_type"] for sample in unique_samples)
    category_counter = Counter(
        sample.get("bucket") or sample.get("meta", {}).get("category", "unknown")
        for sample in unique_samples
    )
    stats: dict[str, Any] = {
        "target_total": target_total,
        "generated_total": len(all_samples),
        "run_deduped_total": len(run_unique_samples),
        "run_selected_total": len(run_selected_samples),
        "existing_total_before_merge": len(existing_samples),
        "deduped_total": len(unique_samples),
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "target_task_distribution": target_counts,
        "task_type_distribution": dict(task_counter),
        "category_distribution": dict(category_counter),
        "request_count": len(existing_request_records) + len(request_records),
        "request_count_this_run": len(request_records),
        "successful_requests": sum(1 for item in request_records if item["parsed_ok"]),
        "failed_requests": sum(1 for item in request_records if not item["parsed_ok"]),
        "successful_requests_total": sum(1 for item in existing_request_records + request_records if item.get("parsed_ok")),
        "failed_requests_total": sum(1 for item in existing_request_records + request_records if not item.get("parsed_ok")),
    }
    manifest = {
        "generator": "teacher-api-sft",
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "teacher_api": {
            "api_type": teacher_api.api_type,
            "base_url": teacher_api.base_url,
            "model": teacher_api.model,
            "api_key_env": teacher_api.api_key_env,
        },
        "cleaning": {
            "version": "v2",
            "worldbuilding_mode": "topic_driven",
            "other_modes": "evidence_grounded",
        },
        "stats": stats,
    }

    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
