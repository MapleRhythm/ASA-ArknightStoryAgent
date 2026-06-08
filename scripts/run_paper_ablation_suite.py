#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON_BIN = Path(os.environ.get("PYTHON_BIN") or sys.executable)

LISTWISE_RELEASE = "data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/reranker_listwise.jsonl"
LISTWISE_EASY = "data/processed/evidence_chain_reranker/easy_hard_split_v1/easy_sample50_listwise.jsonl"
LISTWISE_HARD = "data/processed/evidence_chain_reranker/easy_hard_split_v1/hard_sample50_listwise.jsonl"
LISTWISE_FAILURE = "data/processed/evidence_chain_reranker/failure_hard_pool_v1/failure_hard_pool_listwise.jsonl"
LISTWISE_EVAL50 = "data/processed/eval50_recall_questions_listwise.jsonl"

INDEX_DIR = "indexes/arknights_story"
EMBEDDING_MODEL = "model/embeddings/bge-small-zh-v1.5"
MINIRAG_GRAPH = "indexes/arknights_story_minirag_v3/graph.json"
RERANKER_V6 = "model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch"
RERANKER_OLD = "model/reranker/bge-reranker-v2-m3-evidence-chain-answerability"

SODA_VERIFIER_DIR = "data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_merged"
SODA_BLACKBOX_DIR = "data/processed/llama_factory/soda_eval50_len1800_blackbox_v2_scoped_sweep_soda_lora_gpu3_merged"

ALL_STAGES = [
    "release",
    "easy_hard",
    "failure",
    "source_oracle",
    "prompt_replay",
    "soda_audit",
    "latency",
    "summary",
]


@dataclass
class CommandSpec:
    name: str
    command: list[str]
    log_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ablation suite used by the ASA RAG/SODA paper.")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_ablation_suite"))
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Run a small, fast validation subset.")
    parser.add_argument("--sample", type=int, default=None, help="Override static retrieval sample size.")
    parser.add_argument("--prompt-sample", type=int, default=None)
    parser.add_argument("--latency-repeat", type=int, default=None)
    parser.add_argument("--latency-warmup", type=int, default=None)
    parser.add_argument(
        "--only",
        default=",".join(ALL_STAGES),
        help=f"Comma-separated stages. Available: {','.join(ALL_STAGES)}",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate suite_summary.json, suite_report.md and asa_rag_soda_paper_suite.md from existing artifacts.",
    )
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def rel(path: Path | str) -> str:
    resolved = resolve(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_stage_set(raw: str) -> list[str]:
    stages = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [stage for stage in stages if stage not in ALL_STAGES]
    if invalid:
        raise SystemExit(f"Invalid --only stage(s): {', '.join(invalid)}")
    return stages


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CONDA_NO_PLUGINS"] = "true"
    env["DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT / '.python_packages' / 'train'}:"
        f"{PROJECT_ROOT / 'src'}:"
        f"{PROJECT_ROOT}"
        f"{':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''}"
    )
    return env


def static_common_args(args: argparse.Namespace, top_ks: str, sample: int | None) -> list[str]:
    common = [
        "--device",
        args.device,
        "--top-ks",
        top_ks,
        "--index-dir",
        INDEX_DIR,
        "--embedding-model",
        EMBEDDING_MODEL,
        "--dense-top-k",
        "120",
        "--sparse-top-k",
        "120",
        "--minirag-top-k",
        "120",
        "--fusion-top-k",
        "80",
        "--reranker-candidate-top-k",
        "120",
        "--rerank-batch-size",
        "4",
        "--minirag-mode-weights",
        "fact=0.5,relation=1.0,causality=1.0,reveal=1.0,reasoning=0.75,mystery=0.75,answerability=0.75",
    ]
    if sample:
        common.extend(["--sample", str(sample)])
    return common


def retrieval_cmd(
    args: argparse.Namespace,
    *,
    listwise: str,
    output: Path,
    tag: str,
    top_ks: str,
    sample: int | None,
    reranker: str = RERANKER_V6,
    minirag: bool = True,
    neighbor: bool = True,
    skip_rerank: bool = False,
    oracle_sources: bool = False,
    include_records: bool = False,
) -> list[str]:
    cmd = [
        str(args.python_bin),
        "scripts/evaluate_retrieval_recall.py",
        "--listwise",
        listwise,
        "--output",
        str(output),
        "--tag",
        tag,
        "--reranker-model",
        reranker,
        *static_common_args(args, top_ks, sample),
    ]
    if minirag:
        cmd.extend(["--minirag-index", MINIRAG_GRAPH])
    if neighbor:
        cmd.append("--enable-neighbor-expansion")
        cmd.extend(["--neighbor-max-seed-docs", "16"])
    if skip_rerank:
        cmd.append("--skip-rerank")
    if oracle_sources:
        cmd.append("--oracle-sources")
    if include_records:
        cmd.append("--include-records")
    return cmd


def add_static_variant_commands(
    commands: list[CommandSpec],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    prefix: str,
    listwise: str,
    top_ks: str,
    sample: int | None,
    tag_prefix: str = "",
) -> None:
    variants = [
        ("full_current_v6", {"reranker": RERANKER_V6, "minirag": True, "neighbor": True}),
        ("no_neighbor_v6", {"reranker": RERANKER_V6, "minirag": True, "neighbor": False}),
        ("no_minirag_v6", {"reranker": RERANKER_V6, "minirag": False, "neighbor": True}),
        ("old_reranker_full", {"reranker": RERANKER_OLD, "minirag": True, "neighbor": True}),
        ("skip_rerank_full", {"reranker": RERANKER_V6, "minirag": True, "neighbor": True, "skip_rerank": True}),
    ]
    for variant, kwargs in variants:
        tag = f"{tag_prefix}{variant}"
        output = output_dir / f"{tag}.json"
        commands.append(
            CommandSpec(
                name=f"{prefix}:{tag}",
                command=retrieval_cmd(
                    args,
                    listwise=listwise,
                    output=output,
                    tag=tag,
                    top_ks=top_ks,
                    sample=sample,
                    **kwargs,
                ),
                log_name=f"{prefix}_{tag}.log",
            )
        )


def build_commands(args: argparse.Namespace, run_dir: Path, stages: list[str]) -> list[CommandSpec]:
    commands: list[CommandSpec] = []
    static_sample = args.sample if args.sample is not None else (3 if args.smoke else None)
    release_sample = args.sample if args.sample is not None else (3 if args.smoke else 50)
    prompt_sample = args.prompt_sample if args.prompt_sample is not None else (2 if args.smoke else None)
    latency_repeat = args.latency_repeat if args.latency_repeat is not None else (1 if args.smoke else 5)
    latency_warmup = args.latency_warmup if args.latency_warmup is not None else (0 if args.smoke else 2)
    static_top_ks = "1,5" if args.smoke else "1,5,10,20,32"

    if "release" in stages:
        add_static_variant_commands(
            commands,
            args,
            output_dir=run_dir / "ablation_release",
            prefix="release",
            listwise=LISTWISE_RELEASE,
            top_ks=static_top_ks,
            sample=release_sample,
        )

    if "easy_hard" in stages:
        for split, listwise in (("easy", LISTWISE_EASY), ("hard", LISTWISE_HARD)):
            add_static_variant_commands(
                commands,
                args,
                output_dir=run_dir / "ablation_easy_hard",
                prefix=f"easy_hard:{split}",
                listwise=listwise,
                top_ks=static_top_ks,
                sample=static_sample,
                tag_prefix=f"{split}_",
            )

    if "failure" in stages:
        add_static_variant_commands(
            commands,
            args,
            output_dir=run_dir / "ablation_failure_hard",
            prefix="failure",
            listwise=LISTWISE_FAILURE,
            top_ks=static_top_ks,
            sample=static_sample,
            tag_prefix="failure_",
        )

    if "source_oracle" in stages:
        top_ks = "1,3,5,8,10,12,20,32,50,80,120"
        output = run_dir / "source_oracle" / "eval50_minirag_v3_prerank_oracle.json"
        commands.append(
            CommandSpec(
                name="source_oracle:eval50_minirag_v3",
                command=retrieval_cmd(
                    args,
                    listwise=LISTWISE_EVAL50,
                    output=output,
                    tag="eval50_minirag_v3_prerank_oracle",
                    top_ks=top_ks,
                    sample=static_sample,
                    reranker=RERANKER_V6,
                    minirag=True,
                    neighbor=True,
                    oracle_sources=True,
                    include_records=not args.smoke,
                ),
                log_name="source_oracle_eval50_minirag_v3.log",
            )
        )

    if "prompt_replay" in stages:
        output = run_dir / "prompt_replay" / "scoped_sweep_replay.json"
        cmd = [
            str(args.python_bin),
            "scripts/evaluate_trace_replay_prompt_gold_recall.py",
            "--trace",
            f"{SODA_VERIFIER_DIR}/teacher_full_chain.jsonl",
            "--audit-records",
            f"{SODA_BLACKBOX_DIR}/audit_records.jsonl",
            "--runtime-config",
            "configs/runtime_inference_gpu.json",
            "--output",
            str(output),
            "--device",
            args.device,
            "--modes",
            "current,no_scoped_no_sweep",
            "--top-ks",
            "1,3,5,8,10,12",
        ]
        if prompt_sample:
            cmd.extend(["--sample", str(prompt_sample)])
        commands.append(CommandSpec("prompt_replay:scoped_sweep", cmd, "prompt_replay_scoped_sweep.log"))

    if "soda_audit" in stages:
        commands.append(
            CommandSpec(
                name="soda_audit:api_verifier",
                command=[
                    str(args.python_bin),
                    "scripts/analyze_soda_api_verifier_dataset.py",
                    "--dataset-dir",
                    SODA_VERIFIER_DIR,
                    "--output",
                    str(run_dir / "soda_audit" / "api_verifier_audit.md"),
                ],
                log_name="soda_audit_api_verifier.log",
            )
        )
        commands.append(
            CommandSpec(
                name="soda_audit:gold_topk",
                command=[
                    str(args.python_bin),
                    "scripts/analyze_soda_gold_evidence_topk.py",
                    "--audit-records",
                    f"{SODA_BLACKBOX_DIR}/audit_records.jsonl",
                    "--output",
                    str(run_dir / "soda_audit" / "gold_topk.json"),
                ],
                log_name="soda_audit_gold_topk.log",
            )
        )

    if "latency" in stages:
        output = run_dir / "latency" / f"retrieval_latency_{args.device.replace(':', '_')}.json"
        commands.append(
            CommandSpec(
                name="latency:retrieval",
                command=[
                    str(args.python_bin),
                    "scripts/benchmark_retrieval_latency.py",
                    "--device",
                    args.device,
                    "--query",
                    "烛煌的真实身份是什么？",
                    "--query",
                    "Logos和菈玛莲是什么关系？",
                    "--query",
                    "沙卒在萨尔贡黑市的地位和影响力如何？",
                    "--warmup",
                    str(latency_warmup),
                    "--repeat",
                    str(latency_repeat),
                    "--output",
                    str(output),
                ],
                log_name="latency_retrieval.log",
            )
        )

    if "summary" in stages:
        commands.append(
            CommandSpec(
                name="summary:paper_experiments",
                command=[
                    str(args.python_bin),
                    "scripts/summarize_paper_experiments.py",
                    "--output-dir",
                    str(run_dir / "paper_experiments"),
                ],
                log_name="summary_paper_experiments.log",
            )
        )

    return commands


def run_command(spec: CommandSpec, *, run_dir: Path, env: dict[str, str], dry_run: bool) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / (Path(spec.log_name).stem + ".stdout.log")
    stderr_path = logs_dir / (Path(spec.log_name).stem + ".stderr.log")
    started = time.perf_counter()
    record: dict[str, Any] = {
        "name": spec.name,
        "command": spec.command,
        "stdout": rel(stdout_path),
        "stderr": rel(stderr_path),
        "dry_run": dry_run,
    }
    print(f"[suite] {spec.name}", flush=True)
    if dry_run:
        record.update({"returncode": 0, "elapsed_seconds": 0.0})
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return record

    proc = subprocess.run(
        spec.command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    record.update(
        {
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stdout_tail": proc.stdout[-1200:],
            "stderr_tail": proc.stderr[-2000:],
        }
    )
    return record


def write_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def metric_row(path: Path) -> dict[str, Any] | None:
    payload = read_json_if_exists(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("overall"), dict):
        return None
    overall = payload["overall"]
    recall = overall.get("recall") if isinstance(overall.get("recall"), dict) else {}
    count = overall.get("count")
    missed = overall.get("missed")
    return {
        "tag": payload.get("tag") or path.stem,
        "path": rel(path),
        "count": count,
        "missed": missed,
        "mrr": overall.get("mrr"),
        "mean_first_hit_rank": overall.get("mean_first_hit_rank"),
        "r@1": recall.get("@1"),
        "r@5": recall.get("@5"),
        "r@10": recall.get("@10"),
        "r@20": recall.get("@20"),
        "r@32": recall.get("@32"),
        "wall_seconds": payload.get("wall_seconds"),
    }


def summarize_static_dir(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for item in sorted(path.glob("*.json")):
        row = metric_row(item)
        if row:
            rows.append(row)
    return rows


def summarize_source_oracle(path: Path) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(path.glob("*.json")) if path.exists() else []:
        payload = read_json_if_exists(item)
        if not isinstance(payload, dict):
            continue
        sources = (payload.get("source_oracle") or {}).get("sources") if isinstance(payload.get("source_oracle"), dict) else {}
        if not isinstance(sources, dict):
            continue
        for source, stats in sources.items():
            recall = stats.get("recall") if isinstance(stats.get("recall"), dict) else {}
            rows.append(
                {
                    "run": payload.get("tag") or item.stem,
                    "source": source,
                    "count": stats.get("count"),
                    "missed": stats.get("missed"),
                    "mrr": stats.get("mrr"),
                    "r@5": recall.get("@5"),
                    "r@32": recall.get("@32"),
                    "r@120": recall.get("@120"),
                    "path": rel(item),
                }
            )
    return rows


def summarize_prompt_replay(path: Path) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(path.glob("*.json")) if path.exists() else []:
        if "_shard" in item.stem:
            continue
        payload = read_json_if_exists(item)
        if not isinstance(payload, dict):
            continue
        for mode, data in (payload.get("modes") or {}).items():
            counts = data.get("counts", {})
            coverage = data.get("prompt_gold_coverage", {})
            rows.append(
                {
                    "mode": mode,
                    "prompts": counts.get("prompts"),
                    "questions": counts.get("questions"),
                    "gold_units": counts.get("gold_units"),
                    "gold_unit@12": ((data.get("gold_unit_cumulative") or {}).get("@12") or {}).get("ratio"),
                    "prompt_any@12": ((data.get("prompt_any_gold_cumulative") or {}).get("@12") or {}).get("ratio"),
                    "prompt_all@12": ((data.get("prompt_all_gold_cumulative") or {}).get("@12") or {}).get("ratio"),
                    "coverage_mean": coverage.get("mean"),
                    "path": rel(item),
                }
            )
    return rows


def summarize_latency(path: Path) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(path.glob("*.json")) if path.exists() else []:
        payload = read_json_if_exists(item)
        if not isinstance(payload, dict):
            continue
        env = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
        stage = payload.get("stage_latency_summary") if isinstance(payload.get("stage_latency_summary"), dict) else {}
        rows.append(
            {
                "device": env.get("device"),
                "query_count": env.get("query_count"),
                "repeat": env.get("repeat"),
                "end_to_end_mean_ms": (stage.get("end_to_end_s") or {}).get("mean_ms"),
                "rerank_mean_ms": (stage.get("rerank_s") or {}).get("mean_ms"),
                "sparse_mean_ms": (stage.get("sparse_total_s") or {}).get("mean_ms"),
                "path": rel(item),
            }
        )
    return rows


def write_suite_report(run_dir: Path, manifest: dict[str, Any]) -> None:
    summary = {
        "run_id": manifest.get("run_id"),
        "run_dir": manifest.get("run_dir"),
        "success": manifest.get("success"),
        "smoke": manifest.get("smoke"),
        "commands": [
            {
                "name": command.get("name"),
                "returncode": command.get("returncode"),
                "elapsed_seconds": command.get("elapsed_seconds"),
            }
            for command in manifest.get("commands", [])
        ],
        "ablation_release": summarize_static_dir(run_dir / "ablation_release"),
        "ablation_easy_hard": summarize_static_dir(run_dir / "ablation_easy_hard"),
        "ablation_failure_hard": summarize_static_dir(run_dir / "ablation_failure_hard"),
        "source_oracle": summarize_source_oracle(run_dir / "source_oracle"),
        "prompt_replay": summarize_prompt_replay(run_dir / "prompt_replay"),
        "latency": summarize_latency(run_dir / "latency"),
        "soda_audit": {
            "api_verifier_audit": rel(run_dir / "soda_audit" / "api_verifier_audit.md")
            if (run_dir / "soda_audit" / "api_verifier_audit.md").exists()
            else None,
            "gold_topk": rel(run_dir / "soda_audit" / "gold_topk.json")
            if (run_dir / "soda_audit" / "gold_topk.json").exists()
            else None,
        },
        "paper_experiments": {
            "summary": rel(run_dir / "paper_experiments" / "summary.json")
            if (run_dir / "paper_experiments" / "summary.json").exists()
            else None,
            "paper": rel(run_dir / "paper_experiments" / "asa_rag_soda_paper.md")
            if (run_dir / "paper_experiments" / "asa_rag_soda_paper.md").exists()
            else None,
        },
    }
    (run_dir / "suite_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Paper Ablation Suite Report",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- success: `{summary['success']}`",
        f"- smoke: `{summary['smoke']}`",
        "",
        "## Commands",
        "",
        table(
            ["Name", "Return", "Elapsed s"],
            [[row["name"], row.get("returncode"), row.get("elapsed_seconds")] for row in summary["commands"]],
        ),
    ]
    for title, key in (
        ("Release Static Retrieval", "ablation_release"),
        ("Easy/Hard Static Retrieval", "ablation_easy_hard"),
        ("Failure-Hard Static Retrieval", "ablation_failure_hard"),
    ):
        rows = summary[key]
        if not rows:
            continue
        lines.extend(["", f"## {title}", ""])
        lines.append(
            table(
                ["Tag", "Count", "Missed", "MRR", "R@1", "R@5", "R@10", "R@32", "Wall s"],
                [
                    [
                        f"`{row['tag']}`",
                        row.get("count"),
                        row.get("missed"),
                        fmt(row.get("mrr")),
                        fmt(row.get("r@1")),
                        fmt(row.get("r@5")),
                        fmt(row.get("r@10")),
                        fmt(row.get("r@32")),
                        fmt(row.get("wall_seconds"), 2),
                    ]
                    for row in rows
                ],
            )
        )
    if summary["source_oracle"]:
        lines.extend(["", "## Source Oracle", ""])
        lines.append(
            table(
                ["Run", "Source", "Count", "Missed", "MRR", "R@5", "R@32", "R@120"],
                [
                    [
                        f"`{row['run']}`",
                        row.get("source"),
                        row.get("count"),
                        row.get("missed"),
                        fmt(row.get("mrr")),
                        fmt(row.get("r@5")),
                        fmt(row.get("r@32")),
                        fmt(row.get("r@120")),
                    ]
                    for row in summary["source_oracle"]
                ],
            )
        )
    if summary["prompt_replay"]:
        lines.extend(["", "## Prompt Replay", ""])
        lines.append(
            table(
                ["Mode", "Prompts", "Questions", "Gold units", "Gold @12", "Any @12", "All @12", "Coverage"],
                [
                    [
                        f"`{row['mode']}`",
                        row.get("prompts"),
                        row.get("questions"),
                        row.get("gold_units"),
                        fmt(row.get("gold_unit@12")),
                        fmt(row.get("prompt_any@12")),
                        fmt(row.get("prompt_all@12")),
                        fmt(row.get("coverage_mean")),
                    ]
                    for row in summary["prompt_replay"]
                ],
            )
        )
    if summary["latency"]:
        lines.extend(["", "## Latency", ""])
        lines.append(
            table(
                ["Device", "Queries", "Repeat", "E2E ms", "Sparse ms", "Rerank ms"],
                [
                    [
                        row.get("device"),
                        row.get("query_count"),
                        row.get("repeat"),
                        fmt(row.get("end_to_end_mean_ms"), 3),
                        fmt(row.get("sparse_mean_ms"), 3),
                        fmt(row.get("rerank_mean_ms"), 3),
                    ]
                    for row in summary["latency"]
                ],
            )
        )
    lines.extend(["", "## Generated Files", ""])
    for group in ("soda_audit", "paper_experiments"):
        for _, path in summary[group].items():
            if path:
                lines.append(f"- `{path}`")
    (run_dir / "suite_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    paper_source = PROJECT_ROOT / "docs" / "asa_rag_soda_paper.md"
    if paper_source.exists():
        paper_text = paper_source.read_text(encoding="utf-8").rstrip()
        appendix = "\n".join(lines)
        (run_dir / "asa_rag_soda_paper_suite.md").write_text(
            paper_text
            + "\n\n---\n\n# 本次全量消融批次附录\n\n"
            + appendix
            + "\n",
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    stages = parse_stage_set(args.only)
    run_dir = resolve(args.output_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        manifest_path = run_dir / "manifest.json"
        manifest = read_json_if_exists(manifest_path)
        if not isinstance(manifest, dict):
            raise FileNotFoundError(rel(manifest_path))
        write_suite_report(run_dir, manifest)
        print(json.dumps({"run_dir": rel(run_dir), "report_only": True}, ensure_ascii=False))
        return 0

    commands = build_commands(args, run_dir, stages)
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "run_dir": rel(run_dir),
        "smoke": bool(args.smoke),
        "stages": stages,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_bin": str(args.python_bin),
        "device": args.device,
        "commands": [],
    }
    write_manifest(run_dir, manifest)

    env = base_env()
    failed = False
    for spec in commands:
        record = run_command(spec, run_dir=run_dir, env=env, dry_run=args.dry_run)
        manifest["commands"].append(record)
        write_manifest(run_dir, manifest)
        if record.get("returncode") != 0:
            failed = True
            print(f"[suite] failed: {spec.name}", file=sys.stderr)
            print(str(record.get("stderr_tail") or "")[-1200:], file=sys.stderr)
            if not args.continue_on_error:
                break

    manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["success"] = not failed
    write_manifest(run_dir, manifest)
    write_suite_report(run_dir, manifest)
    print(json.dumps({"run_dir": rel(run_dir), "success": not failed, "commands": len(manifest["commands"])}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
