#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("train_evidence_chain_reranker")
HARD_NEGATIVE_TYPES = {
    "background_only",
    "answer_adjacent",
    "same_entity_distractor",
    "partial_answer",
    "misleading_chain",
    "weak_original_gold",
}
ANSWERABILITY_QUERY_TYPES = {"causality", "reasoning", "reveal", "mystery", "answerability"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            records.append(record)
    return records


def _maybe_prepend_chain_metadata(text: str, chain_structure: dict[str, Any] | None) -> str:
    """Prepend chain structure metadata to evidence text if structure info is available."""
    if not chain_structure:
        return text
    meta_parts = [
        f"[CHAIN_LEN={chain_structure.get('chain_length', '?')}]",
        f"[CAUSAL_ORDER={chain_structure.get('causal_order', 'unknown')}]",
        f"[EVIDENCE_TYPES=({'|'.join(chain_structure.get('evidence_types', []))})]",
    ]
    return f"{' '.join(meta_parts)}\n{text}"


def normalize_pairwise_record(record: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    query = str(record.get("query") or "").strip()
    positive = str(record.get("positive") or "").strip()
    negative = str(record.get("negative") or "").strip()
    if not query or not positive or not negative:
        LOGGER.warning("Skipping invalid pairwise record from %s: missing query/positive/negative", source)
        return None
    if positive == negative:
        LOGGER.warning("Skipping invalid pairwise record from %s: positive == negative", source)
        return None

    # Prepend chain structure metadata to positive/negative text if available
    positive_text = _maybe_prepend_chain_metadata(
        str(record.get("positive") or ""),
        record.get("positive_chain_structure"),
    )
    negative_text = _maybe_prepend_chain_metadata(
        str(record.get("negative") or ""),
        record.get("negative_chain_structure"),
    )
    if not positive_text or not negative_text:
        LOGGER.warning("Skipping invalid pairwise record from %s: empty after metadata", source)
        return None

    positive_score = float(record.get("positive_score", 1.0))
    negative_score = float(record.get("negative_score", 0.0))
    negative_type = str(record.get("negative_type") or "").strip()
    query_type = str(record.get("query_type") or "").strip()
    weight = max(positive_score - negative_score, 0.05)
    if negative_type in HARD_NEGATIVE_TYPES:
        weight *= 1.5
    if query_type in ANSWERABILITY_QUERY_TYPES:
        weight *= 1.25
    if negative_type == "background_only":
        weight *= 1.15
    weight = min(weight, 3.0)
    return {
        "query": query,
        "positive": positive_text,
        "negative": negative_text,
        "weight": weight,
        "negative_type": negative_type,
        "query_type": query_type,
        "source_name": record.get("source_name", source),
        "answer": record.get("answer", ""),
        "answer_evidence": record.get("answer_evidence", []),
        "answer_focus": record.get("answer_focus", ""),
        "positive_chain_role_tags": record.get("positive_chain_role_tags", []),
        "negative_chain_role_tags": record.get("negative_chain_role_tags", []),
    }


class PairwiseRerankerDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


@dataclass(slots=True)
class PairwiseDataCollator:
    tokenizer: Any
    max_length: int

    def _tokenize(self, queries: list[str], docs: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        queries = [str(item["query"]) for item in features]
        positives = [str(item["positive"]) for item in features]
        negatives = [str(item["negative"]) for item in features]
        weights = torch.tensor([float(item.get("weight", 1.0)) for item in features], dtype=torch.float32)

        pos_inputs = self._tokenize(queries, positives)
        neg_inputs = self._tokenize(queries, negatives)
        batch: dict[str, torch.Tensor] = {"weights": weights}
        for key, value in pos_inputs.items():
            batch[f"pos_{key}"] = value
        for key, value in neg_inputs.items():
            batch[f"neg_{key}"] = value
        return batch


class PairwiseRerankerTrainer(Trainer):
    def __init__(self, *args: Any, loss_type: str = "softplus", dpo_beta: float = 0.1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.dpo_beta = dpo_beta

    def compute_loss(self, model: Any, inputs: dict[str, torch.Tensor], return_outputs: bool = False, **_: Any) -> Any:
        device = next(model.parameters()).device
        weights = inputs.pop("weights").to(device)
        pos_inputs = {
            key.removeprefix("pos_"): value
            for key, value in inputs.items()
            if key.startswith("pos_")
        }
        neg_inputs = {
            key.removeprefix("neg_"): value
            for key, value in inputs.items()
            if key.startswith("neg_")
        }

        pos_scores = model(**pos_inputs).logits.view(-1)
        neg_scores = model(**neg_inputs).logits.view(-1)
        score_diff = pos_scores - neg_scores
        if self.loss_type == "dpo":
            loss = (-F.logsigmoid(self.dpo_beta * score_diff) * weights).mean()
        else:
            loss = (F.softplus(-score_diff) * weights).mean()

        if return_outputs:
            return loss, {
                "pos_scores": pos_scores.detach(),
                "neg_scores": neg_scores.detach(),
                "score_diff": score_diff.detach(),
            }
        return loss

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, None, None]:
        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def split_records(
    records: list[dict[str, Any]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("source_name") or ""), str(record.get("query") or ""))
        grouped.setdefault(key, []).append(record)
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    eval_group_count = int(len(groups) * eval_ratio)
    if eval_ratio > 0 and eval_group_count == 0 and len(groups) > 1:
        eval_group_count = 1
    if eval_group_count <= 0:
        return [record for group in groups for record in group], []
    eval_groups = groups[:eval_group_count]
    train_groups = groups[eval_group_count:]
    return (
        [record for group in train_groups for record in group],
        [record for group in eval_groups for record in group],
    )


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_query_type: dict[str, int] = {}
    by_negative_type: dict[str, int] = {}
    weights: list[float] = []
    for record in records:
        by_query_type[str(record.get("query_type") or "unknown")] = by_query_type.get(str(record.get("query_type") or "unknown"), 0) + 1
        by_negative_type[str(record.get("negative_type") or "unknown")] = by_negative_type.get(str(record.get("negative_type") or "unknown"), 0) + 1
        weights.append(float(record.get("weight", 1.0)))
    return {
        "query_types": dict(sorted(by_query_type.items())),
        "negative_types": dict(sorted(by_negative_type.items())),
        "weight_min": min(weights) if weights else None,
        "weight_max": max(weights) if weights else None,
        "weight_avg": sum(weights) / len(weights) if weights else None,
    }


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "do_train": True,
        "do_eval": args.eval_ratio > 0,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
        "report_to": args.report_to,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "save_safetensors": True,
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps" if args.eval_ratio > 0 else "no"
    else:
        kwargs["evaluation_strategy"] = "steps" if args.eval_ratio > 0 else "no"
    kwargs["eval_steps"] = args.eval_steps

    if "ddp_find_unused_parameters" in signature.parameters:
        kwargs["ddp_find_unused_parameters"] = False
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return TrainingArguments(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a cross-encoder reranker on evidence-chain pairwise data.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/evidence_chain_reranker/batch_v1/reranker_pairwise.jsonl"),
        help="Path to reranker_pairwise.jsonl exported by scripts/evidence_chain_dataset.py.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=Path,
        default=Path("model/reranker/bge-reranker-v2-m3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/reranker/bge-reranker-v2-m3-evidence-chain-answerability"),
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--loss-type", choices=("softplus", "dpo"), default="softplus")
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", help="Validate data/tokenizer/training args without loading the model or training.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    args.train_file = resolve_path(args.train_file)
    args.model_name_or_path = resolve_path(args.model_name_or_path)
    args.output_dir = resolve_path(args.output_dir)

    if not args.train_file.exists():
        raise FileNotFoundError(f"Training file does not exist: {args.train_file}")
    if not args.model_name_or_path.exists():
        raise FileNotFoundError(f"Reranker model does not exist: {args.model_name_or_path}")

    set_seed(args.seed)
    raw_records = read_jsonl(args.train_file)
    records = [
        normalized
        for index, record in enumerate(raw_records, start=1)
        if (normalized := normalize_pairwise_record(record, source=f"{args.train_file.name}:{index}")) is not None
    ]
    if not records:
        raise RuntimeError(f"No valid pairwise records found in {args.train_file}")

    train_records, eval_records = split_records(records, eval_ratio=args.eval_ratio, seed=args.seed)
    LOGGER.info("Loaded records: train=%d eval=%d source=%s", len(train_records), len(eval_records), args.train_file)
    LOGGER.info("Train distribution: %s", json.dumps(summarize_records(train_records), ensure_ascii=False))
    if eval_records:
        LOGGER.info("Eval distribution: %s", json.dumps(summarize_records(eval_records), ensure_ascii=False))

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    if args.dry_run:
        collator = PairwiseDataCollator(tokenizer=tokenizer, max_length=args.max_length)
        sample_batch = collator(train_records[: min(2, len(train_records))])
        training_args = build_training_arguments(args)
        summary = {
            "dry_run": True,
            "train_file": str(args.train_file),
            "base_model": str(args.model_name_or_path),
            "output_dir": str(args.output_dir),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "train_distribution": summarize_records(train_records),
            "eval_distribution": summarize_records(eval_records),
            "batch_keys": sorted(sample_batch),
            "pos_input_shape": list(sample_batch["pos_input_ids"].shape),
            "neg_input_shape": list(sample_batch["neg_input_ids"].shape),
            "effective_train_batch_size_per_process": training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps,
            "loss": args.loss_type,
            "dpo_beta": args.dpo_beta,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_name_or_path),
        trust_remote_code=True,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainer = PairwiseRerankerTrainer(
        model=model,
        args=build_training_arguments(args),
        train_dataset=PairwiseRerankerDataset(train_records),
        eval_dataset=PairwiseRerankerDataset(eval_records) if eval_records else None,
        data_collator=PairwiseDataCollator(tokenizer=tokenizer, max_length=args.max_length),
        loss_type=args.loss_type,
        dpo_beta=args.dpo_beta,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    manifest = {
        "base_model": str(args.model_name_or_path),
        "train_file": str(args.train_file),
        "output_dir": str(args.output_dir),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_distribution": summarize_records(train_records),
        "eval_distribution": summarize_records(eval_records),
        "max_length": args.max_length,
        "loss": "weighted_pairwise_dpo" if args.loss_type == "dpo" else "weighted_pairwise_softplus",
        "dpo_beta": args.dpo_beta if args.loss_type == "dpo" else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evidence_chain_reranker_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved fine-tuned reranker to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
