#!/usr/bin/env python3
"""Train ASA's cross-encoder with grouped graded-relevance supervision.

The loss is query-group listwise rather than independent template pairs.  A
positive target distribution gives extra mass to candidates that directly
answer the query.  A hardest-negative margin keeps helpful candidates above
same-query zero-relevance distractors.  Model inputs remain query + document;
answers and gold evidence are never tokenized.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
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
LOGGER = logging.getLogger("train_coverage_reranker")

TEACHER_GAIN = {
    "direct_answer": 3.0,
    "partial_answer": 2.0,
    "supporting_context": 1.0,
    "hard_context": 0.0,
    "irrelevant": 0.0,
}
POSITIVE_LABELS = ("direct_answer", "partial_answer", "supporting_context")


def stable_candidate_key(query: str, text: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\n{query}\n{text}".encode("utf-8")).digest()


def select_group_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    max_group_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a compact query group while preserving teacher difficulty.

    The teacher's ranking is meaningful inside a label tier.  For a 16-item
    group we therefore keep roughly six graded positives, eight X hard
    contexts, and two I negatives.  Budgets scale proportionally for other
    group sizes and any unused slots are filled deterministically from the
    most useful remaining candidates.
    """

    by_label = {
        label: [item for item in candidates if str(item.get("label") or "") == label]
        for label in TEACHER_GAIN
    }
    if not any(by_label.values()):
        # Compatibility for legacy relevance-only dry runs.  Teacher-labelled
        # production data always takes the tier-aware branch above.
        if max_group_size <= 0 or len(candidates) <= max_group_size:
            return list(candidates)
        positives = [item for item in candidates if float(item["relevance"]) > 0]
        negatives = [item for item in candidates if float(item["relevance"]) <= 0]
        positives.sort(
            key=lambda item: (
                -float(item["relevance"]),
                stable_candidate_key(query, str(item["text"]), seed),
            )
        )
        negatives.sort(key=lambda item: stable_candidate_key(query, str(item["text"]), seed))
        positive_budget = min(len(positives), max(1, round(max_group_size * 0.375)))
        return (positives[:positive_budget] + negatives[: max_group_size - positive_budget])[
            :max_group_size
        ]

    def teacher_order(item: dict[str, Any]) -> tuple[int, bytes]:
        return (
            int(item.get("teacher_rank") or 10**9),
            stable_candidate_key(query, str(item["text"]), seed),
        )

    for label in by_label:
        by_label[label].sort(key=teacher_order)

    if max_group_size <= 0 or len(candidates) <= max_group_size:
        return list(candidates)

    positive_budget = max(1, round(max_group_size * 0.375))
    irrelevant_budget = max(1, round(max_group_size * 0.125))
    hard_budget = max_group_size - positive_budget - irrelevant_budget
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def take(items: list[dict[str, Any]], count: int) -> None:
        for item in items:
            if count <= 0:
                return
            marker = id(item)
            if marker in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(marker)
            count -= 1

    # First preserve D/P/S diversity, then fill the positive budget in gain and
    # teacher-rank order.  This avoids selecting six duplicate D snippets while
    # dropping a useful partial/supporting bridge.
    for label in POSITIVE_LABELS:
        take(by_label[label], 1)
    remaining_positive_budget = max(0, positive_budget - len(selected))
    remaining_positives = sorted(
        [item for label in POSITIVE_LABELS for item in by_label[label] if id(item) not in selected_ids],
        key=lambda item: (-float(item["relevance"]), teacher_order(item)),
    )
    take(remaining_positives, remaining_positive_budget)

    # X is explicitly labelled as high-confusion context and must dominate the
    # zero-gain portion; a small I slice keeps global irrelevance calibrated.
    take(by_label["hard_context"], hard_budget)
    take(by_label["irrelevant"], irrelevant_budget)

    # Sparse-label and ambiguous edge cases can leave budgets unused.  Fill in
    # priority order without changing any gain or label.
    leftovers = sorted(
        [item for item in candidates if id(item) not in selected_ids],
        key=lambda item: (
            0 if str(item.get("label")) == "hard_context" else
            1 if float(item["relevance"]) > 0 else 2,
            -float(item["relevance"]),
            teacher_order(item),
        ),
    )
    take(leftovers, max_group_size - len(selected))
    return selected


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_groups(path: Path, *, max_group_size: int, seed: int) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            query = str(record.get("query") or "").strip()
            teacher_ranking = {
                str(item.get("candidate_id") or ""): {
                    "label": str(item.get("label") or ""),
                    "rank": int(item.get("rank") or rank),
                }
                for rank, item in enumerate(
                    (record.get("teacher") or {}).get("ranking") or [], start=1
                )
            }
            candidates = []
            for candidate in record.get("candidates") or []:
                text = str(candidate.get("text") or "").strip()
                candidate_id = str(candidate.get("candidate_id") or "")
                if teacher_ranking:
                    teacher_item = teacher_ranking.get(candidate_id) or {}
                    label = str(teacher_item.get("label") or "")
                    if label not in TEACHER_GAIN:
                        raise ValueError(
                            f"{path}:{line_number} has incomplete/invalid teacher "
                            f"ranking for candidate {candidate_id!r}"
                        )
                    relevance = TEACHER_GAIN[label]
                    teacher_rank = int(teacher_item["rank"])
                else:
                    relevance = max(float(candidate.get("relevance") or 0.0), 0.0)
                    label = ""
                    teacher_rank = len(candidates) + 1
                if text:
                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "text": text,
                            "relevance": relevance,
                            "label": label,
                            "teacher_rank": teacher_rank,
                        }
                    )
            candidates = select_group_candidates(
                query, candidates, max_group_size=max_group_size, seed=seed
            )
            positives = sum(item["relevance"] > 0 for item in candidates)
            if not query or len(candidates) < 2 or positives == 0 or positives == len(candidates):
                LOGGER.warning("Skipping non-trainable group %s:%d", path, line_number)
                continue
            groups.append(
                {
                    "query": query,
                    "query_type": str(record.get("query_type") or "unknown"),
                    "source_name": str(record.get("source_name") or ""),
                    "story_family": str(record.get("story_family") or ""),
                    "candidates": candidates,
                }
            )
    return groups


def group_keys(groups: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(group["source_name"]), str(group["query"])) for group in groups}


def story_families(groups: list[dict[str, Any]]) -> set[str]:
    return {str(group.get("story_family") or "") for group in groups if group.get("story_family")}


class CoverageGroupDataset(Dataset[dict[str, Any]]):
    def __init__(self, groups: list[dict[str, Any]]) -> None:
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.groups[index]


@dataclass(slots=True)
class CoverageDataCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        queries: list[str] = []
        documents: list[str] = []
        relevances: list[float] = []
        group_sizes: list[int] = []
        for group in features:
            candidates = list(group["candidates"])
            group_sizes.append(len(candidates))
            queries.extend([str(group["query"])] * len(candidates))
            documents.extend(str(candidate["text"]) for candidate in candidates)
            relevances.extend(float(candidate["relevance"]) for candidate in candidates)
        inputs = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs["relevances"] = torch.tensor(relevances, dtype=torch.float32)
        inputs["group_sizes"] = torch.tensor(group_sizes, dtype=torch.long)
        return inputs


class CoverageRerankerTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        listwise_weight: float = 1.0,
        pairwise_weight: float = 0.75,
        score_gap_margin: float = 0.75,
        target_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.listwise_weight = listwise_weight
        self.pairwise_weight = pairwise_weight
        self.score_gap_margin = score_gap_margin
        self.target_temperature = target_temperature

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ) -> Any:
        group_sizes = inputs.pop("group_sizes")
        relevances = inputs.pop("relevances")
        scores = model(**inputs).logits.view(-1)
        group_losses: list[torch.Tensor] = []
        offset = 0
        for raw_group_size in group_sizes.tolist():
            group_size = int(raw_group_size)
            group_scores = scores[offset : offset + group_size]
            group_relevance = relevances[offset : offset + group_size].to(group_scores.device)
            offset += group_size
            positive_mask = group_relevance > 0
            negative_mask = ~positive_mask
            if not positive_mask.any() or not negative_mask.any():
                continue

            target = torch.softmax(
                group_relevance / max(self.target_temperature, 1e-6), dim=0
            )
            listwise_loss = -(target * F.log_softmax(group_scores, dim=0)).sum()

            positive_scores = group_scores[positive_mask]
            positive_weights = group_relevance[positive_mask]
            positive_weights = positive_weights / positive_weights.sum().clamp_min(1e-8)
            hardest_negative = group_scores[negative_mask].max()
            margin_losses = F.softplus(
                self.score_gap_margin - (positive_scores - hardest_negative)
            )
            pairwise_loss = (positive_weights * margin_losses).sum()
            group_losses.append(
                self.listwise_weight * listwise_loss + self.pairwise_weight * pairwise_loss
            )

        if not group_losses:
            loss = scores.sum() * 0.0
        else:
            loss = torch.stack(group_losses).mean()
        if return_outputs:
            return loss, {"scores": scores.detach()}
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


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = [len(group["candidates"]) for group in groups]
    positive_counts = [
        sum(float(candidate["relevance"]) > 0 for candidate in group["candidates"])
        for group in groups
    ]
    label_counts: dict[str, int] = {}
    for group in groups:
        for candidate in group["candidates"]:
            label = str(candidate.get("label") or "legacy")
            label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "groups": len(groups),
        "story_families": len(story_families(groups)),
        "mean_candidates": round(sum(candidate_counts) / max(len(candidate_counts), 1), 3),
        "mean_positives": round(sum(positive_counts) / max(len(positive_counts), 1), 3),
        "mean_labels": {
            label: round(count / max(len(groups), 1), 3)
            for label, count in sorted(label_counts.items())
        },
    }


def build_training_arguments(args: argparse.Namespace, *, has_eval: bool) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "do_train": True,
        "do_eval": has_eval,
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
        kwargs["eval_strategy"] = "steps" if has_eval else "no"
    else:
        kwargs["evaluation_strategy"] = "steps" if has_eval else "no"
    if has_eval:
        if args.save_steps % args.eval_steps != 0:
            raise ValueError("--save-steps must be a multiple of --eval-steps")
        kwargs.update(
            {
                "eval_steps": args.eval_steps,
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    if "ddp_find_unused_parameters" in signature.parameters:
        kwargs["ddp_find_unused_parameters"] = False
    return TrainingArguments(
        **{key: value for key, value in kwargs.items() if key in signature.parameters}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument(
        "--model-name-or-path", type=Path, default=Path("model/reranker/bge-reranker-v2-m3")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Pair token limit; keep aligned with online CrossEncoderReranker (default: 1024).",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-group-size", type=int, default=24)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=4e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.75)
    parser.add_argument("--score-gap-margin", type=float, default=0.75)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    args.train_file = resolve(args.train_file)
    args.eval_file = resolve(args.eval_file) if args.eval_file else None
    args.model_name_or_path = resolve(args.model_name_or_path)
    args.output_dir = resolve(args.output_dir)
    set_seed(args.seed)

    train_groups = read_groups(
        args.train_file, max_group_size=args.max_group_size, seed=args.seed
    )
    eval_groups = (
        read_groups(args.eval_file, max_group_size=args.max_group_size, seed=args.seed)
        if args.eval_file
        else []
    )
    if not train_groups:
        raise RuntimeError("No trainable coverage groups")
    overlap = group_keys(train_groups) & group_keys(eval_groups)
    family_overlap = story_families(train_groups) & story_families(eval_groups)
    if overlap or family_overlap:
        raise ValueError(
            f"train/eval leakage: query_groups={len(overlap)} story_families={sorted(family_overlap)}"
        )
    LOGGER.info("Train: %s", json.dumps(summarize(train_groups), ensure_ascii=False))
    if eval_groups:
        LOGGER.info("Eval: %s", json.dumps(summarize(eval_groups), ensure_ascii=False))

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    collator = CoverageDataCollator(tokenizer=tokenizer, max_length=args.max_length)
    training_args = build_training_arguments(args, has_eval=bool(eval_groups))
    if args.dry_run:
        batch = collator(train_groups[:1])
        print(
            json.dumps(
                {
                    "train": summarize(train_groups),
                    "eval": summarize(eval_groups),
                    "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
                    "training_args": training_args.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0

    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_name_or_path), trust_remote_code=True
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.config.use_cache = False
    trainer = CoverageRerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=CoverageGroupDataset(train_groups),
        eval_dataset=CoverageGroupDataset(eval_groups) if eval_groups else None,
        data_collator=collator,
        processing_class=tokenizer,
        listwise_weight=args.listwise_weight,
        pairwise_weight=args.pairwise_weight,
        score_gap_margin=args.score_gap_margin,
        target_temperature=args.target_temperature,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    LOGGER.info("Saved coverage reranker to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
