#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger("transformers_peft.train_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3.5 with Transformers + PEFT.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "src" / "config" / "transformers_peft_train.yaml",
        help="Path to the training config YAML.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def normalize_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    if isinstance(tools, str):
        parsed = json.loads(tools)
        return parsed if parsed else None
    return tools


def count_trainable_parameters(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return trainable, total


def load_model(model_name_or_path: str, trust_remote_code: bool, torch_dtype: torch.dtype | None) -> Any:
    common_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_name_or_path,
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype is not None:
        common_kwargs["torch_dtype"] = torch_dtype

    try:
        from transformers import AutoModelForImageTextToText

        LOGGER.info("Loading model via AutoModelForImageTextToText.")
        return AutoModelForImageTextToText.from_pretrained(**common_kwargs)
    except Exception as first_error:
        LOGGER.warning("Falling back to AutoModelForCausalLM: %s", first_error)
        return AutoModelForCausalLM.from_pretrained(**common_kwargs)


class SupervisedDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [feature["labels"] for feature in features]
        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ],
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_length - len(label)))

        batch["labels"] = torch.tensor(padded_labels, dtype=batch["input_ids"].dtype)
        return batch


def build_training_arguments(config: dict[str, Any]) -> TrainingArguments:
    report_to = config.get("report_to", "none")
    if report_to in (None, "", []):
        report_to = "none"

    return TrainingArguments(
        output_dir=resolve_path(config["output_dir"]),
        run_name=config.get("run_name"),
        overwrite_output_dir=bool(config.get("overwrite_output_dir", False)),
        do_train=True,
        do_eval=bool(config.get("eval_file")),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=int(config.get("eval_steps", 100)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        num_train_epochs=float(config.get("num_train_epochs", 3.0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        logging_strategy="steps",
        logging_steps=int(config.get("logging_steps", 5)),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=int(config.get("save_steps", 100)),
        save_total_limit=int(config.get("save_total_limit", 4)),
        dataloader_num_workers=int(config.get("dataloader_num_workers", 2)),
        remove_unused_columns=False,
        report_to=report_to,
        bf16=bool(config.get("bf16", False)),
        fp16=bool(config.get("fp16", False)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        ddp_find_unused_parameters=bool(config.get("ddp_find_unused_parameters", False)),
        save_safetensors=True,
        seed=int(config.get("seed", 42)),
    )


def render_and_tokenize_example(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any] | None:
    messages = example.get("messages") or []
    tools = normalize_tools(example.get("tools"))
    if not messages:
        return None

    full_input_ids: list[int] = []
    labels: list[int] = []

    for index in range(len(messages)):
        current_ids = tokenizer.apply_chat_template(
            messages[: index + 1],
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        added_ids = current_ids[len(full_input_ids) :]
        full_input_ids = current_ids

        role = messages[index].get("role")
        if role == "assistant":
            labels.extend(added_ids)
        else:
            labels.extend([-100] * len(added_ids))

    if len(full_input_ids) != len(labels):
        raise ValueError(
            f"Token/label length mismatch for sample {example.get('id')}: "
            f"{len(full_input_ids)} != {len(labels)}"
        )

    full_input_ids = full_input_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        return None

    return {
        "input_ids": full_input_ids,
        "attention_mask": [1] * len(full_input_ids),
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    setup_logging()

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    LOGGER.info("Loaded config from %s", config_path)

    model_name_or_path = resolve_path(config["model_name_or_path"])
    train_file = resolve_path(config["train_file"])
    eval_file = resolve_path(config.get("eval_file"))
    max_length = int(config.get("max_length", 4096))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer.padding_side = "right"

    dtype: torch.dtype | None = None
    if config.get("bf16"):
        dtype = torch.bfloat16
    elif config.get("fp16"):
        dtype = torch.float16

    model = load_model(
        model_name_or_path=model_name_or_path,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        torch_dtype=dtype,
    )
    if hasattr(model.config, "use_cache") and config.get("gradient_checkpointing", True):
        model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.get("lora_rank", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=config.get("lora_bias", "none"),
        target_modules=config.get("lora_target_modules", "all-linear"),
        modules_to_save=config.get("modules_to_save"),
    )
    model = get_peft_model(model, lora_config)

    trainable_params, total_params = count_trainable_parameters(model)
    LOGGER.info(
        "LoRA model ready: trainable=%s total=%s ratio=%.4f%%",
        trainable_params,
        total_params,
        100 * trainable_params / max(total_params, 1),
    )

    data_files = {"train": train_file}
    if eval_file:
        data_files["validation"] = eval_file

    dataset_dict = load_dataset("json", data_files=data_files)
    LOGGER.info("Loaded dataset splits: %s", {k: len(v) for k, v in dataset_dict.items()})

    preprocessing_num_workers = int(config.get("preprocessing_num_workers", 1))

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        rendered = render_and_tokenize_example(
            example,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if rendered is None:
            return {"input_ids": None, "attention_mask": None, "labels": None}
        return rendered

    processed = dataset_dict.map(
        preprocess,
        remove_columns=dataset_dict["train"].column_names,
        num_proc=preprocessing_num_workers,
        desc="Formatting chat examples",
    )
    processed = processed.filter(
        lambda record: record["input_ids"] is not None and len(record["input_ids"]) > 0,
        desc="Dropping empty tokenized rows",
    )

    LOGGER.info("Processed dataset splits: %s", {k: len(v) for k, v in processed.items()})

    trainer = Trainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=processed["train"],
        eval_dataset=processed["validation"] if "validation" in processed else None,
        data_collator=SupervisedDataCollator(tokenizer),
        tokenizer=tokenizer,
    )

    resume_from_checkpoint = config.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        resume_from_checkpoint = resolve_path(resume_from_checkpoint)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(resolve_path(config["output_dir"]))


if __name__ == "__main__":
    main()
