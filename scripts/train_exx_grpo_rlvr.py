#!/usr/bin/env python3
"""Train grounded_action_exx_v1 with GRPO and verifiable rewards.

This is intentionally a fixed-evidence RLVR stage.  It optimizes protocol
validity, action selection, and teacher evidence-ID selection; it does not
claim to verify the semantic entailment between arbitrary fact text and an
evidence passage.  The latter needs an additional verifier or an interactive
retrieval environment.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVIDENCE_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
ACTIONS = ("answer_directly", "retrieve_more", "abstain")
FORBIDDEN = {"quote", "final_answer", "inferred_facts", "evidence_refs", "answer"}
FOLLOW_UP_REQUIRED = {"question", "query_type", "entities", "keywords", "expected_answer_type"}
FOLLOW_UP_OPTIONAL = {"dialogue_context"}


@dataclass(frozen=True)
class TrainingExample:
    row_id: str
    prompt: list[dict[str, str]]
    prompt_tokens: int
    visible_ids: list[str]
    gold_action: str
    gold_evidence_ids: list[str]


def parse_json_object(text: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(str(text or "").strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, Mapping):
            return str(last.get("content") or "")
    if isinstance(completion, Mapping):
        return str(completion.get("content") or "")
    return str(completion or "")


def validate_payload(payload: dict[str, Any] | None, visible_ids: set[str]) -> list[str]:
    """Return strict grounded_action_exx_v1 structural problems."""
    if payload is None:
        return ["invalid_json"]
    problems: list[str] = []
    legacy = FORBIDDEN.intersection(payload)
    if legacy:
        problems.append("legacy_fields:" + ",".join(sorted(legacy)))
    action = str(payload.get("next_action") or "")
    if action not in ACTIONS:
        return [*problems, "invalid_action"]

    if action == "answer_directly":
        if set(payload) != {"next_action", "supported_facts"}:
            problems.append("answer_top_schema")
        facts = payload.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            return [*problems, "invalid_fact_count"]
        for index, fact in enumerate(facts, start=1):
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                problems.append(f"fact_{index}_schema")
                continue
            text = fact.get("fact")
            evidence_ids = fact.get("evidence_ids")
            if not isinstance(text, str) or not text.strip():
                problems.append(f"fact_{index}_empty")
            if (
                not isinstance(evidence_ids, list)
                or not 1 <= len(evidence_ids) <= 2
                or len({str(item) for item in evidence_ids}) != len(evidence_ids)
            ):
                problems.append(f"fact_{index}_id_count")
            elif any(str(item) not in visible_ids for item in evidence_ids):
                problems.append(f"fact_{index}_unknown_id")
    elif action == "retrieve_more":
        if set(payload) != {"next_action", "follow_up_hypothesis"}:
            problems.append("retrieve_top_schema")
        follow_up = payload.get("follow_up_hypothesis")
        if not isinstance(follow_up, dict):
            problems.append("missing_follow_up")
        else:
            keys = set(follow_up)
            if not FOLLOW_UP_REQUIRED.issubset(keys) or keys - FOLLOW_UP_REQUIRED - FOLLOW_UP_OPTIONAL:
                problems.append("follow_up_schema")
            if not isinstance(follow_up.get("question"), str) or not follow_up.get("question", "").strip():
                problems.append("follow_up_question")
            for key in ("query_type", "expected_answer_type"):
                if not isinstance(follow_up.get(key), str):
                    problems.append(f"follow_up_{key}")
            for key in ("entities", "keywords"):
                value = follow_up.get(key)
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    problems.append(f"follow_up_{key}")
            if "dialogue_context" in follow_up and not isinstance(follow_up["dialogue_context"], str):
                problems.append("follow_up_dialogue_context")
    else:
        if set(payload) != {"next_action", "reason"}:
            problems.append("abstain_top_schema")
        if not isinstance(payload.get("reason"), str) or not payload.get("reason", "").strip():
            problems.append("abstain_reason")
    return problems


def cited_evidence_ids(payload: dict[str, Any] | None) -> set[str]:
    if not payload or payload.get("next_action") != "answer_directly":
        return set()
    result: set[str] = set()
    facts = payload.get("supported_facts")
    if not isinstance(facts, list):
        return result
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("evidence_ids"), list):
            continue
        result.update(str(item) for item in fact["evidence_ids"])
    return result


def json_reward(completions: Sequence[Any], **_: Any) -> list[float]:
    return [1.0 if parse_json_object(completion_text(item)) is not None else 0.0 for item in completions]


def schema_reward(
    completions: Sequence[Any], visible_ids: Sequence[Sequence[str]], **_: Any
) -> list[float]:
    return [
        1.0
        if not validate_payload(parse_json_object(completion_text(item)), set(visible))
        else 0.0
        for item, visible in zip(completions, visible_ids, strict=True)
    ]


def action_reward(
    completions: Sequence[Any], gold_action: Sequence[str], **_: Any
) -> list[float]:
    result = []
    for item, gold in zip(completions, gold_action, strict=True):
        payload = parse_json_object(completion_text(item))
        result.append(1.0 if payload and payload.get("next_action") == gold else 0.0)
    return result


def evidence_selection_reward(
    completions: Sequence[Any],
    gold_action: Sequence[str],
    gold_evidence_ids: Sequence[Sequence[str]],
    **_: Any,
) -> list[float]:
    """Jaccard reward for hidden teacher E-IDs on answer_directly rows.

    Non-answer rows receive 1 only when their action is correct.  This keeps
    the reward defined for every action while avoiding a free reward merely
    for emitting no citations.
    """
    rewards: list[float] = []
    for item, gold, expected_values in zip(
        completions, gold_action, gold_evidence_ids, strict=True
    ):
        payload = parse_json_object(completion_text(item))
        predicted_action = str((payload or {}).get("next_action") or "")
        if gold != "answer_directly":
            rewards.append(1.0 if predicted_action == gold else 0.0)
            continue
        expected = set(expected_values)
        predicted = cited_evidence_ids(payload)
        union = expected | predicted
        rewards.append(len(expected & predicted) / len(union) if union else 0.0)
    return rewards


def premature_answer_penalty(
    completions: Sequence[Any], gold_action: Sequence[str], **_: Any
) -> list[float]:
    penalties = []
    for item, gold in zip(completions, gold_action, strict=True):
        payload = parse_json_object(completion_text(item))
        action = str((payload or {}).get("next_action") or "")
        penalties.append(-1.0 if gold != "answer_directly" and action == "answer_directly" else 0.0)
    return penalties


def tokenized_length(tokenizer: Any, prompt: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    else:
        input_ids = encoded
    if input_ids and isinstance(input_ids[0], Sequence) and not isinstance(input_ids[0], (str, bytes)):
        if len(input_ids) != 1:
            raise ValueError("expected one tokenized prompt")
        input_ids = input_ids[0]
    return len(input_ids)


def read_training_examples(
    path: Path, tokenizer: Any, *, max_prompt_tokens: int
) -> tuple[list[TrainingExample], Counter[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON array: {path}")
    examples: list[TrainingExample] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if row.get("task_type") != "grounded_action_generation":
            counts["skip_non_grounded"] += 1
            continue
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            counts["skip_conversations"] += 1
            continue
        user = str(conversations[0].get("value") or "")
        gold = parse_json_object(conversations[-1].get("value"))
        action = str((gold or {}).get("next_action") or "")
        visible = EVIDENCE_RE.findall(user)
        if action not in ACTIONS or not visible or validate_payload(gold, set(visible)):
            counts["skip_invalid_gold"] += 1
            continue
        prompt = [
            {"role": "system", "content": str(row.get("system") or "")},
            {"role": "user", "content": user},
        ]
        n_tokens = tokenized_length(tokenizer, prompt)
        if n_tokens > max_prompt_tokens:
            counts["skip_prompt_too_long"] += 1
            continue
        examples.append(
            TrainingExample(
                row_id=str(row.get("id") or row.get("task_id") or index),
                prompt=prompt,
                prompt_tokens=n_tokens,
                visible_ids=visible,
                gold_action=action,
                gold_evidence_ids=sorted(cited_evidence_ids(gold)),
            )
        )
        counts[f"keep:{action}"] += 1
    return examples, counts


def select_examples(
    examples: list[TrainingExample], *, max_rows: int, selection_order: str, seed: int
) -> list[TrainingExample]:
    if not max_rows or max_rows >= len(examples):
        selected = list(examples)
        random.Random(seed).shuffle(selected)
        return selected
    if selection_order == "random":
        selected = list(examples)
        random.Random(seed).shuffle(selected)
        return selected[:max_rows]

    groups: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        groups[example.gold_action].append(example)
    for values in groups.values():
        values.sort(
            key=lambda item: (item.prompt_tokens, item.row_id),
            reverse=selection_order == "stratified-longest",
        )
    selected: list[TrainingExample] = []
    while len(selected) < max_rows and any(groups.values()):
        for action in ACTIONS:
            if groups[action] and len(selected) < max_rows:
                selected.append(groups[action].pop(0))
    return selected


def adapter_key_coverage(model: Any, adapter_path: Path) -> tuple[int, int, list[str]]:
    from safetensors import safe_open

    adapter_file = adapter_path / "adapter_model.safetensors"
    with safe_open(adapter_file, framework="pt", device="cpu") as handle:
        expected = set(handle.keys())
    actual = {
        name.replace(".default.", ".")
        for name, _ in model.named_parameters()
        if ".lora_" in name and ".default." in name
    }
    missing = sorted(expected - actual)
    return len(expected) - len(missing), len(expected), missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--selection-order",
        choices=("random", "stratified-shortest", "stratified-longest"),
        default="random",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=10000)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--num-completions-to-print", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.num_generations < 2:
        raise ValueError("GRPO requires at least two generations")
    generation_batch_size = args.batch_size * args.gradient_accumulation_steps
    if generation_batch_size % args.num_generations:
        raise ValueError(
            "batch_size * gradient_accumulation_steps must be divisible by num_generations"
        )

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    examples, input_counts = read_training_examples(
        args.train_file, tokenizer, max_prompt_tokens=args.max_prompt_tokens
    )
    selected = select_examples(
        examples,
        max_rows=args.max_rows,
        selection_order=args.selection_order,
        seed=args.seed,
    )
    if len(selected) < args.num_generations:
        raise ValueError(f"too few selected examples: {len(selected)}")
    selected_counts = Counter(item.gold_action for item in selected)
    print(
        json.dumps(
            {
                "input_counts": dict(input_counts),
                "selected_rows": len(selected),
                "selected_actions": dict(selected_counts),
                "prompt_tokens": {
                    "min": min(item.prompt_tokens for item in selected),
                    "max": max(item.prompt_tokens for item in selected),
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    dataset = Dataset.from_list(
        [
            {
                "row_id": item.row_id,
                "prompt": item.prompt,
                "visible_ids": item.visible_ids,
                "gold_action": item.gold_action,
                "gold_evidence_ids": item.gold_evidence_ids,
            }
            for item in selected
        ]
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(args.sft_adapter),
        is_trainable=True,
        autocast_adapter_dtype=False,
    )
    loaded, total, missing = adapter_key_coverage(model, args.sft_adapter)
    print(f"adapter key coverage: {loaded}/{total}", flush=True)
    if missing:
        raise RuntimeError(f"adapter/model mismatch; first missing keys: {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "grounded_action_exx_v1",
        "method": "GRPO with verifiable rewards (fixed evidence)",
        "limitations": [
            "does not verify semantic entailment between arbitrary fact text and evidence",
            "does not execute interactive retrieval actions",
        ],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_counts": dict(input_counts),
        "selected_actions": dict(selected_counts),
        "selected_ids": [item.row_id for item in selected],
        "adapter_key_coverage": {"loaded": loaded, "total": total},
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=generation_batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        chat_template_kwargs={"enable_thinking": False},
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        torch_empty_cache_steps=1,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        mask_truncated_completions=True,
        reward_weights=[1.0, 1.0, 1.5, 1.0, 0.5],
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[
            json_reward,
            schema_reward,
            action_reward,
            evidence_selection_reward,
            premature_answer_penalty,
        ],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"SAVED {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
