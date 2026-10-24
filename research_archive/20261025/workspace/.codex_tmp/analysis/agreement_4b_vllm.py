#!/usr/bin/env python3
"""Run base Qwen3.5-4B zero-shot verifier with a continuous support score."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams  # noqa: E402


DEFAULT_MODEL = "${REMOTE_STORE}/ASA-ArknightStoryAgent/model/qwen3.5-4b"
PROMPT = (
    "判断下面这段【剧情证据】是否直接支持这条【主张】。"
    "只输出一个词：支持 或 不支持。\n\n"
    "主张：{claim}\n\n"
    "剧情证据：{chunk}\n\n"
    "答案："
)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def logprob_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = getattr(value, "logprob", None)
    return float(value) if isinstance(value, (int, float)) else None


def first_step_logprobs(output: Any) -> dict[int, float]:
    values = getattr(output, "logprobs", None)
    if not values:
        return {}
    first = values[0] or {}
    result: dict[int, float] = {}
    for token_id, item in first.items():
        value = logprob_value(item)
        if value is not None:
            result[int(token_id)] = value
    return result


def classify_text(text: str) -> int:
    normalized = re.sub(r"\s+", "", text or "").lower()
    if "不支持" in normalized or "unsupported" in normalized or "not supported" in normalized:
        return 0
    if "支持" in normalized or "supported" in normalized:
        return 1
    return 0


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    pairs = load_jsonl(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    support_ids = tokenizer.encode("支持", add_special_tokens=False)
    unsupported_ids = tokenizer.encode("不支持", add_special_tokens=False)
    support_id = int(support_ids[-1])
    unsupported_id = int(unsupported_ids[-1])
    prompts = [
        PROMPT.format(claim=str(pair["claim"])[:200], chunk=str(pair["chunk_text"]))
        for pair in pairs
    ]
    # vLLM 0.26.0 caps returned sample logprobs at 20.
    sampling = SamplingParams(temperature=0.0, max_tokens=8, logprobs=20)
    start = time.time()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.time() - start
    rows: list[dict[str, Any]] = []
    for pair, request_output in zip(pairs, outputs):
        output = request_output.outputs[0]
        text = str(output.text or "").strip()
        lps = first_step_logprobs(output)
        support_lp = lps.get(support_id)
        unsupported_lp = lps.get(unsupported_id)
        if support_lp is not None and unsupported_lp is not None:
            score = sigmoid(support_lp - unsupported_lp)
            score_method = "first_token_logprob_diff"
        else:
            # The generated label is still a valid deterministic decision; use
            # it as a conservative fallback score when a target token falls
            # outside the returned top-100 logprob set.
            score = float(classify_text(text))
            score_method = "generated_label_fallback"
        pred = classify_text(text)
        row = dict(pair)
        row.update(
            {
                "fourb_verdict": "supported" if pred else "unsupported",
                "fourb_supports": bool(pred),
                "fourb_text": text,
                "fourb_support_score": float(score),
                "fourb_score_method": score_method,
                "fourb_support_logprob": support_lp,
                "fourb_unsupported_logprob": unsupported_lp,
                "fourb_first_token_id": int(output.token_ids[0]) if output.token_ids else None,
                "fourb_model": args.model,
                "fourb_prompt_chars": len(prompts[len(rows)]),
                "fourb_elapsed_seconds": elapsed,
                "judged_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        rows.append(row)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact(row) + "\n")
    print(
        json.dumps(
            {
                "output": args.output,
                "pairs": len(rows),
                "supported": sum(r["fourb_supports"] for r in rows),
                "unsupported": sum(not r["fourb_supports"] for r in rows),
                "score_logprob_diff": sum(
                    r["fourb_score_method"] == "first_token_logprob_diff" for r in rows
                ),
                "elapsed_seconds": elapsed,
                "pairs_per_second": len(rows) / elapsed if elapsed else None,
                "support_token_id": support_id,
                "unsupported_token_id": unsupported_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
