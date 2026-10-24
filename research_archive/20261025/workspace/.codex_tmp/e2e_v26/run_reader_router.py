"""Pre-registered, answer-blind routing diagnostic on eight development cases.

No retrieval, production writes, API calls, label inheritance or output repair.
Uses a frozen copy of 241's runtime; model assets are hashed before and after.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import time

from selector_protocol import (
    CASE_IDS, SELECTOR_RULES, blocks, reader_prompt, strict_object, validate_selection,
)

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
RELEASE = ROOT / "release/ASA-ArknightStoryAgent"
SOURCE = HERE / "src"
GPU_UUID = "GPU-90144416-2626-05e7-0b89-a4e45559ddf9"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def gpu_idle():
    output = subprocess.check_output([
        "nvidia-smi", "-i", GPU_UUID, "--query-gpu=uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    uuid, memory, utilization = (value.strip() for value in output.split(","))
    processes = subprocess.check_output([
        "nvidia-smi", "-i", GPU_UUID, "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    if uuid != GPU_UUID or int(memory) >= 256 or int(utilization) >= 10 or processes:
        raise RuntimeError("GPU is not idle; no processes will be interrupted")
    return {"at_utc": datetime.now(timezone.utc).isoformat(),
            "memory_mib": int(memory), "utilization": int(utilization), "compute_pids": []}


def prepare():
    if (HERE / "preflight.json").exists():
        raise RuntimeError("preflight exists")
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import build_minimal_conclusion_prompt
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    from transformers import AutoTokenizer

    config = load(HERE / "runtime.json")
    assert config["retrieval"]["enable_minirag"] is False
    engine = config["generator"]["vllm"]
    base = (RELEASE / engine["base_model_path"]).resolve()
    adapter = (RELEASE / engine["lora_path"]).resolve()
    tokenizer_path = adapter if (adapter / "tokenizer_config.json").exists() else base
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    visible = load(HERE / "visible_answer_analysis.json")
    raw = [json.loads(line) for line in (HERE / "results.jsonl").read_text().splitlines() if line]
    assert len(raw) == len(visible["results"]) == 58
    assert visible["source_results_sha256"] == sha(HERE / "results.jsonl")
    rows = []
    for index, (source, saved) in enumerate(zip(visible["results"], raw, strict=True)):
        assert source["question"] == saved["question"]
        if source["id"] not in CASE_IDS:
            continue
        evidence = source["evidence_brief"]
        assert source["status"] == "ok"
        blocks(evidence)
        hypothesis = HypothesisDocument(**saved["retrieval_trace"][-1]["hypothesis"])
        empty, brief = build_minimal_conclusion_prompt(
            question=source["question"], current_hypothesis=hypothesis,
            evidence=[], prompt_evidence=[], current_round=2, max_retrieval_rounds=2,
            prompt_evidence_top_k=10, grounding_mode="quote",
        )
        assert brief == ""
        control = reader_prompt(empty, evidence)
        selection_prompt = render_qwen_chat_prompt(
            SELECTOR_RULES,
            json.dumps({"question": source["question"], "visible_evidence": evidence}, ensure_ascii=False),
        )
        seed = int(digest("reader-router-v27:" + source["id"])[:8], 16) % (2**31)
        row = {
            "id": source["id"], "question": source["question"],
            "evidence_brief": evidence, "evidence_sha256": digest(evidence),
            "hypothesis": asdict(hypothesis), "empty_reader_prompt": empty,
            "control_prompt": control, "selector_prompt": selection_prompt,
            "sampling_seed": seed,
            "reader_order": ["control", "routed"] if len(rows) % 2 == 0 else ["routed", "control"],
        }
        for name in ("control_prompt", "selector_prompt"):
            ids = tokenizer.encode(row[name], add_special_tokens=True)
            assert len(ids) + 2048 <= engine["max_model_len"], "No truncation allowed"
            row[name + "_sha256"] = digest(row[name])
            row[name + "_tokens"] = len(ids)
            row[name + "_token_ids_sha256"] = digest(json.dumps(ids))
        rows.append(row)
    assert [row["id"] for row in rows] == CASE_IDS
    files = [
        HERE / "runtime.json", HERE / "selector_protocol.py", Path(__file__),
        HERE / "test_selector_protocol.py", HERE / "results.jsonl",
        HERE / "visible_answer_analysis.json",
        HERE / "indexes/arknights_story/operator_aliases.json",
        *sorted(SOURCE.rglob("*.py")),
    ]
    assets = [
        path for folder in (base, adapter) for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".safetensors", ".txt", ".model", ".tiktoken"}
    ]
    manifest = {str(path): sha(path) for path in [*files, *assets]}
    report = {
        "version": "answer_blind_evidence_router8_v27",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "base_model": str(base), "adapter": str(adapter), "tokenizer": str(tokenizer_path),
        "runtime": engine, "generator": config["generator"], "sha256": manifest,
        "selection_rationale": "Known subject/causal failure cases 011/015/017/024/034; no-answer 031; positive 028 and partial 023 controls. Selected development diagnostic, not holdout.",
        "protocol": [
            "Eight selector calls use base without LoRA in the same LoRA-enabled engine.",
            "Both reader conditions use the unchanged actual SODA LoRA, seed, sampling and quote gate.",
            "Selector sees only original question and all reconstructed visible evidence, never answers/labels/source-review.",
            "Only selected whole original E blocks reach routed reader; selector reason and sufficiency are not transmitted.",
            "No relabeling, reordering, truncation, automatic E repair or quote threshold relaxation.",
            "Reader uses standardized minimal conclusion prompt at 2/2, not claimed identical to original E2E fallback prompt.",
            "No hidden-corpus/retrieval/API access; errors remain in eight-case denominator.",
            "All selector/parser/context errors preserved; no retries or output repair beyond known opening-brace prefill.",
            "Do not promote based on eight selected cases or mechanical pass rate.",
        ],
        "rows": rows,
    }
    save(HERE / "preflight.json", report)
    print(json.dumps({"prepared": len(rows), "longest_tokens": max(
        row[name] for row in rows for name in ("control_prompt_tokens", "selector_prompt_tokens")
    ), "preflight_sha256": sha(HERE / "preflight.json")}), flush=True)


def parse_prefilled(text):
    try:
        return strict_object(text), "strict"
    except (ValueError, json.JSONDecodeError):
        if text.lstrip().startswith("{"):
            raise
        return strict_object("{" + text), "known_opening_brace_prefill"


def run():
    if (HERE / "generations.progress.jsonl").exists():
        raise RuntimeError("generation journal exists; no duplicate generation")
    preflight = load(HERE / "preflight.json")
    for path, expected in preflight["sha256"].items():
        assert sha(Path(path)) == expected, path
    idle_samples = load(HERE / "idle_samples.json")
    assert len(idle_samples["samples"]) >= 13 and idle_samples["elapsed_seconds"] >= 60
    gpu_idle()
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    from asa_arknight_story_agent.inference.generation.conclusion_generation import (
        normalize_minimal_conclusion_output, parse_conclusion_output,
    )
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    import torch
    import vllm

    engine = preflight["runtime"]
    runner = VllmRunner(
        base_model_path=Path(preflight["base_model"]), lora_path=Path(preflight["adapter"]),
        tensor_parallel_size=1, gpu_memory_utilization=engine["gpu_memory_utilization"],
        max_model_len=engine["max_model_len"], max_tokens=2048,
        temperature=0.1, top_p=0.8, repeat_penalty=1.05,
        dtype=engine["dtype"], max_num_batched_tokens=engine["max_num_batched_tokens"],
        enforce_eager=engine["enforce_eager"],
    )
    llm, configured_adapter = runner._ensure_engine()
    assert configured_adapter is not None
    tokenizer = llm.get_tokenizer()
    original_generate = llm.generate
    state = {}
    calls = []

    def dispatch(prompts, params, **kwargs):
        assert len(prompts) == 1 and kwargs["lora_request"] is configured_adapter
        params.seed = state["seed"]
        effective = None if state["kind"] == "selector" else configured_adapter
        kwargs["lora_request"] = effective
        receipt = {
            "id": state["id"], "kind": state["kind"], "seed": params.seed,
            "lora_id": None if effective is None else effective.lora_int_id,
            "prompt_sha256": digest(prompts[0]), "actual_prompt": prompts[0],
        }
        calls.append(receipt)
        expected_ids = tokenizer.encode(prompts[0], add_special_tokens=True)
        result = original_generate(prompts, params, **kwargs)
        assert len(result) == len(result[0].outputs) == 1
        actual = result[0]
        assert list(actual.prompt_token_ids) == expected_ids
        echoed = getattr(actual, "lora_request", None)
        assert (None if echoed is None else echoed.lora_int_id) == receipt["lora_id"]
        receipt.update(
            prompt_tokens=len(expected_ids), prompt_token_ids_sha256=digest(json.dumps(expected_ids)),
            finish_reason=actual.outputs[0].finish_reason, generated_tokens=len(actual.outputs[0].token_ids),
            returned_lora_id=None if echoed is None else echoed.lora_int_id,
            raw_output_with_special_tokens=actual.outputs[0].text,
        )
        return result

    llm.generate = dispatch

    def generate(row, kind, prompt):
        state.update(id=row["id"], kind=kind, seed=row["sampling_seed"])
        before = len(calls)
        output = runner.generate(prompt, max_tokens=2048, temperature=0.1, top_p=0.8, repeat_penalty=1.05)
        assert len(calls) == before + 1
        call = calls[-1]
        assert call["finish_reason"] == "stop", "length result retained, not accepted"
        return output

    def read(row, kind, evidence):
        prompt = reader_prompt(row["empty_reader_prompt"], evidence)
        result = {"status": "error", "evidence_sha256": digest(evidence), "prompt_sha256": digest(prompt)}
        try:
            result["raw_output"] = generate(row, kind, prompt)
            hypothesis = HypothesisDocument(**row["hypothesis"])
            normalized = normalize_minimal_conclusion_output(result["raw_output"], "minimal")
            result["normalized_output"] = normalized
            parsed = parse_conclusion_output(normalized, question=row["question"],
                                             current_hypothesis=hypothesis, max_round_reached=True)
            result["parsed"] = asdict(parsed)
            checked = validate_conclusion_grounding(
                question=row["question"], hypothesis=hypothesis, evidence=[],
                conclusion=parsed, max_round_reached=True, mode="quote",
                evidence_prompt_text=evidence,
            )
            result.update(status="ok", grounded=asdict(checked))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    results = []
    with (HERE / "generations.progress.jsonl").open("x", encoding="utf-8") as journal:
        for row in preflight["rows"]:
            start = len(calls)
            result = {"id": row["id"], "question": row["question"], "selector": {"status": "error"}}
            selected = None
            try:
                text = generate(row, "selector", row["selector_prompt"])
                result["selector"]["raw_output"] = text
                payload, parsed_as = parse_prefilled(text)
                selected = validate_selection(payload, row["evidence_brief"])
                result["selector"].update(status="ok", payload=payload, parsed_as=parsed_as,
                                          selected_evidence=selected)
            except Exception as exc:
                result["selector"]["error"] = f"{type(exc).__name__}: {exc}"
            for kind in row["reader_order"]:
                if kind == "routed" and not selected:
                    result[kind] = {"status": "not_generated", "reason": "invalid_or_empty_selection"}
                else:
                    evidence = row["evidence_brief"] if kind == "control" else selected
                    result[kind] = read(row, kind, evidence)
            result["calls"] = calls[start:]
            results.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps({
                "completed": len(results), "total": 8, "id": row["id"],
                "selector": result["selector"]["status"],
                "selected": result["selector"].get("payload", {}).get("evidence_ids"),
                **{kind: result[kind].get("grounded", {}).get("next_action", result[kind]["status"])
                   for kind in ("control", "routed")},
            }, ensure_ascii=False), flush=True)
    changed = [path for path, expected in preflight["sha256"].items() if sha(Path(path)) != expected]
    report = {
        "version": preflight["version"], "evaluation_only": True, "training_ready": False,
        "rows": results, "preflight_sha256": sha(HERE / "preflight.json"),
        "postflight_files_unchanged": not changed, "changed_paths": changed,
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "actual_call_count": len(calls), "input_truncated": False, "output_budget_reduced": False,
        "production_changed": False, "semantic_review_complete": False,
    }
    save(HERE / "generations.json", report)
    if changed:
        raise SystemExit("frozen artifacts changed")


def idle_then_run():
    import fcntl
    lock = Path("${REMOTE_ASA_ROOT}/rank_v8_coverage/locks/gpu2_asa_e2e.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start, samples = time.monotonic(), []
        while True:
            samples.append(gpu_idle())
            print(json.dumps({"idle_sample": len(samples), **samples[-1]}), flush=True)
            if time.monotonic() - start >= 60:
                break
            time.sleep(5)
        save(HERE / "idle_samples.json", {"samples": samples, "elapsed_seconds": time.monotonic() - start})
        run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "idle_then_run"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else idle_then_run()
