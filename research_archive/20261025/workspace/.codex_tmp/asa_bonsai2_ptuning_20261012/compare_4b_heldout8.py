"""Bounded frozen-evidence comparison; never invokes retrieval or answer repair."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time


EXPECTED_ADAPTER_SHA = "20c369a0042478f50b4128c047eb03844bd0e74c8eccdf6139e3344a396ef0ce"


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--bonsai-results", type=Path, required=True)
    ap.add_argument("--historical-pipeline", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    adapter = args.repo / "model/lora/teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2"
    base = args.repo / "model/qwen3.5-4b"
    assert sha_file(adapter / "adapter_model.safetensors") == EXPECTED_ADAPTER_SHA
    selection = json.loads(args.selection.read_text())["ids"]
    indexed = {r["id"]: r for r in load_rows(args.data / "val.jsonl")}
    assert len(selection) == len(set(selection)) == 8
    rows = [indexed[i] for i in selection]
    bonsai_rows = load_rows(args.bonsai_results)
    assert len(bonsai_rows) == 24
    for row in rows:
        assert len(row["documents"]) == 20 and not row["truncated"]
        for doc in row["documents"]:
            assert sha_text(doc["text"]) == doc["sha256"]
            assert doc["text"] in row["prompt"]
        counterparts = [x for x in bonsai_rows if x["id"] == row["id"]]
        assert len(counterparts) == 3
        assert all(x["prompt_sha256"] == sha_text(row["prompt"]) for x in counterparts)
        assert all(x["evidence_count"] == 20 and not x["input_truncated"] for x in counterparts)

    # Historical prompt builder only. No invocation of its pipeline or validator.
    spec = importlib.util.spec_from_file_location("historical_4b_pipeline", args.historical_pipeline)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    native_prompts = {}
    for row in rows:
        evidence = [{"document": {"id": d["document_id"], "clean_text": d["text"]},
                     "prompt_prefer_clean_text": True} for d in row["documents"]]
        hyp = module.HypothesisDocument(
            question=row["question"], query_type="reasoning",
            expected_answer_type="剧情解释", intent="剧情解释",
            entities=[], keywords=[], dialogue_context="")
        current, total = map(int, row["round"].split("/"))
        prompt = module.build_conclusion_prompt(
            row["question"], hyp, evidence, [], current, total, 20,
            prompt_evidence=evidence, evidence_max_chars_per_doc=None,
            evidence_max_total_chars=None, prompt_mode="minimal")
        assert all(normalize(d["text"]) in normalize(prompt) for d in row["documents"])
        assert all(d["document_id"] in prompt for d in row["documents"])
        native_prompts[row["id"]] = prompt

    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "paired_frozen_evidence_reader_not_full_chain_or_unseen_benchmark",
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "question_ids": selection, "question_count": 8, "generation_count": 16,
        "base_model": str(base), "adapter": str(adapter),
        "adapter_sha256": EXPECTED_ADAPTER_SHA,
        "adapter_config_sha256": sha_file(adapter / "adapter_config.json"),
        "base_config_sha256": sha_file(base / "config.json"),
        "selection_sha256": sha_file(args.selection),
        "data_sha256": sha_file(args.data / "val.jsonl"),
        "bonsai_results_sha256": sha_file(args.bonsai_results),
        "historical_pipeline_sha256": sha_file(args.historical_pipeline),
        "script_sha256": sha_file(__file__), "protocol_sha256": sha_file(args.protocol),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "answer_validation": False, "input_truncation": False,
        "retrieval": False, "reference_answers_sent": False, "api_calls": 0,
        "greedy": True, "max_new_tokens": 1024, "repetition_penalty": 1.0,
        "arms": ["4b_shared_exx_prompt", "4b_native_quote_prompt"],
        "native_prompt_note": "Historical minimal quote builder; only question, evidence and round supplied; whitespace-normalized full bodies, original document IDs.",
        "known_difference": "4B native quote protocol differs from E-ID protocol; same-prompt arm separately measures protocol transfer.",
        "production_modified": False, "other_jobs_stopped": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (args.out / "bonsai_reference.jsonl").write_bytes(args.bonsai_results.read_bytes())
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    engine = LLM(
        model=str(base), tokenizer=str(adapter), trust_remote_code=True,
        enable_lora=True, max_lora_rank=16, tensor_parallel_size=1,
        gpu_memory_utilization=0.70, max_model_len=12000, dtype="bfloat16",
        enforce_eager=True, disable_log_stats=True, seed=20261014,
        max_num_seqs=1, max_num_batched_tokens=8192,
    )
    tokenizer = engine.get_tokenizer()
    request = LoRARequest("historical_sft_kto", 1, str(adapter))
    eos = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    assert all(isinstance(i, int) and i >= 0 for i in eos)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, repetition_penalty=1.0,
                              max_tokens=1024, stop_token_ids=eos,
                              skip_special_tokens=False)
    count = 0
    with (args.out / "results.jsonl").open("x") as results, (args.out / "requests.jsonl").open("x") as requests:
        for arm in manifest["arms"]:
            for row in rows:
                prompt = row["prompt"] if arm == "4b_shared_exx_prompt" else native_prompts[row["id"]]
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                assert len(ids) + 1024 <= 12000, "refuse truncation"
                receipt = {
                    "arm": arm, "id": row["id"], "question": row["question"],
                    "prompt": prompt, "prompt_sha256": sha_text(prompt),
                    "input_tokens": len(ids),
                    "input_ids_sha256": sha_text(json.dumps(ids, separators=(",", ":"))),
                    "documents": row["documents"], "round": row["round"],
                    "evidence_count": 20, "input_truncated": False,
                    "max_new_tokens": 1024, "answer_validation": False,
                }
                requests.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                requests.flush()
                begin = time.monotonic()
                outputs = engine.generate([{"prompt_token_ids": ids}], sampling,
                                          lora_request=request, use_tqdm=False)
                assert len(outputs) == 1 and list(outputs[0].prompt_token_ids) == ids
                choice = outputs[0].outputs[0]
                raw = choice.text
                # Record a reversible protocol prefill separately; never repair content.
                prefix = "{" if arm == "4b_native_quote_prompt" else ""
                content = prefix + raw
                for special in ("<|im_end|>", "<|endoftext|>"):
                    content = content.removesuffix(special)
                try:
                    parsed = json.loads(content)
                except ValueError:
                    parsed = None
                result = {k: v for k, v in receipt.items() if k not in ("prompt", "documents")}
                result.update(raw_output=raw, protocol_prefill=prefix,
                              parsed_output_for_offline_inspection_only=parsed,
                              output_token_ids=list(choice.token_ids),
                              finish_reason=choice.finish_reason, stop_reason=choice.stop_reason,
                              output_reached_length_limit=choice.finish_reason == "length",
                              elapsed_seconds=time.monotonic() - begin)
                results.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.flush()
                count += 1
                print(json.dumps({"arm": arm, "id": row["id"], "completed": count,
                                  "finish_reason": choice.finish_reason,
                                  "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    completion = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "generations": count, "expected": 16,
        "results_sha256": sha_file(args.out / "results.jsonl"),
        "elapsed_seconds": time.monotonic() - started,
        "semantic_adjudication_complete": False,
    }
    (args.out / "completion.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
