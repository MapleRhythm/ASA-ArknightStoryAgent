"""Six-case full-evidence reader replay; no retrieval, training, or API calls."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalized(value):
    return re.sub(r"\s+", " ", value).strip()


def write_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from goldenglow.config import QueryConfig
    from goldenglow.inference.cpu_pipeline import (
        CPUInferencePipeline, HypothesisDocument, VllmRunner, strip_internal_evidence_meta,
    )

    rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    assert len(rows) == 6
    cfg_path = args.repo / "configs/runtime_inference_gpu.json"
    cfg = json.loads(cfg_path.read_text())
    inf, gen = cfg["inference"], cfg["generator"]
    assert inf["prompt_evidence_top_k"] == 20
    assert inf["prompt_evidence_require_full_documents"]
    assert not inf["enable_crag_refinement"]
    base = (args.repo / gen["vllm"]["base_model_path"]).resolve()
    adapter = (args.repo / gen["vllm"]["lora_path"]).resolve()
    for row in rows:
        assert base == Path(row["model_runtime"]["base_model_path"])
        assert adapter == Path(row["model_runtime"]["lora_path"])
    args.output_dir.mkdir(exist_ok=True, parents=True)
    manifest = {
        "mode": "frozen_final_evidence_reader_replay",
        "questions": 6,
        "top_k": 20,
        "retrieval_rerun": False,
        "round_for_generation": 2,
        "round_note": "Final-round conclusion and existing direct-answer fallback; not an end-to-end retrieval rerun.",
        "source_sha256": sha(args.source),
        "runtime_config_sha256": sha(cfg_path),
        "pipeline_sha256": sha(args.repo / "src/goldenglow/inference/cpu_pipeline.py"),
        "script_sha256": sha(Path(__file__)),
        "base_model_path": str(base),
        "lora_path": str(adapter),
        "adapter_config_sha256": sha(adapter / "adapter_config.json"),
        "adapter_weights_sha256": sha(adapter / "adapter_model.safetensors"),
        "physical_gpu": 0,
        "gpu_memory_utilization": 0.62,
        "enforce_eager": True,
        "prompt_mode": inf["conclusion_prompt_mode"],
        "grounding_mode": inf["answer_grounding_mode"],
    }
    with (args.output_dir / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    with (args.output_dir / "calls.jsonl").open("x") as calls, (args.output_dir / "results.jsonl").open("x") as results:
        class AuditedRunner(VllmRunner):
            expected_evidence = []
            question_index = 0
            current_record = None
            records = []
            wrapped = False

            def _ensure_engine(self):
                engine, request = super()._ensure_engine()
                if not self.wrapped:
                    original_generate = engine.generate

                    def audited_generate(prompts, sampling_params, **kwargs):
                        assert len(prompts) == 1
                        token_ids = list(prompts[0]["prompt_token_ids"])
                        record = self.current_record
                        record["actual_input_token_ids"] = token_ids
                        record["actual_input_tokens"] = len(token_ids)
                        record["input_token_ids_sha256"] = digest(token_ids)
                        record["max_output_tokens"] = sampling_params.max_tokens
                        record["temperature"] = sampling_params.temperature
                        write_record(calls, {"event": "request", **record})
                        outputs = original_generate(prompts, sampling_params, **kwargs)
                        record["engine_outputs"] = [
                            {
                                "raw_text": choice.text,
                                "finish_reason": choice.finish_reason,
                                "stop_reason": choice.stop_reason,
                                "output_tokens": len(choice.token_ids),
                            }
                            for choice in outputs[0].outputs
                        ]
                        observed = outputs[0].prompt_token_ids
                        record["engine_input_matches"] = list(observed) == token_ids
                        assert record["engine_input_matches"], "Engine changed the checked input tokens"
                        return outputs

                    engine.generate = audited_generate
                    self.wrapped = True
                return engine, request

            def generate(self, prompt, **kwargs):
                record = {
                    "question_index": self.question_index,
                    "call_index": len(self.records) + 1,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "complete_evidence_ids": [],
                }
                for item in self.expected_evidence:
                    body = normalized(strip_internal_evidence_meta(item["clean_text"]))
                    assert body in normalized(prompt), f"Incomplete body: {item['id']}"
                    record["complete_evidence_ids"].append(item["id"])
                assert len(record["complete_evidence_ids"]) == 20
                self.current_record = record
                started = time.monotonic()
                try:
                    generated = super().generate(prompt, **kwargs)
                    record.update(status="ok", returned_text=generated)
                    return generated
                except Exception as exc:
                    record.update(status="error", error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    record["seconds"] = round(time.monotonic() - started, 3)
                    self.records.append(record)
                    write_record(calls, {"event": "response", **record})

        runner = AuditedRunner(
            base_model_path=base, lora_path=adapter,
            tensor_parallel_size=1, gpu_memory_utilization=0.62,
            max_model_len=gen["vllm"]["max_model_len"],
            max_tokens=gen["max_tokens"], temperature=gen["temperature"],
            top_p=gen["top_p"], repeat_penalty=gen["repeat_penalty"],
            dtype=gen["vllm"]["dtype"], enforce_eager=True,
        )
        print("Loading the same 4B base + LoRA on GPU 0", flush=True)
        runner._ensure_engine()
        print("Engine ready; six bounded reader cases begin", flush=True)
        obj = CPUInferencePipeline(
            retriever=None, generator=runner, query_config=QueryConfig(),
            max_retrieval_rounds=2, prompt_evidence_top_k=20,
            prompt_evidence_require_full_documents=True,
            prompt_evidence_max_chars_per_doc=0,
            prompt_conclusion_evidence_max_total_chars=0,
            enable_crag_refinement=False, conclusion_prompt_mode=inf["conclusion_prompt_mode"],
            answer_grounding_mode=inf["answer_grounding_mode"],
            self_consistency_samples=inf["self_consistency_samples"],
        )
        for index, row in enumerate(rows, 1):
            runner.question_index = index
            runner.expected_evidence = row["evidence"][:20]
            runner.records = []
            obj.prompt_snapshots = []
            hyp = HypothesisDocument(**row["retrieval_trace"][-1]["hypothesis"])
            evidence = [{"document": item, "evidence_chain_text": item.get("evidence_chain_text")} for item in row["evidence"]]
            started = time.monotonic()
            result = {"question_index": index, "question": row["question"], "old_4b_answer": row["answer"]}
            print(f"Starting case {index}/6", flush=True)
            try:
                conclusion = obj.generate_conclusion(row["question"], hyp, evidence, [], 2)
                result.update(status="ok", answer=conclusion.answer, conclusion=asdict(conclusion))
            except Exception as exc:
                result.update(status="error", error_type=type(exc).__name__, error=str(exc))
            result.update(
                seconds=round(time.monotonic() - started, 3),
                model_runtime=runner.describe_runtime(),
                full_prompt_snapshots=obj.prompt_snapshots,
                calls=[
                    {k: v for k, v in rec.items() if k not in ("prompt", "actual_input_token_ids")}
                    for rec in runner.records
                ],
            )
            write_record(results, result)
            print(json.dumps({
                k: v for k, v in result.items()
                if k not in ("model_runtime", "full_prompt_snapshots", "calls")
            }, ensure_ascii=False), flush=True)
            if result.get("error_type") in ("OutOfMemoryError", "PromptContextOverflowError"):
                print("Stopping on resource/capacity failure; no truncation or retry", flush=True)
                break


if __name__ == "__main__":
    main()
