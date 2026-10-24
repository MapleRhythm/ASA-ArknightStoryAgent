"""Observe the existing CLI's real full pipeline without changing its decisions."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time


def sha(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    return digest.hexdigest()


def token_sha(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    questions = [x.strip() for x in args.questions.read_text().splitlines() if x.strip()]
    assert len(questions) == 20 and len(set(questions)) == 20
    cfg_path = repo / "configs/runtime_inference_gpu.json"
    cfg = json.loads(cfg_path.read_text())
    inf, gen = cfg["inference"], cfg["generator"]
    assert inf["prompt_evidence_top_k"] == 20
    assert inf["prompt_evidence_require_full_documents"] is True
    assert not inf["enable_crag_refinement"]
    assert gen["vllm"]["max_model_len"] == 10000
    adapter = (repo / gen["vllm"]["lora_path"]).resolve()
    assert sha(adapter / "adapter_model.safetensors") == (
        "20c369a0042478f50b4128c047eb03844bd0e74c8eccdf6139e3344a396ef0ce"
    )
    manifest = {
        "mode": "real_cli_end_to_end",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "question_count": 20,
        "questions_sha256": sha(args.questions),
        "protocol_sha256": sha(args.protocol),
        "runtime_config": cfg,
        "runtime_config_sha256": sha(cfg_path),
        "pipeline_sha256": sha(repo / "src/goldenglow/inference/cpu_pipeline.py"),
        "cli_sha256": sha(repo / "scripts/run_cpu_inference.py"),
        "wrapper_sha256": sha(Path(__file__)),
        "base_model_path": str((repo / gen["vllm"]["base_model_path"]).resolve()),
        "lora_path": str(adapter),
        "adapter_weights_sha256": sha(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha(adapter / "adapter_config.json"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "enforce_eager": True,
        "reference_answers_passed_to_model": False,
        "frozen_evidence_passed_to_model": False,
        "external_llm_api_used": False,
        "index_files": {},
    }
    for name in ("index_meta.json", "documents.jsonl", "faiss.index",
                 "bm25_tokens.pkl", "sparse_index.pkl"):
        path = repo / "indexes/arknights_story" / name
        if path.exists():
            info = {"path": str(path), "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns}
            if name == "index_meta.json":
                info.update(sha256=sha(path), content=json.loads(path.read_text()))
            manifest["index_files"][name] = info
    with (out / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    state = {"question_index": 0, "question": None, "stage": "startup",
             "call_index": 0, "record": None}
    with (out / "calls.jsonl").open("x") as calls, \
         (out / "pipeline_audit.jsonl").open("x") as audit:
        def emit(handle, record):
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()

        def context():
            return {k: state[k] for k in ("question_index", "question", "stage")}

        from goldenglow.inference import cpu_pipeline as module
        orig_engine = module.VllmRunner._ensure_engine
        orig_generate = module.VllmRunner.generate
        orig_run = module.CPUInferencePipeline.run
        orig_retrieve = module.CPUInferencePipeline._retrieve_round
        orig_snapshot = module.CPUInferencePipeline.record_full_prompt

        def ensure_engine(self):
            engine, lora = orig_engine(self)
            if not getattr(engine, "_asa_e2e_observation_wrapped", False):
                original = engine.generate

                def observed_engine_generate(prompts, sampling_params, **kwargs):
                    record = state["record"]
                    ids = list(prompts[0]["prompt_token_ids"])
                    encoded = engine.get_tokenizer().encode(
                        record["prompt"], add_special_tokens=True
                    )
                    record.update(
                        actual_input_tokens=len(ids),
                        input_token_ids_sha256=token_sha(ids),
                        full_prompt_token_match=(ids == list(encoded)),
                        max_output_tokens=sampling_params.max_tokens,
                        temperature=sampling_params.temperature,
                    )
                    emit(calls, {"event": "engine_request", **record})
                    outputs = original(prompts, sampling_params, **kwargs)
                    record["engine_outputs"] = [
                        {"raw_text": choice.text,
                         "finish_reason": choice.finish_reason,
                         "stop_reason": choice.stop_reason,
                         "output_tokens": len(choice.token_ids)}
                        for choice in outputs[0].outputs
                    ] if outputs else []
                    record["engine_input_matches"] = bool(
                        outputs and list(outputs[0].prompt_token_ids) == ids
                    )
                    return outputs

                engine.generate = observed_engine_generate
                engine._asa_e2e_observation_wrapped = True
            return engine, lora

        def generate(self, prompt, **kwargs):
            state["call_index"] += 1
            record = {
                **context(), "call_index": state["call_index"],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "generation_overrides": kwargs,
                "max_model_len": self.max_model_len,
            }
            state["record"] = record
            emit(calls, {"event": "generation_start", **record})
            started = time.monotonic()
            try:
                answer = orig_generate(self, prompt, **kwargs)
                record.update(status="ok", returned_text=answer)
                return answer
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                record["elapsed_sec"] = round(time.monotonic() - started, 3)
                emit(calls, {"event": "generation_end", **record})
                state["record"] = None

        def snapshot(self, stage, prompt, evidence):
            result = orig_snapshot(self, stage, prompt, evidence)
            normalized = re.sub(r"\s+", " ", prompt).strip()
            checks = []
            for item in evidence:
                doc = item["document"]
                body = re.sub(r"\s+", " ", module.strip_internal_evidence_meta(
                    doc["clean_text"]
                )).strip()
                checks.append({"id": doc["id"], "full_body_present": body in normalized})
            emit(audit, {
                "event": "prompt_evidence", **context(), "snapshot_stage": stage,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "evidence_count": len(evidence), "body_checks": checks,
                "documents": [x["document"] for x in evidence],
            })
            return result

        def retrieve(self, *r_args, **r_kwargs):
            started = time.monotonic()
            result = orig_retrieve(self, *r_args, **r_kwargs)
            emit(audit, {
                "event": "retrieval", **context(),
                "elapsed_sec": round(time.monotonic() - started, 3),
                "queries": r_args[2] if len(r_args) > 2 else None,
                "evidence": result[2],
            })
            return result

        def run(self, question, dialogue_context="", progress_callback=None):
            state["question_index"] += 1
            state.update(question=question, stage="initial", call_index=0)
            assert question == questions[state["question_index"] - 1]
            assert not dialogue_context
            emit(audit, {"event": "question_start", **context()})

            def progress(stage):
                state["stage"] = stage
                if progress_callback:
                    progress_callback(stage)

            started = time.monotonic()
            try:
                result = orig_run(self, question, dialogue_context, progress)
                emit(audit, {"event": "question_end", **context(), "status": "ok",
                             "answer": result.answer})
                return result
            except Exception as exc:
                emit(audit, {
                    "event": "question_end", **context(), "status": "error",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "elapsed_sec": round(time.monotonic() - started, 3),
                    "model_runtime": self.generator.describe_runtime(),
                    "full_prompt_snapshots": self.prompt_snapshots,
                })
                raise

        module.VllmRunner._ensure_engine = ensure_engine
        module.VllmRunner.generate = generate
        module.CPUInferencePipeline.run = run
        module.CPUInferencePipeline._retrieve_round = retrieve
        module.CPUInferencePipeline.record_full_prompt = snapshot
        sys.argv = [
            str(repo / "scripts/run_cpu_inference.py"),
            "--runtime-config", str(cfg_path),
            "--questions-file", str(args.questions.resolve()),
            "--batch-output", str(out / "results.jsonl"),
            "--device", "cuda", "--backend", "vllm", "--enforce-eager",
        ]
        manifest["cli_argv"] = sys.argv
        emit(audit, {"event": "cli_start", "argv": sys.argv})
        started = time.monotonic()
        runpy.run_path(sys.argv[0], run_name="__main__")
        rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
        completion = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "total_elapsed_sec": round(time.monotonic() - started, 3),
            "expected_count": 20, "actual_count": len(rows),
            "questions_match": [x["question"] for x in rows] == questions,
            "runtime_error_count": sum(bool(x.get("error")) for x in rows),
            "results_sha256": sha(out / "results.jsonl"),
        }
        with (out / "completion.json").open("x") as handle:
            json.dump(completion, handle, ensure_ascii=False, indent=2)
        print(json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
