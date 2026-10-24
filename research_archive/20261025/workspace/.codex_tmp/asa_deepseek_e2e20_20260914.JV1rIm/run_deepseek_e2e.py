"""Same real CLI pipeline; replace only model generation with audited DeepSeek HTTP."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import ssl
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
CHATML = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL
)
EXPECTED = {
    "questions": "8217bab42c62d6130582d970a46400434eefeab28bdb9eed3771fb89e2fcb4e7",
    "rubric": "2e5d297272385c258311ec35311c1f3890c297d424cf6c9d8636cabf2673ae7f",
    "pipeline": "29dfbba72879cec2fa58dabb9c7f6ecc7a36359a457f2482e008e399db790fdc",
    "runtime": "49dfe81d5a2349f213b556abdee218f24a644ae3ab92bb57a8a460740e854527",
    "cli": "c6193c19cee8f7dbcce73162c9c0042b3f117e3600ea81932c678420cd8105d5",
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True
    ).encode()).hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def messages_from_prompt(prompt):
    parsed = CHATML.findall(prompt)
    messages = [{"role": role, "content": text} for role, text in parsed
                if role != "assistant"]
    if [x["role"] for x in messages] != ["system", "user"]:
        raise ValueError("Unexpected ChatML structure; refusing lossy conversion")
    prefills = [text for role, text in parsed if role == "assistant"]
    if len(prefills) != 1 or not prefills[0].lstrip().startswith("{"):
        raise ValueError("Unexpected assistant prefill")
    return messages, prefills[0]


class ApiGenerationError(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output_dir.resolve()
    cfg_path = repo / "configs/runtime_inference_gpu.json"
    cfg = json.loads(cfg_path.read_text())
    cli = repo / "scripts/run_cpu_inference.py"
    paths = {
        "questions": args.questions, "rubric": args.rubric,
        "pipeline": repo / "src/goldenglow/inference/cpu_pipeline.py",
        "runtime": cfg_path, "cli": cli,
    }
    for name, path in paths.items():
        assert sha(path) == EXPECTED[name], f"Unexpected {name} change"
    questions = [x.strip() for x in args.questions.read_text().splitlines() if x.strip()]
    assert len(questions) == len(set(questions)) == 20
    assert not out.exists(), "Refusing to overwrite run directory"
    secret = getpass.getpass("DeepSeek API key (hidden, not logged): ").strip()
    assert secret and not any(x.isspace() for x in secret), "Invalid credential input"
    out.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline_manifest.read_text())
    index_files = {}
    for name, old in baseline["index_files"].items():
        path = Path(old["path"])
        current = {"path": str(path), "bytes": path.stat().st_size,
                   "mtime_ns": path.stat().st_mtime_ns}
        assert current["bytes"] == old["bytes"] and current["mtime_ns"] == old["mtime_ns"], (
            f"Retrieval index changed since 4B test: {name}"
        )
        if name == "index_meta.json":
            current.update(sha256=sha(path), content=json.loads(path.read_text()))
            assert current["sha256"] == old["sha256"]
        index_files[name] = current
    manifest = {
        "mode": "real_cli_end_to_end_deepseek_generator_swap",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "question_count": 20,
        "questions_sha256": sha(args.questions), "rubric_sha256": sha(args.rubric),
        "protocol_sha256": sha(args.protocol), "wrapper_sha256": sha(Path(__file__)),
        "baseline_manifest_sha256": sha(args.baseline_manifest),
        "runtime_config": cfg, "runtime_config_sha256": sha(cfg_path),
        "pipeline_sha256": sha(paths["pipeline"]), "cli_sha256": sha(cli),
        "requested_model": MODEL, "endpoint": ENDPOINT, "thinking": "disabled",
        "max_requests": 120, "automatic_http_retries": 0, "timeout_seconds": 120,
        "chatml_conversion": "system/user unchanged, remove assistant JSON prefill only",
        "system_appendix_added": False, "response_format_override": False,
        "reference_answers_passed_to_model": False,
        "frozen_evidence_passed_to_model": False, "index_files": index_files,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "generator_weights_loaded_locally": False,
        "qwen_token_guard": "same tokenizer and input-plus-output <=10000; not provider tokens",
        "sampling_difference": "DeepSeek non-thinking top_p fixed1, vLLM repeat penalty not mapped",
        "provider_docs_checked": "https://api-docs.deepseek.com/guides/thinking_mode/",
    }
    with (out / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    state = {
        "question_index": 0, "question": None, "stage": "startup",
        "call_index": 0, "request_count": 0, "fatal_api_status": None,
        "evidence_by_prompt": {},
    }
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    with (out / "calls.jsonl").open("x") as calls, \
         (out / "pipeline_audit.jsonl").open("x") as audit:
        def emit(handle, value):
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            handle.write(serialized.replace(secret, "[REDACTED]") + "\n")
            handle.flush()

        def context():
            return {k: state[k] for k in ("question_index", "question", "stage")}

        from goldenglow.inference import cpu_pipeline as module
        from transformers import AutoTokenizer
        base_runner = module.VllmRunner
        orig_run = module.CPUInferencePipeline.run
        orig_retrieve = module.CPUInferencePipeline._retrieve_round
        orig_snapshot = module.CPUInferencePipeline.record_full_prompt

        class DeepSeekRunner(base_runner):
            backend_name = "deepseek_api"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                tok_path = self.lora_path if self.lora_path and (
                    self.lora_path / "tokenizer_config.json"
                ).exists() else self.base_model_path
                self.guard_tokenizer = AutoTokenizer.from_pretrained(
                    str(tok_path), trust_remote_code=True, local_files_only=True
                )
                self.guard_tokenizer_path = str(tok_path)

            def _ensure_engine(self):
                raise RuntimeError("4B model loading is forbidden in API comparison")

            def describe_runtime(self):
                return {
                    "generator_backend": self.backend_name, "api_model": MODEL,
                    "api_endpoint": ENDPOINT, "thinking": "disabled",
                    "base_model_path": None, "lora_path": None,
                    "local_weights_loaded": False, "max_model_len": self.max_model_len,
                    "qwen_guard_tokenizer_path": self.guard_tokenizer_path,
                    "qwen_guard_not_provider_tokenizer": True,
                    "http_automatic_retries": 0, "http_timeout_seconds": 120,
                    "top_p_effective": 1.0, "repeat_penalty_mapped": False,
                }

            def generate(self, prompt, **kwargs):
                state["call_index"] += 1
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                record = {
                    **context(), "call_index": state["call_index"],
                    "prompt": prompt, "prompt_sha256": prompt_hash,
                    "generation_overrides": kwargs, "max_model_len": self.max_model_len,
                }
                started = time.monotonic()
                emit(calls, {"event": "generation_start", **record})
                try:
                    output_budget = kwargs.get("max_tokens")
                    if output_budget is None:
                        output_budget = self.max_tokens
                    ids = self.guard_tokenizer.encode(prompt, add_special_tokens=True)
                    record.update(qwen_guard_input_tokens=len(ids), max_output_tokens=output_budget)
                    if len(ids) + output_budget > self.max_model_len:
                        raise module.PromptContextOverflowError(
                            f"Complete prompt exceeds common Qwen guard: input_tokens={len(ids)}, "
                            f"output_budget={output_budget}, max_model_len={self.max_model_len}; "
                            "no evidence truncated."
                        )
                    messages, prefill = messages_from_prompt(prompt)
                    temperature = kwargs.get("temperature")
                    if temperature is None:
                        temperature = self.temperature
                    body = {
                        "model": MODEL, "messages": messages,
                        "temperature": temperature, "max_tokens": output_budget,
                        "stream": False, "thinking": {"type": "disabled"},
                    }
                    merged = normalized("\n".join(x["content"] for x in messages))
                    body_checks = []
                    for doc in state["evidence_by_prompt"].get(prompt_hash, []):
                        text = normalized(module.strip_internal_evidence_meta(doc["clean_text"]))
                        body_checks.append({"id": doc["id"], "full_body_present": text in merged})
                    assert all(x["full_body_present"] for x in body_checks)
                    record.update(
                        request_body=body, request_sha256=digest(body),
                        removed_assistant_prefill=prefill,
                        outgoing_evidence_body_checks=body_checks,
                    )
                    if state["fatal_api_status"] is not None:
                        raise ApiGenerationError(
                            f"Network requests disabled after fatal HTTP {state['fatal_api_status']}"
                        )
                    if state["request_count"] >= 120:
                        raise ApiGenerationError("Bounded API request budget exhausted")
                    state["request_count"] += 1
                    record["request_number"] = state["request_count"]
                    emit(calls, {"event": "http_request", **record})
                    request = urllib.request.Request(
                        ENDPOINT, data=json.dumps(body, ensure_ascii=False).encode(),
                        headers={"Content-Type": "application/json",
                                 "Authorization": "Bearer " + secret},
                        method="POST",
                    )
                    network_started = time.monotonic()
                    try:
                        with opener.open(request, timeout=120) as response:
                            record["http_status"] = response.status
                            payload = json.loads(response.read(2 * 1024 * 1024 + 1))
                    except urllib.error.HTTPError as exc:
                        record["http_status"] = exc.code
                        if exc.code in (401, 402, 403, 429):
                            state["fatal_api_status"] = exc.code
                        raise ApiGenerationError(f"DeepSeek HTTP {exc.code}") from None
                    finally:
                        record["http_elapsed_sec"] = round(time.monotonic() - network_started, 3)
                    choice = payload["choices"][0]
                    content = str(choice["message"].get("content") or "").replace(secret, "[REDACTED]")
                    record.update(
                        returned_model=payload.get("model"), response_id=payload.get("id"),
                        created=payload.get("created"), usage=payload.get("usage"),
                        finish_reason=choice.get("finish_reason"), raw_content=content,
                        reasoning_content_nonempty=bool(choice["message"].get("reasoning_content")),
                    )
                    if choice.get("finish_reason") != "stop":
                        raise ApiGenerationError(f"Non-stop finish: {choice.get('finish_reason')}")
                    if not content.strip():
                        raise ApiGenerationError("Empty API answer content")
                    answer = module.sanitize_generation_output(content, prompt)
                    record.update(status="ok", returned_text=answer)
                    return answer
                except Exception as exc:
                    record.update(status="error", error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    record["elapsed_sec"] = round(time.monotonic() - started, 3)
                    emit(calls, {"event": "generation_end", **record})

        def snapshot(self, stage, prompt, evidence):
            result = orig_snapshot(self, stage, prompt, evidence)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            docs = [x["document"] for x in evidence]
            state["evidence_by_prompt"][prompt_hash] = docs
            prompt_norm = normalized(prompt)
            checks = [
                {"id": doc["id"], "full_body_present": normalized(
                    module.strip_internal_evidence_meta(doc["clean_text"])
                ) in prompt_norm} for doc in docs
            ]
            emit(audit, {
                "event": "prompt_evidence", **context(), "snapshot_stage": stage,
                "prompt_sha256": prompt_hash, "evidence_count": len(docs),
                "body_checks": checks, "documents": docs,
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
            state.update(question=question, stage="initial", call_index=0, evidence_by_prompt={})
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

        module.VllmRunner = DeepSeekRunner
        module.CPUInferencePipeline.run = run
        module.CPUInferencePipeline._retrieve_round = retrieve
        module.CPUInferencePipeline.record_full_prompt = snapshot
        sys.argv = [
            str(cli), "--runtime-config", str(cfg_path),
            "--questions-file", str(args.questions.resolve()),
            "--batch-output", str(out / "results.jsonl"),
            "--device", "cuda", "--backend", "vllm", "--enforce-eager",
        ]
        emit(audit, {"event": "cli_start", "argv": sys.argv,
                     "runner_replaced_in_memory": True})
        print("DeepSeek full E2E 20 begins; no local 4B weights and no HTTP retries", flush=True)
        started = time.monotonic()
        runpy.run_path(str(cli), run_name="__main__")
        rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
        completion = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "total_elapsed_sec": round(time.monotonic() - started, 3),
            "expected_count": 20, "actual_count": len(rows),
            "questions_match": [x["question"] for x in rows] == questions,
            "runtime_error_count": sum(bool(x.get("error")) for x in rows),
            "http_request_count": state["request_count"],
            "fatal_api_status": state["fatal_api_status"],
            "results_sha256": sha(out / "results.jsonl"),
        }
        with (out / "completion.json").open("x") as handle:
            json.dump(completion, handle, ensure_ascii=False, indent=2)
        print(json.dumps(completion, ensure_ascii=False), flush=True)
    secret = None


if __name__ == "__main__":
    main()
