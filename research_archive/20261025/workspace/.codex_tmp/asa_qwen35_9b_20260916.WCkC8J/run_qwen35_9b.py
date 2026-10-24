"""Isolated Qwen3.5-9B runtime: official pipeline, full evidence, no answer validation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical_prompt(tokenizer, prompt):
    match = re.fullmatch(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n"
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n"
        r"<\|im_start\|>assistant\n(.*)",
        prompt, flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Unknown prompt envelope; refusing to silently change it")
    system, user, prefix = match.groups()
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert system in rendered and user in rendered
    assert "<think>\n\n</think>" in rendered
    return rendered + prefix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--verification", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--questions", type=Path)
    source.add_argument("--question")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--retrieval-device", default="cuda:1")
    args = p.parse_args()
    repo, model, out = args.repo.resolve(), args.model.resolve(), args.out.resolve()
    verification = json.loads(args.verification.read_text())
    assert verification["all_passed"] and Path(verification["model"]).resolve() == model
    questions = args.questions.read_text().splitlines() if args.questions else [args.question]
    assert questions and all(q.strip() for q in questions)
    out.mkdir(parents=True, exist_ok=False)
    (out / "questions.txt").write_text("\n".join(questions) + "\n")
    sys.path.insert(0, str(repo / "src"))
    from run_noguard50 import install_no_answer_validation
    from asa_arknight_story_agent.inference.cpu_pipeline import CPUInferencePipeline
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta
    from asa_arknight_story_agent.inference.generation import (
        answer_conclusion_generation as ac, direct_answer_generation as da,
        hypothesis_generation as hg,
    )
    state = {"q": 0, "question": None, "call": 0}
    llm_options = {
        "model": str(model), "tokenizer": str(model), "trust_remote_code": False,
        "enable_lora": False, "tensor_parallel_size": 1, "dtype": "bfloat16",
        "language_model_only": True, "gpu_memory_utilization": 0.90,
        "max_model_len": 10000, "max_num_batched_tokens": 4096, "max_num_seqs": 4,
        "enforce_eager": True, "disable_log_stats": True, "seed": 0,
    }
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "repo": str(repo), "model": str(model), "model_verification": verification,
        "questions_sha256": sha(out / "questions.txt"), "count": len(questions),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "llm_options": llm_options, "retrieval_device": args.retrieval_device,
        "answer_validation_enabled": False, "thinking_enabled": False,
        "lora_enabled": False, "quantization": None,
        "old_answers_or_gold_sent": False, "kind": "qwen35_9b_real_pipeline_noguard",
        "baseline_config": json.loads((repo / "configs/runtime_exx_official.json").read_text()),
        "release_lock": json.loads((repo / "RELEASE_LOCK.json").read_text()),
        "versions": {n: importlib.metadata.version(n) for n in ("vllm", "transformers", "torch")},
        "runner_sha256": sha(Path(__file__)),
        "comparison_caveat": "9B without 4B adapter, native non-thinking chat envelope; retrieval configuration unchanged, actual retrieved sets may differ.",
        "production_files_modified": False,
    }
    with (out / "manifest.json").open("x") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    with (out / "calls.jsonl").open("x") as calls, (out / "audit.jsonl").open("x") as audit:
        def emit(handle, data):
            handle.write(json.dumps({**state, **data}, ensure_ascii=False, default=str) + "\n")
            handle.flush()

        install_no_answer_validation(lambda data: emit(audit, data))
        original_hypothesis_normalizer = hg.normalize_hypothesis_payload

        def normalize_hypothesis_payload_9b(payload, **kw):
            # Qwen3.5 occasionally echoes the schema name as a single wrapper.
            # This is a structural compatibility repair only: the inner model
            # fields are passed through unchanged and no fact/content is edited.
            if isinstance(payload, dict) and len(payload) == 1:
                for wrapper in ("hypothesis", "hypothesis_v2", "follow_up_hypothesis_v2"):
                    inner = payload.get(wrapper)
                    if isinstance(inner, dict):
                        payload = inner
                        emit(audit, {"event": "hypothesis_schema_unwrapped",
                                     "wrapper": wrapper})
                        break
            try:
                normalized = original_hypothesis_normalizer(payload, **kw)
            except Exception as exc:
                emit(audit, {"event": "hypothesis_schema", "valid": False,
                             "error": f"{type(exc).__name__}: {exc}",
                             "payload_keys": sorted(payload) if isinstance(payload, dict) else None})
                raise
            emit(audit, {"event": "hypothesis_schema", "valid": True,
                         "payload_keys": sorted(payload) if isinstance(payload, dict) else None})
            return normalized

        hg.normalize_hypothesis_payload = normalize_hypothesis_payload_9b
        original_generate = VllmRunner.generate
        original_run = CPUInferencePipeline.run
        original_retrieve = CPUInferencePipeline._retrieve_round

        def ensure_engine(self):
            if self._llm is not None:
                return self._llm, None
            if self._engine_init_error is not None:
                raise RuntimeError("Previous 9B initialization failed") from self._engine_init_error
            assert self.lora_path is None and self.base_model_path.resolve() == model
            from vllm import LLM
            try:
                self._llm = LLM(**llm_options)
            except Exception as e:
                self._engine_init_error = e
                raise
            engine_generate = self._llm.generate

            def observed_generate(prompts, sampling_params, **kw):
                ids = self._llm.get_tokenizer().encode(prompts[0], add_special_tokens=True)
                result = engine_generate(prompts, sampling_params, **kw)
                emit(calls, {
                    "event": "engine_result", "input_tokens": len(ids),
                    "actual_input_matches": list(result[0].prompt_token_ids) == list(ids),
                    "max_tokens": sampling_params.max_tokens,
                    "choices": [{"text": c.text, "tokens": len(c.token_ids),
                                 "finish_reason": c.finish_reason} for c in result[0].outputs],
                })
                return result
            self._llm.generate = observed_generate
            return self._llm, None

        def generate(self, prompt, **kw):
            state["call"] += 1
            started = time.monotonic()
            try:
                engine, _ = self._ensure_engine()
                native = canonical_prompt(engine.get_tokenizer(), prompt)
                emit(calls, {"event": "request", "prompt": native, "original_prompt": prompt,
                             "options": kw, "native_chat_template": True, "thinking": False})
                response = original_generate(self, native, **kw)
                emit(calls, {"event": "response", "text": response,
                             "diagnostics": self.last_generation_diagnostics,
                             "elapsed_sec": time.monotonic() - started})
                return response
            except Exception as e:
                emit(calls, {"event": "error", "error": f"{type(e).__name__}: {e}"})
                raise

        def wrap_builder(module, name):
            original = getattr(module, name)
            def observed(*a, **kw):
                result = original(*a, **kw)
                prompt = result[0] if isinstance(result, tuple) else result
                docs = [x["document"] for x in kw.get("prompt_evidence", [])]
                normalized = re.sub(r"\s+", " ", prompt)
                checks = [
                    {"id": d["id"], "full": re.sub(r"\s+", " ",
                     strip_internal_evidence_meta(d["clean_text"])).strip() in normalized}
                    for d in docs
                ]
                emit(audit, {"event": "prompt_evidence", "builder": name, "documents": docs,
                             "checks": checks, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
                return result
            setattr(module, name, observed)

        original_hypothesis_prompt = hg.build_hypothesis_prompt

        def build_hypothesis_prompt_9b(question, dialogue_context="", **kw):
            prompt = original_hypothesis_prompt(question, dialogue_context, **kw)
            schema_hint = (
                "\n输出格式补充：output_schema中的字段必须直接位于JSON顶层；"
                "不要外包为hypothesis字段，不要添加version或其他字段。"
            )
            return prompt.replace("只输出 JSON。", "只输出 JSON。" + schema_hint, 1)

        hg.build_hypothesis_prompt = build_hypothesis_prompt_9b
        original_follow_up_prompt = hg.build_follow_up_hypothesis_prompt

        def build_follow_up_hypothesis_prompt_9b(*a, **kw):
            prompt = original_follow_up_prompt(*a, **kw)
            schema_hint = (
                "\n输出格式补充：output_schema中的字段必须直接位于JSON顶层；"
                "不要外包为hypothesis字段，不要添加version或其他字段。"
            )
            return prompt.replace("只输出 JSON。", "只输出 JSON。" + schema_hint, 1)

        hg.build_follow_up_hypothesis_prompt = build_follow_up_hypothesis_prompt_9b

        def run(self, question, dialogue_context="", progress_callback=None):
            state["q"] += 1
            state.update(question=question, call=0)
            assert question == questions[state["q"] - 1] and not dialogue_context
            assert self.prompt_evidence_top_k == 20 and self.prompt_evidence_require_full_documents
            assert self.answer_grounding_mode == "evidence_id"
            assert not getattr(self, "enable_experimental_grounding_repair", False)
            return original_run(self, question, dialogue_context, progress_callback)

        def retrieve(self, *a, **kw):
            result = original_retrieve(self, *a, **kw)
            emit(audit, {"event": "retrieval", "queries": kw.get("queries", a[2] if len(a) > 2 else None),
                         "counts": [len(x) for x in result],
                         "reranked_ids": [h["document"]["id"] for h in result[2]]})
            return result

        VllmRunner._ensure_engine = ensure_engine
        VllmRunner.generate = generate
        CPUInferencePipeline.run = run
        CPUInferencePipeline._retrieve_round = retrieve
        for module, name in ((ac, "build_conclusion_prompt"), (da, "build_answer_prompt"),
                             (hg, "build_follow_up_hypothesis_prompt")):
            wrap_builder(module, name)
        sys.argv = [
            str(repo / "scripts/run_cpu_inference.py"),
            "--runtime-config", str(repo / "configs/runtime_exx_official.json"),
            "--base-model", str(model), "--disable-lora",
            "--device", args.retrieval_device, "--ctx-size", "10000", "--max-tokens", "1536",
            "--gpu-memory-utilization", "0.90", "--tensor-parallel-size", "1",
            "--dtype", "bfloat16", "--enforce-eager", "--max-num-batched-tokens", "4096",
            "--questions-file", str(out / "questions.txt"), "--batch-output", str(out / "results.jsonl"),
        ]
        started = time.monotonic()
        code, error = 0, None
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as e:
            code = e.code or 0
        except Exception as e:
            code, error = 1, f"{type(e).__name__}: {e}"
        rows = ([json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
                if (out / "results.jsonl").exists() else [])
        complete = {
            "completed_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code,
            "error": error, "count": len(rows), "expected_count": len(questions),
            "questions_match": [r["question"] for r in rows] == questions,
            "runtime_errors": sum(bool(r.get("error")) for r in rows),
            "elapsed_sec": time.monotonic() - started,
            "results_sha256": sha(out / "results.jsonl") if rows else None,
        }
        with (out / "completion.json").open("x") as f:
            json.dump(complete, f, ensure_ascii=False, indent=2)
        print(json.dumps(complete, ensure_ascii=False), flush=True)
        raise SystemExit(code or int(not complete["questions_match"] or complete["runtime_errors"] > 0))


if __name__ == "__main__":
    main()
