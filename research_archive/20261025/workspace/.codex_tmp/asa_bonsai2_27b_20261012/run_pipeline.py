"""Process-local Bonsai generator adapter; production code is not edited."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
import time

from bonsai_api import BonsaiClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:18127")
    p.add_argument("--retrieval-device", default="cuda:0")
    args = p.parse_args()
    repo = args.repo.resolve()
    questions = args.questions.read_text().splitlines()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo / "src"))
    from run_noguard50 import install_no_answer_validation
    from asa_arknight_story_agent.inference.cpu_pipeline import CPUInferencePipeline
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta
    from asa_arknight_story_agent.inference.generation import (
        answer_conclusion_generation as ac, direct_answer_generation as da,
        hypothesis_generation as hg,
    )
    client = BonsaiClient(args.url)
    state = {"q": 0, "question": None, "call": 0}
    config = json.loads((repo / "configs/runtime_exx_official.json").read_text())
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "model": str(args.model),
        "kind": "bonsai2_live_pipeline_noguard", "repo": str(repo),
        "questions": questions, "runtime_config": config,
        "release_lock": json.loads((repo / "RELEASE_LOCK.json").read_text()),
        "answer_validation_enabled": False, "old_answers_or_gold_sent": False,
        "thinking": False, "p_tuning": False, "production_modified": False,
        "server_properties": client.request("/props"),
        "comparison_caveat": "Native GGUF backend/template, no adapter; retrieval unchanged but actual evidence may differ.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (args.out / "questions.txt").write_text(args.questions.read_text())
    with (args.out / "calls.jsonl").open("x") as calls, (args.out / "audit.jsonl").open("x") as audit:
        def emit(handle, data):
            handle.write(json.dumps({**state, **data}, ensure_ascii=False, default=str) + "\n")
            handle.flush()

        install_no_answer_validation(lambda data: emit(audit, data))
        original_normalizer = hg.normalize_hypothesis_payload

        def normalize(payload, **kw):
            # Same structural compatibility as the valid 9B comparison.
            if isinstance(payload, dict) and len(payload) == 1:
                for wrapper in ("hypothesis", "hypothesis_v2", "follow_up_hypothesis_v2"):
                    if isinstance(payload.get(wrapper), dict):
                        emit(audit, {"event": "hypothesis_schema_unwrapped", "wrapper": wrapper})
                        payload = payload[wrapper]
                        break
            try:
                result = original_normalizer(payload, **kw)
            except Exception as exc:
                emit(audit, {"event": "hypothesis_schema", "valid": False, "error": str(exc)})
                raise
            emit(audit, {"event": "hypothesis_schema", "valid": True})
            return result
        hg.normalize_hypothesis_payload = normalize

        def schema_hint(original):
            def wrapped(*a, **kw):
                prompt = original(*a, **kw)
                return prompt.replace(
                    "只输出 JSON。",
                    "只输出 JSON。\n输出格式补充：output_schema中的字段必须直接位于JSON顶层；"
                    "不要外包为hypothesis字段，不要添加version或其他字段。", 1,
                )
            return wrapped
        hg.build_hypothesis_prompt = schema_hint(hg.build_hypothesis_prompt)
        hg.build_follow_up_hypothesis_prompt = schema_hint(hg.build_follow_up_hypothesis_prompt)

        def observe_builder(module, name):
            original = getattr(module, name)
            def wrapped(*a, **kw):
                result = original(*a, **kw)
                prompt = result[0] if isinstance(result, tuple) else result
                docs = [x["document"] for x in kw.get("prompt_evidence", [])]
                normalized = re.sub(r"\s+", " ", prompt)
                checks = [{
                    "id": d["id"], "full": re.sub(r"\s+", " ",
                    strip_internal_evidence_meta(d["clean_text"])).strip() in normalized,
                } for d in docs]
                emit(audit, {"event": "prompt_evidence", "builder": name,
                             "documents": docs, "checks": checks})
                if docs and (len(docs) != 20 or not all(c["full"] for c in checks)):
                    raise ValueError("Evidence input incomplete; no generation allowed")
                return result
            setattr(module, name, wrapped)
        for module, name in ((ac, "build_conclusion_prompt"), (da, "build_answer_prompt"),
                             (hg, "build_follow_up_hypothesis_prompt")):
            observe_builder(module, name)

        def generate(self, prompt, **kw):
            state["call"] += 1
            assert self.lora_path is None
            begin = time.monotonic()
            emit(calls, {"event": "request", "original_prompt": prompt, "options": kw})
            try:
                response = client.generate(prompt, thinking=False, **kw)
                emit(calls, {"event": "response", **response,
                             "elapsed_sec": time.monotonic() - begin})
                self.last_generation_diagnostics = {
                    "backend": "prism_llama_server", "finish_reason": response["finish_reason"],
                    "input_tokens": response["input_tokens"], "native_template": True,
                }
                return response["text"]
            except Exception as exc:
                emit(calls, {"event": "error", "error": f"{type(exc).__name__}: {exc}"})
                raise

        def describe(self):
            return {
                "generator_backend": "prism_llama_server", "gguf_model_path": str(args.model),
                "base_model_path": str(args.model), "lora_path": None,
                "runtime_mode": "ternary_base_gguf_native_chat",
                "max_model_len": client.context, "p_tuning_applied": False,
            }
        VllmRunner.generate = generate
        VllmRunner.describe_runtime = describe
        original_run = CPUInferencePipeline.run
        original_retrieve = CPUInferencePipeline._retrieve_round

        def run(self, question, dialogue_context="", progress_callback=None):
            state["q"] += 1
            state.update(question=question, call=0)
            assert question == questions[state["q"] - 1]
            assert self.prompt_evidence_top_k == 20 and self.prompt_evidence_require_full_documents
            assert not dialogue_context
            return original_run(self, question, dialogue_context, progress_callback)

        def retrieve(self, *a, **kw):
            result = original_retrieve(self, *a, **kw)
            emit(audit, {"event": "retrieval", "counts": [len(x) for x in result],
                         "reranked_ids": [x["document"]["id"] for x in result[2]]})
            return result
        CPUInferencePipeline.run = run
        CPUInferencePipeline._retrieve_round = retrieve
        sys.argv = [
            str(repo / "scripts/run_cpu_inference.py"), "--runtime-config",
            str(repo / "configs/runtime_exx_official.json"), "--base-model", str(args.model),
            "--disable-lora", "--device", args.retrieval_device,
            "--ctx-size", "32768", "--max-tokens", "1536",
            "--questions-file", str(args.out / "questions.txt"),
            "--batch-output", str(args.out / "results.jsonl"),
        ]
        began = time.monotonic()
        code, error = 0, None
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            code = exc.code or 0
        except Exception as exc:
            code, error = 1, f"{type(exc).__name__}: {exc}"
        results = ([json.loads(x) for x in (args.out / "results.jsonl").read_text().splitlines()]
                   if (args.out / "results.jsonl").exists() else [])
        completion = {
            "exit_code": code, "error": error, "count": len(results),
            "expected": len(questions), "elapsed_sec": time.monotonic() - began,
            "runtime_errors": sum(bool(r.get("error")) for r in results),
        }
        (args.out / "completion.json").write_text(json.dumps(completion, indent=2))
        print(json.dumps(completion), flush=True)
        raise SystemExit(code or int(len(results) != len(questions) or completion["runtime_errors"] > 0))


if __name__ == "__main__":
    main()
