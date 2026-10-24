"""One-factor ablation: official Exx prompts/retrieval, answer validation bypassed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict


def install_no_answer_validation(emit):
    """No lexical, citation, or semantic decision is made in this path."""
    from asa_arknight_story_agent.inference.generation import (
        answer_conclusion_generation as ac, direct_answer_generation as da,
    )
    from asa_arknight_story_agent.inference.grounding import (
        validation, evidence_id_checks, evidence_id_validation, partial_support,
        quote_checks, token_checks,
    )

    def passthrough(**kw):
        conclusion = kw["conclusion"]
        before = asdict(conclusion)
        conclusion.generation_diagnostics["grounding_status"] = "disabled_for_experiment"
        after = asdict(conclusion)
        before.pop("generation_diagnostics")
        after.pop("generation_diagnostics")
        assert before == after
        emit({"event": "answer_validation_bypassed", "input_equals_output": True,
              "conclusion": after, "requested_mode_for_prompt_only": kw.get("mode")})
        return conclusion

    def no_partial_recheck(**kw):
        return kw["conclusion"]

    def unexpected_validator(*a, **kw):
        emit({"event": "unexpected_answer_validator"})
        raise RuntimeError("An answer validator ran during no-validation experiment")

    for module in (ac, da, validation):
        module.validate_conclusion_grounding = passthrough
    ac.preserve_partial_support = no_partial_recheck
    partial_support.preserve_partial_support = no_partial_recheck
    # Tripwires do not grade answers. Any accidental call invalidates the ablation.
    evidence_id_checks.validate_evidence_id_answer = unexpected_validator
    evidence_id_checks.validate_evidence_id_grounding = unexpected_validator
    evidence_id_validation.validate_evidence_id_grounding = unexpected_validator
    quote_checks.validate_quote_grounding = unexpected_validator
    token_checks.validate_token_grounding = unexpected_validator
    return passthrough


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert sha(args.questions) == "be3c8bfb6cf1b9ac3ab391bb7ff039d673b8aa8ca598c20457bb6ac5d3110184"
    questions = args.questions.read_text().splitlines()
    assert len(questions) == 50
    (out / "questions.txt").write_text(args.questions.read_text())
    sys.path.insert(0, str(repo / "src"))
    from asa_arknight_story_agent.inference.cpu_pipeline import CPUInferencePipeline
    from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner
    from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta
    from asa_arknight_story_agent.inference.generation import (
        answer_conclusion_generation as ac, direct_answer_generation as da,
        hypothesis_generation as hg,
    )
    state = {"question_index": 0, "question": None, "call": 0, "pending_documents": []}
    config = json.loads((repo / "configs/runtime_exx_official.json").read_text())
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                "repo": str(repo), "runtime_config": config, "questions_sha256": sha(args.questions),
                "release_lock": json.loads((repo / "RELEASE_LOCK.json").read_text()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "old_answers_or_gold_sent": False, "index_files": {},
                "kind": "existing_chinese_eval50_no_answer_validation",
                "answer_validation_enabled": False,
                "intervention": "process_local_identity_bypass_of_answer_validation",
                "config_evidence_id_is_prompt_format_only": True,
                "frozen_protocol_sha256": "e8d7453837a96046da3c6aa1186d80092152a0e52448c8ab241b660d3f98f6d7",
                "runner_sha256": sha(Path(__file__))}
    for name in ("index_meta.json", "documents.jsonl", "faiss.index", "bm25_tokens.pkl"):
        file = repo / "indexes/arknights_story" / name
        if file.exists():
            stat = file.stat()
            manifest["index_files"][name] = {"resolved_path": str(file.resolve()), "size": stat.st_size,
                                             "mtime_ns": stat.st_mtime_ns}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    with (out / "calls.jsonl").open("x") as calls, (out / "audit.jsonl").open("x") as audit:
        def emit(handle, data):
            handle.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
            handle.flush()

        def context():
            return {k: state[k] for k in ("question_index", "question", "call")}

        install_no_answer_validation(lambda data: emit(audit, {**context(), **data}))

        def wrap_builder(module, name):
            original = getattr(module, name)
            def observed(*a, **kw):
                result = original(*a, **kw)
                prompt = result[0] if isinstance(result, tuple) else result
                documents = [x["document"] for x in kw.get("prompt_evidence", [])]
                normalized = re.sub(r"\s+", " ", prompt)
                checks = [{"id": d["id"], "full": re.sub(r"\s+", " ",
                           strip_internal_evidence_meta(d["clean_text"])).strip() in normalized}
                          for d in documents]
                expected_count = min(20, len(kw.get("evidence", a[2] if len(a) > 2 else documents)))
                state["pending_documents"] = documents
                emit(audit, {"event": "prompt_evidence", **context(), "builder": name,
                             "documents": [
                                 {k: d.get(k) for k in (
                                     "id", "clean_text", "source_path", "story_key",
                                     "story_name", "activity_name", "segments")}
                                 for d in documents
                             ], "checks": checks,
                             "count_matches_top20": len(documents) == expected_count,
                             "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
                return result
            setattr(module, name, observed)

        wrap_builder(ac, "build_conclusion_prompt")
        wrap_builder(da, "build_answer_prompt")
        wrap_builder(hg, "build_follow_up_hypothesis_prompt")
        orig_generate, orig_engine = VllmRunner.generate, VllmRunner._ensure_engine
        orig_run, orig_retrieve = CPUInferencePipeline.run, CPUInferencePipeline._retrieve_round

        def ensure_engine(self):
            engine, lora = orig_engine(self)
            if not getattr(engine, "_exx_observed", False):
                original = engine.generate
                def engine_generate(prompts, sampling_params, **kw):
                    ids = list(engine.get_tokenizer().encode(prompts[0], add_special_tokens=True))
                    outputs = original(prompts, sampling_params, **kw)
                    emit(calls, {"event": "engine_result", **context(), "input_tokens": len(ids),
                                 "actual_input_matches": list(outputs[0].prompt_token_ids) == ids,
                                 "lora_path": str(self.lora_path),
                                 "max_tokens": sampling_params.max_tokens,
                                 "temperature": sampling_params.temperature, "top_p": sampling_params.top_p,
                                 "choices": [{"text": c.text, "tokens": len(c.token_ids),
                                              "finish_reason": c.finish_reason} for c in outputs[0].outputs]})
                    return outputs
                engine.generate = engine_generate
                engine._exx_observed = True
            return engine, lora

        def generate(self, prompt, **kw):
            state["call"] += 1
            started = time.monotonic()
            emit(calls, {"event": "request", **context(), "prompt": prompt, "options": kw})
            try:
                response = orig_generate(self, prompt, **kw)
                emit(calls, {"event": "response", **context(), "text": response,
                             "diagnostics": self.last_generation_diagnostics,
                             "elapsed_sec": time.monotonic() - started})
                return response
            except Exception as exc:
                emit(calls, {"event": "error", **context(), "error": str(exc)})
                raise

        def retrieve(self, *a, **kw):
            result = orig_retrieve(self, *a, **kw)
            dense, sparse, evidence = result
            emit(audit, {"event": "retrieval", **context(),
                         "queries": kw.get("queries", a[2] if len(a) > 2 else None),
                         "dense_count": len(dense), "sparse_count": len(sparse),
                         "dense_ids": [hit["document"]["id"] for hit in dense],
                         "sparse_ids": [hit["document"]["id"] for hit in sparse],
                         "reranked_evidence": [
                             {"document": {k: hit["document"].get(k) for k in
                                           ("id", "clean_text", "source_path", "story_key")},
                              "rerank_score": hit.get("rerank_score"),
                              "fusion_score": hit.get("fusion_score")}
                             for hit in evidence
                         ]})
            return result

        def run(self, question, dialogue_context="", progress_callback=None):
            state["question_index"] += 1
            state.update(question=question, call=0, pending_documents=[])
            assert question == questions[state["question_index"] - 1] and not dialogue_context
            assert self.prompt_evidence_require_full_documents and self.answer_grounding_mode == "evidence_id"
            assert not getattr(self, "enable_experimental_grounding_repair", False)
            return orig_run(self, question, dialogue_context, progress_callback)

        VllmRunner._ensure_engine, VllmRunner.generate = ensure_engine, generate
        CPUInferencePipeline.run, CPUInferencePipeline._retrieve_round = run, retrieve
        sys.argv = [str(repo / "scripts/run_cpu_inference.py"),
                    "--runtime-config", str(repo / "configs/runtime_exx_official.json"),
                    "--questions-file", str(args.questions),
                    "--batch-output", str(out / "results.jsonl"), "--enforce-eager"]
        started = time.monotonic()
        code = 0
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            code = exc.code or 0
        rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
        complete = {"completed_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code,
                    "elapsed_sec": time.monotonic()-started, "count": len(rows),
                    "questions_match": [x["question"] for x in rows] == questions,
                    "results_sha256": sha(out / "results.jsonl"),
                    "runtime_errors": sum(bool(x.get("error")) for x in rows)}
        (out / "completion.json").write_text(json.dumps(complete, ensure_ascii=False, indent=2))
        print(json.dumps(complete), flush=True)
        raise SystemExit(code or int(len(rows) != 50 or not complete["questions_match"]
                                    or complete["runtime_errors"] != 0))


if __name__ == "__main__":
    main()
