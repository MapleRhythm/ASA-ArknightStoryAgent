"""Native-protocol, frozen full20 evidence comparison; no answer filtering."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ADAPTER_SHA = "20c369a0042478f50b4128c047eb03844bd0e74c8eccdf6139e3344a396ef0ce"
ADAPTER_NAME = "teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2"


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def check_task(task):
    assert len(task["documents"]) == 20
    assert re.fullmatch(r"[12]/2", task["round"])
    assert len({d["evidence_id"] for d in task["documents"]}) == 20
    assert "target" not in task and "reference_answer" not in task
    normal = lambda s: " ".join(s.split())
    for d in task["documents"]:
        assert text_sha(d["text"]) == d["sha256"], "evidence hash mismatch"
        assert d["text"] in task["bonsai_prompt"], "Bonsai missing full evidence"
        assert normal(d["text"]) in normal(task["fourb_prompt"]), "4B missing full evidence"
        assert d["document_id"] in task["fourb_prompt"], "4B missing native ID"
    for arm in ("bonsai", "fourb"):
        assert text_sha(task[f"{arm}_prompt"]) == task[f"{arm}_prompt_sha256"]


def parse_raw(raw, prefix):
    content = prefix + raw
    for token in ("<|im_end|>", "<|endoftext|>"):
        content = content.removesuffix(token)
    try:
        return json.loads(content)
    except ValueError:
        return None


def shard_tasks(tasks, index):
    assert index in (0, 1)
    return tasks[index::2]


def prepare(args):
    from audit_fresh_subset import verify
    verified = verify(args.data)
    source = rows(args.data / "val.jsonl")
    assert len(source) == len({r["id"] for r in source}) == 46
    run = json.loads((args.bonsai_run / "run.json").read_text())
    assert run["training_complete"] and run["backbone_bytewise_unchanged"]
    assert run["data_sha256"]["val"] == file_sha(args.data / "val.jsonl")
    assert run["reviewed_subset_preflight"]["audit_sha256"] == verified["audit_sha256"]
    adapter = args.repo / "model/lora" / ADAPTER_NAME
    assert file_sha(adapter / "adapter_model.safetensors") == ADAPTER_SHA
    spec = importlib.util.spec_from_file_location("native46_historical_pipeline", args.historical_pipeline)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tasks = []
    for position, row in enumerate(source, 1):
        assert not row["truncated"] and row["split"] == "val"
        evidence = [
            {"document": {"id": d["document_id"], "clean_text": d["text"]},
             "prompt_prefer_clean_text": True} for d in row["documents"]]
        hyp = module.HypothesisDocument(
            question=row["question"], query_type="reasoning",
            expected_answer_type="剧情解释", intent="剧情解释",
            entities=[], keywords=[], dialogue_context="")
        current, total = map(int, row["round"].split("/"))
        native = module.build_conclusion_prompt(
            row["question"], hyp, evidence, [], current, total, 20,
            prompt_evidence=evidence, evidence_max_chars_per_doc=None,
            evidence_max_total_chars=None, prompt_mode="minimal")
        task = dict(position=position, id=row["id"], question=row["question"],
                    question_group=re.sub(r"\W", "", row["question"]).lower(),
                    round=row["round"], documents=row["documents"],
                    bonsai_prompt=row["prompt"], fourb_prompt=native,
                    bonsai_prompt_sha256=text_sha(row["prompt"]),
                    fourb_prompt_sha256=text_sha(native))
        check_task(task)
        tasks.append(task)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "tasks.jsonl").open("x") as stream:
        for task in tasks:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    manifest = dict(
        kind="native_protocol_paired46_frozen_evidence_not_unseen_or_full_chain",
        prepared_utc=now(), question_count=len(tasks),
        unique_question_count=len({t["question_group"] for t in tasks}),
        expected_generations=92, task_order=[t["id"] for t in tasks],
        tasks_sha256=file_sha(args.out / "tasks.jsonl"),
        data_val_sha256=file_sha(args.data / "val.jsonl"),
        reviewed_dataset_audit_sha256=verified["audit_sha256"],
        rounds=dict(Counter(t["round"] for t in tasks)),
        base_4b=str(args.repo / "model/qwen3.5-4b"),
        adapter_4b=str(adapter), adapter_4b_sha256=ADAPTER_SHA,
        adapter_4b_config_sha256=file_sha(adapter / "adapter_config.json"),
        base_4b_config_sha256=file_sha(args.repo / "model/qwen3.5-4b/config.json"),
        bonsai_run=str(args.bonsai_run),
        bonsai_run_sha256=file_sha(args.bonsai_run / "run.json"),
        bonsai_adapter_sha256=file_sha(args.bonsai_run / "final_adapter/prompt_encoder.safetensors"),
        bonsai_adapter_config_sha256=file_sha(args.bonsai_run / "final_adapter/adapter_config.json"),
        backend_sha256=file_sha(Path(__file__).with_name("backend.py")),
        script_sha256=file_sha(__file__),
        historical_pipeline_sha256=file_sha(args.historical_pipeline),
        protocol_sha256=file_sha(args.protocol),
        generation=dict(greedy=True, max_new_tokens=1024, repetition_penalty=1.0,
                        temperature=0.0, top_p=1.0, seed=20261014),
        evidence_count=20, input_truncation=False, answer_validation=False,
        reference_answers_sent=False, api_calls=0, retraining=False,
        production_modified=False, other_jobs_stopped=False,
        reuse_previous_outputs=False,
        native_protocols={"fourb": "historical minimal grounded_action_v1 quote",
                          "bonsai": "trained grounded_action_exx_v1 E-ID plus final soft prompt"},
        known_limitations=[
            "historical development questions, not blind/unseen benchmark",
            "three question texts appear twice with different evidence or round; report grouped results too",
            "46 selected without outcome filtering; all heldout val rows",
            "41 answer/4 abstain/1 retrieve are teacher action labels, not semantic gold",
            "different native prompt templates, weight types, tokenizers and numerical backends",
            "output token budget equal but protocols have different verbosity",
        ])
    save_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def worker(args):
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert file_sha(args.out / "tasks.jsonl") == manifest["tasks_sha256"]
    assert file_sha(__file__) == manifest["script_sha256"]
    tasks = rows(args.out / "tasks.jsonl")
    for task in tasks:
        check_task(task)
    if args.arm == "bonsai":
        tasks = shard_tasks(tasks, args.shard)
    name = "fourb" if args.arm == "fourb" else f"bonsai_shard{args.shard}"
    state_path = args.out / f"{name}.status.json"
    assert not state_path.exists()
    state = dict(arm=args.arm, shard=args.shard, pid=os.getpid(), started_utc=now(),
                 stage="loading", completed=0, expected=len(tasks),
                 cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    save_json(state_path, state)
    try:
        free = int(subprocess.check_output(
            ["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        assert free >= 20500, f"insufficient GPU free memory: {free} MiB"
        assert file_sha(manifest["adapter_4b"] + "/adapter_model.safetensors") == ADAPTER_SHA
        if args.arm == "fourb":
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            engine = LLM(
                model=manifest["base_4b"], tokenizer=manifest["adapter_4b"],
                trust_remote_code=True, enable_lora=True, max_lora_rank=16,
                tensor_parallel_size=1, gpu_memory_utilization=0.70,
                max_model_len=12000, dtype="bfloat16", enforce_eager=True,
                disable_log_stats=True, seed=20261014,
                max_num_seqs=1, max_num_batched_tokens=8192)
            tokenizer = engine.get_tokenizer()
            lora = LoRARequest("historical_sft_kto", 1, manifest["adapter_4b"])
            eos = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
            sampling = SamplingParams(temperature=0.0, top_p=1.0, repetition_penalty=1.0,
                                      max_tokens=1024, stop_token_ids=eos,
                                      skip_special_tokens=False)
        else:
            import torch
            from tokenizers import Tokenizer
            from backend import PromptEncoder, load_backend
            root = Path(__file__).resolve().parent
            assert file_sha(root / "backend.py") == manifest["backend_sha256"]
            assert file_sha(Path(manifest["bonsai_run"]) / "final_adapter/prompt_encoder.safetensors") == manifest["bonsai_adapter_sha256"]
            tokenizer = Tokenizer.from_file(str(root / "official/tokenizer.json"))
            model, backbone_manifest = load_backend(root / "artifact", dtype=torch.bfloat16)
            model.eval()
            encoder = PromptEncoder.load(Path(manifest["bonsai_run"]) / "final_adapter")
            encoder.eval()
            eos = [tokenizer.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")]
            state["source_sha256"] = backbone_manifest["source_sha256"]
        assert all(isinstance(i, int) and i >= 0 for i in eos)
        state.update(stage="generating", gpu_free_before_MiB=free)
        save_json(state_path, state)
        with (args.out / f"{name}.results.jsonl").open("x") as stream:
            for task in tasks:
                prompt_key = "fourb" if args.arm == "fourb" else "bonsai"
                prompt = task[f"{prompt_key}_prompt"]
                begin = time.monotonic()
                if args.arm == "fourb":
                    ids = tokenizer.encode(prompt, add_special_tokens=False)
                    assert len(ids) + 1024 <= 12000, "refuse input truncation"
                    outputs = engine.generate([{"prompt_token_ids": ids}], sampling,
                                              lora_request=lora, use_tqdm=False)
                    assert list(outputs[0].prompt_token_ids) == ids
                    choice = outputs[0].outputs[0]
                    tokens = list(choice.token_ids)
                    raw, finish = choice.text, choice.finish_reason
                    prefix = "{"
                    stop = choice.stop_reason
                else:
                    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
                    assert len(ids) + encoder.length + 1024 <= 32768, "refuse input truncation"
                    with torch.no_grad():
                        embedded = model.model.embed_tokens(torch.tensor([ids], device="cuda"))
                        embedded = encoder.inject(embedded)
                        generated = model.generate(
                            inputs_embeds=embedded,
                            attention_mask=torch.ones(embedded.shape[:2], device="cuda", dtype=torch.long),
                            max_new_tokens=1024, do_sample=False, use_cache=True,
                            eos_token_id=eos, pad_token_id=eos[0])
                    tokens = generated[0].tolist()
                    raw = tokenizer.decode(tokens, skip_special_tokens=False)
                    finish = "stop" if tokens and tokens[-1] in eos else "length"
                    prefix = ""
                    stop = tokens[-1] if tokens and tokens[-1] in eos else None
                record = dict(
                    arm=args.arm, shard=args.shard, position=task["position"],
                    id=task["id"], question=task["question"], round=task["round"],
                    question_group=task["question_group"],
                    prompt_sha256=task[f"{prompt_key}_prompt_sha256"],
                    input_tokens=len(ids), input_token_ids_sha256=text_sha(json.dumps(ids)),
                    evidence_count=20, input_truncated=False, answer_validation=False,
                    raw_output=raw, protocol_prefill=prefix, output_token_ids=tokens,
                    finish_reason=finish, stop_reason=stop,
                    output_reached_length_limit=finish == "length",
                    parsed_output_for_offline_inspection_only=parse_raw(raw, prefix),
                    elapsed_seconds=time.monotonic() - begin)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                state.update(completed=state["completed"] + 1, last_id=task["id"], updated_utc=now())
                save_json(state_path, state)
                print(json.dumps({k:record[k] for k in
                                  ("arm", "position", "id", "finish_reason", "elapsed_seconds")}), flush=True)
        state.update(stage="completed", finished_utc=now(),
                     results_sha256=file_sha(args.out / f"{name}.results.jsonl"),
                     semantic_adjudication_complete=False)
        save_json(state_path, state)
    except Exception as error:
        state.update(stage="failed", error=f"{type(error).__name__}: {error}", finished_utc=now())
        save_json(state_path, state)
        raise


def lane(args):
    path = args.out / f"lane{args.lane}.json"
    assert not path.exists()
    record = dict(pid=os.getpid(), lane=args.lane, stage="starting", started_utc=now())
    save_json(path, record)
    stages = [("bonsai", 0)] if args.lane == 2 else [("fourb", 0), ("bonsai", 1)]
    try:
        for arm, shard in stages:
            name = "fourb" if arm == "fourb" else f"bonsai_shard{shard}"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.lane),
                       OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
                       VLLM_USE_FLASHINFER_SAMPLER="0",
                       PYTORCH_ALLOC_CONF="expandable_segments:True",
                       PYTHONPATH="${REMOTE_HOME}/ASA-ArknightStoryAgent/src")
            command = [sys.executable, "-u", __file__, "worker", "--out", str(args.out),
                       "--arm", arm, "--shard", str(shard)]
            with (args.out / f"{name}.log").open("x") as log:
                child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                record.update(stage=name, child_pid=child.pid)
                save_json(path, record)
                code = child.wait()
            record[name + "_exit_code"] = code
            if code:
                raise RuntimeError(f"{name} exit {code}")
            # Allow asynchronous CUDA process teardown before the next independent model.
            for _ in range(20):
                free = int(subprocess.check_output(
                    ["nvidia-smi", "-i", str(args.lane), "--query-gpu=memory.free",
                     "--format=csv,noheader,nounits"], text=True).strip())
                if free >= 20500:
                    break
                time.sleep(1)
        record.update(stage="completed", finished_utc=now())
        save_json(path, record)
    except Exception as error:
        record.update(stage="failed", error=str(error), finished_utc=now())
        save_json(path, record)
        raise


def launch(args):
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert manifest["tasks_sha256"] == file_sha(args.out / "tasks.jsonl")
    path = args.out / "launch.json"
    assert not path.exists()
    for gpu in (1, 2):
        free = int(subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True).strip())
        assert free >= 22000, f"GPU {gpu} insufficient free memory: {free}"
    launched = dict(started_utc=now(), gpu_lanes={}, production_modified=False)
    for gpu in (1, 2):
        with (args.out / f"lane{gpu}.log").open("x") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", __file__, "lane", "--out", str(args.out), "--lane", str(gpu)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        launched["gpu_lanes"][str(gpu)] = child.pid
        save_json(path, launched)
    print(json.dumps(launched), flush=True)


def status(args):
    manifest = json.loads((args.out / "manifest.json").read_text())
    result = dict(checked_utc=now(), expected=manifest["expected_generations"],
                  unique_questions=manifest["unique_question_count"], workers={}, completed=0)
    for name in ("fourb", "bonsai_shard0", "bonsai_shard1"):
        p = args.out / f"{name}.status.json"
        value = json.loads(p.read_text()) if p.exists() else {"stage": "not_started", "completed": 0}
        result["workers"][name] = value
        result["completed"] += value["completed"]
    result["all_generation_complete"] = all(
        x["stage"] == "completed" for x in result["workers"].values())
    result["semantic_adjudication_complete"] = False
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="mode", required=True)
    p = subs.add_parser("prepare")
    for name in ("repo", "data", "bonsai-run", "historical-pipeline", "protocol", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p = subs.add_parser("worker")
    p.add_argument("--arm", choices=("fourb", "bonsai"), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p = subs.add_parser("lane")
    p.add_argument("--lane", type=int, choices=(1, 2), required=True)
    p.add_argument("--out", type=Path, required=True)
    for name in ("launch", "status"):
        p = subs.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
