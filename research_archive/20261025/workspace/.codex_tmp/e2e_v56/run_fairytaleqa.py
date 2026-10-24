"""Read-only model evaluation, create-only receipts, no service or training changes."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import string
import subprocess
import sys
import time
import traceback
from prepare_fairytaleqa import save, sha, digest

HERE = Path(__file__).resolve().parent
TRAIN = Path("${REMOTE_STORE}/exx_grounding_v1/runs/asa_v55_a03_20260911")
BASE = Path("${REMOTE_STORE}/ASA-ArknightStoryAgent/model/qwen3.5-4b")
PARENT = Path("${REMOTE_STORE}/exx_grounding_v1/models/"
              "exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827")
UUID = "GPU-17dfd5ca-b6d1-7cd5-fdc7-b1e95d13d960"
ALLOWED_RESIDENTS = {450914}


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def metrics(prediction, references):
    pred = normalize(prediction).split()
    result = {"normalized_exact_match": 0.0, "token_f1": 0.0, "rouge_l_f1": 0.0}
    for answer in references:
        ref = normalize(answer).split()
        exact = float(pred == ref)
        overlap = sum((Counter(pred) & Counter(ref)).values())
        f1 = 2 * overlap / (len(pred) + len(ref)) if pred or ref else 1.0
        previous = [0] * (len(ref) + 1)
        for word in pred:
            current = [0]
            for j, target in enumerate(ref):
                current.append(previous[j] + 1 if word == target else max(previous[j + 1], current[-1]))
            previous = current
        rouge = 2 * previous[-1] / (len(pred) + len(ref)) if pred or ref else 1.0
        for key, value in zip(result, (exact, f1, rouge)):
            result[key] = max(result[key], value)
    return result


def gpu():
    raw = subprocess.check_output(["nvidia-smi", "-i", "0",
        "--query-gpu=uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid, total, used, util = [x.strip() for x in raw]
    assert uuid == UUID
    rawp = subprocess.check_output(["nvidia-smi", "-i", "0",
        "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    processes = [{"pid": int(p.split(",")[0]), "memory_mib": int(p.split(",")[1])}
                 for p in rawp.splitlines() if p.strip()]
    return {"at_utc": now(), "uuid": uuid, "total_mib": int(total), "used_mib": int(used),
            "utilization": int(util), "processes": processes}


def prepare():
    assert not (HERE / "preflight.json").exists()
    from transformers import AutoTokenizer
    import torch, transformers, peft, vllm
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
    rows = json.loads((HERE / "benchmark/rows.json").read_text())
    protocol = json.loads((HERE / "benchmark/protocol.json").read_text())
    assert sha(HERE / "benchmark/rows.json") == protocol["bound_files"]["rows.json"]
    assert sha(HERE / "prepare_fairytaleqa.py") == protocol["bound_files"]["prepare_fairytaleqa.py"]
    assert not protocol["training_overlap_audit"]["exact_question_substrings"]
    assets = json.loads((TRAIN / "asset_hashes.json").read_text())
    bound = {}
    for kind, root in (("base", BASE), ("parent", PARENT)):
        for name, expected in assets[kind].items():
            path = root / name
            assert sha(path) == expected, str(path)
            bound[str(path)] = expected
    complete = json.loads((TRAIN / "complete.json").read_text())
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        bound[str(TRAIN / "model" / name)] = sha(TRAIN / "model" / name)
    assert bound[str(TRAIN / "model/adapter_model.safetensors")] == complete["adapter_sha256"]
    for relative in ("prepare_fairytaleqa.py", "run_fairytaleqa.py", "test_fairytaleqa.py",
                     "launch_eval.sh", "benchmark/rows.json", "benchmark/protocol.json"):
        bound[str(HERE / relative)] = sha(HERE / relative)
    for row in rows:
        messages = [{"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["user"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(ids) + protocol["generation"]["max_tokens"] <= 12000, row["id"]
        row.update(prompt=prompt, prompt_tokens=len(ids), prompt_sha256=digest(prompt),
                   input_ids_sha256=digest(json.dumps(ids)))
    save(HERE / "prompts.json", rows)
    bound[str(HERE / "prompts.json")] = sha(HERE / "prompts.json")
    record = {
        "at_utc": now(), "bound_files": bound,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__, "vllm": vllm.__version__},
        "conditions": {"parent": str(PARENT), "candidate": str(TRAIN / "model")},
        "prompt_rows_per_adapter": len(rows), "actual_generation_requests": len(rows) * 2,
        "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
        "input_truncated": False, "template": "native apply_chat_template enable_thinking=False",
        "generation": protocol["generation"],
        "engine": {"max_model_len": 12000, "max_num_batched_tokens": 4096, "max_num_seqs": 16,
                   "gpu_memory_utilization": 0.65, "enforce_eager": True,
                   "enable_prefix_caching": False, "language_model_only": True,
                   "enable_lora": True, "max_loras": 1, "max_lora_rank": 16, "dtype": "bfloat16"},
        "score_note": protocol["note"],
    }
    save(HERE / "preflight.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "bound_files"}), flush=True)


def run():
    assert not (HERE / "start.json").exists(), "Do not duplicate or retry this run"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    preflight = json.loads((HERE / "preflight.json").read_text())
    for path, expected in preflight["bound_files"].items():
        assert sha(Path(path)) == expected, path
    with (TRAIN / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        samples = []
        while True:
            sample = gpu()
            assert {p["pid"] for p in sample["processes"]} <= ALLOWED_RESIDENTS
            assert sample["total_mib"] - sample["used_mib"] >= 28000
            samples.append(sample)
            print(json.dumps({"resource_check": len(samples), **sample}), flush=True)
            if time.monotonic() - started >= 60:
                break
            time.sleep(5)
        save(HERE / "start.json", {"at_utc": now(), "pid": os.getpid(), "gpu_samples": samples,
                                  "preflight_sha256": sha(HERE / "preflight.json"),
                                  "shared_gpu": True, "other_jobs_interrupted": False})
        generate(preflight)


def generate(preflight):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=str(BASE), tokenizer=str(BASE), trust_remote_code=False,
              seed=20260911, **preflight["engine"])
    rows = json.loads((HERE / "prompts.json").read_text())
    params = SamplingParams(**preflight["generation"], top_p=1.0, repetition_penalty=1.0)
    all_results = []
    with (HERE / "generations.jsonl").open("x") as journal:
        for adapter_id, (name, path) in enumerate(preflight["conditions"].items(), 1):
            request = LoRARequest(name, adapter_id, path)
            for start in range(0, len(rows), 32):
                batch = rows[start:start + 32]
                outputs = llm.generate([r["prompt"] for r in batch], params,
                                       lora_request=request, use_tqdm=False)
                assert len(outputs) == len(batch)
                for row, output in zip(batch, outputs, strict=True):
                    assert output.lora_request.lora_int_id == adapter_id
                    assert digest(json.dumps(list(output.prompt_token_ids))) == row["input_ids_sha256"]
                    assert len(output.outputs) == 1
                    choice = output.outputs[0]
                    result = {
                        "at_utc": now(), "id": row["id"], "story": row["story"],
                        "attribute": row["attribute"], "condition": row["condition"], "adapter": name,
                        "prompt_sha256": row["prompt_sha256"], "raw_output": choice.text,
                        "generated_tokens": len(choice.token_ids), "finish_reason": choice.finish_reason,
                        "metrics": metrics(choice.text, row["answers"]),
                    }
                    all_results.append(result)
                    journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({"adapter": name, "rows_complete": start + len(batch),
                                  "total_rows": len(rows), "all_requests_complete": len(all_results),
                                  "at_utc": now()}), flush=True)
    expected_count = preflight["actual_generation_requests"]
    assert len(all_results) == expected_count
    summary = {"at_utc": now(), "generations": len(all_results), "groups": {},
               "preflight_sha256": sha(HERE / "preflight.json"),
               "generations_sha256": sha(HERE / "generations.jsonl"),
               "production_promoted": False}
    for name in preflight["conditions"]:
        for condition in ("oracle", "bm25_top3"):
            selected = [r for r in all_results if r["adapter"] == name and r["condition"] == condition]
            assert len(selected) == 1007
            summary["groups"][f"{name}/{condition}"] = {
                "n": len(selected),
                "finish_reasons": dict(Counter(r["finish_reason"] for r in selected)),
                **{key: sum(r["metrics"][key] for r in selected) / len(selected)
                   for key in selected[0]["metrics"]},
            }
    save(HERE / "results.json", summary)
    save(HERE / "complete.json", {"at_utc": now(), "exit_code": 0,
                                 "results_sha256": sha(HERE / "results.json"),
                                 "production_promoted": False})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        {"prepare": prepare, "run": run}[sys.argv[1]]()
    except BaseException as error:
        if sys.argv[1] == "run" and not (HERE / "failure.json").exists():
            save(HERE / "failure.json", {"at_utc": now(), "type": type(error).__name__,
                                       "message": str(error)[:1500], "pid": os.getpid()})
        traceback.print_exc()
        sys.exit(1)
