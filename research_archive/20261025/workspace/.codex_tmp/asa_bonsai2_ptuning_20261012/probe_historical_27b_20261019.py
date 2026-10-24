"""Bounded, inference-only historical 27B diagnostics after checkpoint evaluation.

Never changes adapters on disk. Uses one frozen backbone and swaps small
external adapters in memory. Conditional NLL preference is NOT QA accuracy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
POSITIONS = (1, 27, 31, 33, 46, 34, 37)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def launch(args):
    assert not args.out.exists()
    args.out.mkdir(mode=0o700)
    gpu_uuid = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"
    ], text=True).strip()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
               "--after-evaluation", str(args.after_evaluation), "--out", str(args.out),
               "--gpu", str(args.gpu)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, OMP_NUM_THREADS="8",
               TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True")
    with (args.out / "worker.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, command=command, gpu_uuid=gpu_uuid,
                   after_evaluation=str(args.after_evaluation), positions=POSITIONS,
                   max_wait_seconds=10800, max_compute_seconds=7200,
                   script_sha256=sha(__file__), training_performed=False,
                   original_training_unmodified=True, production_modified=False)
    save(args.out / "launch.json", receipt)
    print(json.dumps(receipt, indent=2))


def worker(args):
    state = dict(stage="waiting_for_evaluation", pid=os.getpid(),
                 started_unix=time.time(), completed_conditions=0)
    status = args.out / "status.json"
    save(status, state)
    try:
        deadline = time.monotonic() + 10800
        while True:
            evaluation = json.loads((args.after_evaluation / "status.json").read_text())
            if evaluation["stage"] == "failed":
                raise RuntimeError("Dependency evaluation failed; refusing to overlap GPU usage")
            if evaluation["stage"] == "complete":
                free = subprocess.check_output([
                    "nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"], text=True).strip()
                if int(free) >= 22000:
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("Evaluation/GPU dependency did not clear within three hours")
            time.sleep(20)
        state.update(stage="loading", compute_started_unix=time.time())
        save(status, state)
        # Bounded compute lifetime, also if a native call never returns.
        import signal
        def compute_timeout(*_):
            raise TimeoutError("Two-hour bounded diagnostic deadline")
        signal.signal(signal.SIGALRM, compute_timeout)
        signal.alarm(7200)
        run_probe(args.out, state, status)
        signal.alarm(0)
        state.update(stage="complete", finished_unix=time.time())
        save(status, state)
    except BaseException as exc:
        state.update(stage="failed", error=repr(exc), finished_unix=time.time())
        save(status, state)
        raise


def run_probe(out, state, status):
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from backend import load_backend, PromptEncoder
    from ptuning_v2_hybrid import DeepPrefix, PrefixController, architecture

    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.95)
    shallow_path = ROOT / "targeted_pt_v2_20261014_attempt02/trained_v2"
    deep_path = ROOT / "ptuning_v2_deep_20261017_attempt01"
    task_path = ROOT / "native46_bonsai_pt_v2_20261016/tasks.jsonl"
    tasks = read_rows(task_path)
    deep_outputs = {r["id"]: r for r in read_rows(deep_path / "generated_native46.jsonl")}
    shallow_outputs = {
        r["id"]: r for shard in (0, 1) for r in read_rows(
            ROOT / f"native46_bonsai_pt_v2_20261016/bonsai_shard{shard}.results.jsonl")}
    selected = []
    tokenizer = Tokenizer.from_file(str(ROOT / "official/tokenizer.json"))
    bound_files = [task_path, deep_path / "generated_native46.jsonl",
                   ROOT / "backend.py", ROOT / "ptuning_v2_hybrid.py"]
    bound_files.extend(ROOT / f"native46_bonsai_pt_v2_20261016/bonsai_shard{shard}.results.jsonl"
                       for shard in (0, 1))
    for path, name in ((shallow_path, "prompt_encoder"), (deep_path, "deep_prefix")):
        for stage in ("initial_adapter", "final_adapter"):
            bound_files.append(path / stage / f"{name}.safetensors")
    identity = dict(
        kind="historical_750_example_inference_only_functional_diagnostic",
        positions=POSITIONS, files_sha256={str(p.relative_to(ROOT)): sha(p) for p in bound_files},
        script_sha256=sha(__file__), no_input_truncation=True, evidence_count=20,
        training_performed=False, QA_accuracy_measured=False, original_training_unmodified=True,
        preference_caveats=[
            "Historical complete/partial answers are candidates, not independent gold.",
            "Canonical JSON controls whitespace only; answer length, facts and citations still differ.",
            "Teacher-forced NLL preference does not establish generated-answer correctness.",
            "Branch ablations test interventions, not exclusive causal attribution.",
        ])
    for position in POSITIONS:
        task = tasks[position - 1]
        assert task["position"] == position and len(task["documents"]) == 20
        assert sha_text(task["bonsai_prompt"]) == task["bonsai_prompt_sha256"]
        assert all(d["text"] in task["bonsai_prompt"] and sha_text(d["text"]) == d["sha256"]
                   for d in task["documents"])
        assert deep_outputs[task["id"]]["prompt_sha256"] == task["bonsai_prompt_sha256"]
        assert shallow_outputs[task["id"]]["prompt_sha256"] == task["bonsai_prompt_sha256"]
        p = tokenizer.encode(task["bonsai_prompt"], add_special_tokens=False).ids
        candidates = {}
        for name, parsed in (
            ("historical_shallow_answer",
             shallow_outputs[task["id"]]["parsed_output_for_offline_inspection_only"]),
            ("historical_deep_answer", deep_outputs[task["id"]]["parsed"])):
            text = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            t = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False).ids
            assert 16 + len(p) + len(t) <= 32768
            candidates[name] = t
        selected.append((position, task, p, candidates))
    save(out / "identity.json", identity)
    model, manifest = load_backend(ROOT / "artifact", dtype=torch.bfloat16)
    model.eval()
    model.gradient_checkpointing_disable()
    initial_shallow = PromptEncoder.load(shallow_path / "initial_adapter")
    final_shallow = PromptEncoder.load(shallow_path / "final_adapter")
    initial_shallow.eval()
    final_shallow.eval()
    with torch.no_grad():
        initial_seed = initial_shallow().detach().clone()
        final_seed = final_shallow().detach().clone()
    del initial_shallow, final_shallow
    spec = architecture(model.config)
    initial_deep = DeepPrefix.load(
        deep_path / "initial_adapter", spec, manifest["source_sha256"])
    final_deep = DeepPrefix.load(
        deep_path / "final_adapter", spec, manifest["source_sha256"])
    adapter = DeepPrefix(spec, 16).cuda()
    control = PrefixController(model, adapter)
    identity["backbone_source_sha256"] = manifest["source_sha256"]
    save(out / "identity.json", identity)

    def prepare(ids, seed=None):
        embedded = model.model.embed_tokens(ids)
        if seed is None:
            return adapter.prepare(embedded)
        return torch.cat((seed.to(embedded.dtype).unsqueeze(0), embedded), dim=1)

    @torch.no_grad()
    def score(p, t, seed=None):
        ids = torch.tensor([p + t], device="cuda")
        positions = torch.arange(16 + len(p) - 1, 16 + len(p) + len(t) - 1, device="cuda")
        with control.activate(prefill=seed is None):
            logits = model(inputs_embeds=prepare(ids, seed), use_cache=False,
                           logits_to_keep=positions).logits[0].float()
        loss = F.cross_entropy(logits, torch.tensor(t, device="cuda"), reduction="sum").item()
        return dict(tokens=len(t), total_nll=loss, mean_nll=loss / len(t))

    @torch.no_grad()
    def next_logits(p, seed=None):
        ids = torch.tensor([p], device="cuda")
        with control.activate(prefill=seed is None):
            return model(inputs_embeds=prepare(ids, seed), use_cache=False,
                         logits_to_keep=1).logits[0, -1].float()

    def emit(stream, entry):
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        stream.flush()
        state.update(stage="probing", completed_conditions=state["completed_conditions"] + 1,
                     last_condition=entry.get("arm", entry["event"]), updated_unix=time.time())
        save(status, state)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    try:
        with torch.no_grad(), (out / "measurements.jsonl").open("x") as stream:
            short = torch.tensor([tokenizer.encode("只根据证据回答。", add_special_tokens=False).ids],
                                 device="cuda")
            control.calibrate(final_seed, short)
            # Keep the historical FIXED seed: changes only legal deep prefix params.
            adapter.input_seed.copy_(initial_deep.input_seed)
            transplant = {k: p.detach().clone() for k, p in adapter.state_dict().items()}
            for position, task, p, candidates in selected:
                reference = next_logits(p, final_seed)
                got = next_logits(p)
                delta = (reference - got).abs()
                emit(stream, dict(
                    event="trained_shallow_to_fixed_seed_deep_mapping", position=position,
                    prompt_tokens=len(p), max_abs=delta.max().item(), mean_abs=delta.mean().item(),
                    top1_equal=bool(reference.argmax() == got.argmax()),
                    tolerance_gate_passed=delta.max().item() <= 0.01))
            for arm in ("initial_shallow", "final_shallow", "initial_deep", "final_deep",
                        "transplanted_final_shallow", "gdn_delta_only", "kv_delta_only"):
                seed = initial_seed if arm == "initial_shallow" else (
                    final_seed if arm == "final_shallow" else None)
                if arm in ("initial_deep", "final_deep"):
                    adapter.load_state_dict(
                        (initial_deep if arm == "initial_deep" else final_deep).state_dict())
                elif arm == "transplanted_final_shallow":
                    adapter.load_state_dict(transplant)
                elif arm in ("gdn_delta_only", "kv_delta_only"):
                    adapter.load_state_dict(initial_deep.state_dict())
                    for name, parameter in adapter.prefix.items():
                        selected_branch = name.endswith("_gdn") == (arm == "gdn_delta_only")
                        if selected_branch:
                            parameter.copy_(final_deep.prefix[name])
                for position, task, p, candidates in selected:
                    measurements = {name: score(p, t, seed) for name, t in candidates.items()}
                    emit(stream, dict(
                        event="paired_candidate_teacher_forced_nll_not_accuracy",
                        arm=arm, position=position, candidates=measurements,
                        mean_nll_deep_minus_shallow=(
                            measurements["historical_deep_answer"]["mean_nll"]
                            - measurements["historical_shallow_answer"]["mean_nll"])))
            # Long real-prompt cache check, with baseline to expose inherent BF16 error.
            position, task, p, _ = selected[4]  # Historical regression #46.
            adapter.load_state_dict(final_deep.state_dict())
            for arm, seed in (("final_shallow_cache", final_seed), ("final_deep_cache", None)):
                ids = torch.tensor([p], device="cuda")
                with control.activate(prefill=seed is None):
                    output = model(inputs_embeds=prepare(ids, seed),
                                   use_cache=True, logits_to_keep=1)
                generated = []
                for step in range(8):
                    token = int(output.logits[0, -1].argmax())
                    generated.append(token)
                    with control.activate(prefill=False):
                        output = model(
                            input_ids=torch.tensor([[token]], device="cuda"),
                            past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
                    cached = output.logits[0, -1].float()
                    full = next_logits(p + generated, seed)
                    delta = (cached - full).abs()
                    emit(stream, dict(
                        event="long_real_prompt_cache_vs_recompute", arm=arm,
                        position=position, decode_step=step + 1, prompt_tokens=len(p),
                        max_abs=delta.max().item(), mean_abs=delta.mean().item(),
                        top1_equal=bool(cached.argmax() == full.argmax()),
                        full_top1=int(full.argmax()), cached_top1=int(cached.argmax())))
        assert identity["files_sha256"] == {
            str(p.relative_to(ROOT)): sha(p) for p in bound_files}
        save(out / "receipt.json", dict(
            completed=True, conditions=state["completed_conditions"],
            measurements_sha256=sha(out / "measurements.jsonl"),
            historical_weights_and_code_unchanged=True,
            generated_QA_accuracy_not_measured=True))
    finally:
        control.close()


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "worker"))
    parser.add_argument("--after-evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.after_evaluation = args.after_evaluation.resolve()
    args.out = args.out.resolve()
    (launch if args.command == "launch" else worker)(args)


if __name__ == "__main__":
    main()
