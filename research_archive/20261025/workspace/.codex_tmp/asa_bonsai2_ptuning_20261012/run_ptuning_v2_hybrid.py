"""Isolated deep-prefix gate -> train -> fixed-evidence generation workflow.

No API calls, production writes, answer validation, evidence truncation, or
changes to the historical shallow-prompt experiments.
"""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parent
CODE_FILES = ("backend.py", "full_gate.py", "ptuning_v2_hybrid.py",
              "run_ptuning_v2_hybrid.py", "test_ptuning_v2_hybrid.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def emit(out, event, **payload):
    entry = dict(event=event, unix_time=time.time(), **payload)
    with (out / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def environment(out):
    require(out.is_absolute() and out.stat().st_dev != Path.home().stat().st_dev,
            "Use the experiment data filesystem, not the home/system disk")
    require(shutil.disk_usage(out).free >= 20 * 1024**3, "Need 20 GiB data disk space")
    available = next(int(line.split()[1]) * 1024
                     for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    require(available >= 60 * 1024**3, "Need 60 GiB available RAM for activation offload")
    cache = out / "runtime_cache"
    require(not cache.is_symlink(), "Refusing symlinked cache")
    cache.mkdir(mode=0o700, exist_ok=True)
    for key, sub in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                     ("CUDA_CACHE_PATH", "cuda"), ("XDG_CACHE_HOME", "xdg"), ("TMPDIR", "tmp")):
        target = cache / sub
        require(not target.is_symlink(), "Refusing symlinked cache directory")
        target.mkdir(mode=0o700, exist_ok=True)
        os.environ[key] = str(target)


def load_data(data, length):
    from tokenizers import Tokenizer
    from targeted_pt_v2 import verify_merged
    manifest = verify_merged(data)
    require(manifest["ready_for_training"] and manifest["train"] == 750
            and manifest["val"] == 46, "Require unchanged audited 750/46 data")
    tokenizer = Tokenizer.from_file(str(ROOT / "official/tokenizer.json"))
    rows = {}
    for split in ("train", "val"):
        rows[split] = []
        for line in (data / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            require(row["evidence_count"] == 20 and len(row["documents"]) == 20
                    and not row["truncated"], "Incomplete evidence")
            require(all(d["text"] in row["prompt"] for d in row["documents"]),
                    "Evidence missing from actual prompt")
            p = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
            t = tokenizer.encode(row["target"] + "<|im_end|>", add_special_tokens=False).ids
            require(p and t and len(p) + len(t) + length <= 32768,
                    f"Refusing empty/oversize sequence: {row['id']}")
            rows[split].append((row, p, t))
    return manifest, tokenizer, rows


def loss_for(model, adapter, item):
    import torch
    import torch.nn.functional as F
    _, p, t = item
    ids = torch.tensor([p + t], device="cuda")
    embedded = model.model.embed_tokens(ids).detach()
    positions = torch.arange(adapter.length + len(p) - 1,
                             adapter.length + len(p) + len(t) - 1, device="cuda")
    logits = model(inputs_embeds=adapter.prepare(embedded), use_cache=False,
                   logits_to_keep=positions).logits[0]
    return F.cross_entropy(logits.float(), torch.tensor(t, device="cuda"))


def backward(model, adapter, control, item):
    import torch
    # Scope includes backward: checkpoint re-execution must see the same hooks.
    with control.activate(), torch.autograd.graph.save_on_cpu(pin_memory=True):
        loss = loss_for(model, adapter, item)
        require(bool(torch.isfinite(loss)), "Nonfinite loss")
        loss.backward()
    stats = adapter.assert_gradients()
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()),
            "Backbone not frozen")
    return loss, stats


def cache_check(model, adapter, control, ids, inject=True):
    import torch
    values = []
    with torch.no_grad(), control.activate(prefill=inject):
        out = model(inputs_embeds=adapter.prepare(model.model.embed_tokens(ids)),
                    use_cache=True, logits_to_keep=1)
    for _ in range(3):
        next_id = out.logits[:, -1:].argmax(-1)
        ids = torch.cat((ids, next_id), dim=1)
        with torch.no_grad(), control.activate(prefill=False):
            out = model(input_ids=next_id, past_key_values=out.past_key_values,
                        use_cache=True, logits_to_keep=1)
        with torch.no_grad(), control.activate(prefill=inject):
            full = model(inputs_embeds=adapter.prepare(model.model.embed_tokens(ids)),
                         use_cache=False, logits_to_keep=1).logits
        a, b = out.logits[0, -1].float(), full[0, -1].float()
        pa, pb = a.log_softmax(-1), b.log_softmax(-1)
        values.append(dict(
            max_abs=(a - b).abs().max().item(),
            probability_max_abs=(pa.exp() - pb.exp()).abs().max().item(),
            kl=(pb.exp() * (pb - pa)).sum().clamp_min(0).item(),
            top1_equal=bool(a.argmax() == b.argmax()),
        ))
    return values


def run_gate(out, model, manifest, adapter, control, tokenizer, rows, initial_path, identity):
    import torch
    from backend import PromptEncoder
    from full_gate import checksum_backbone
    from ptuning_v2_hybrid import DeepPrefix
    torch.manual_seed(20261012)
    gate = dict(completed=False, kind="real_27b_deep_prefix_training_gate",
                identity=identity, activation_offload=True)
    atomic_json(out / "gate.json", gate)
    before = checksum_backbone(model)
    emit(out, "gate_backbone_hashed", sha256=before)
    ids = torch.tensor([tokenizer.encode(
        "<|im_start|>user\n只根据证据回答：一加一等于几？<|im_end|>\n"
        "<|im_start|>assistant\n", add_special_tokens=False).ids], device="cuda")
    initial_config = json.loads((initial_path / "adapter_config.json").read_text())
    require(initial_config["source_sha256"] == manifest["source_sha256"],
            "Initial shallow adapter backbone mismatch")
    require(initial_config["length"] >= adapter.length, "Not enough initial prefix tokens")
    shallow = PromptEncoder.load(initial_path)
    seed = shallow().detach()[:adapter.length]
    reference = control.calibrate(seed, ids)
    del shallow, seed
    with torch.no_grad(), control.activate():
        got = model(inputs_embeds=adapter.prepare(model.model.embed_tokens(ids)),
                    use_cache=False, logits_to_keep=1).logits
    gate["calibration_logits_max_abs"] = (reference.float() - got.float()).abs().max().item()
    atomic_json(out / "gate.json", gate)
    require(gate["calibration_logits_max_abs"] <= 0.01,
            "Deep initialization failed to reproduce shallow prompt")
    adapter.save(out / "initial_adapter", manifest["source_sha256"])
    restored = DeepPrefix.load(out / "initial_adapter", adapter.spec, manifest["source_sha256"])
    require(all(torch.equal(v, restored.state_dict()[k]) for k, v in adapter.state_dict().items()),
            "Adapter reload mismatch")
    gate["reload_parameters_exact"] = True
    del restored, reference, got
    emit(out, "gate_calibration_and_reload_passed",
         calibration_logits_max_abs=gate["calibration_logits_max_abs"])
    model.eval()
    gate["baseline_cache_comparison"] = cache_check(model, adapter, control, ids, inject=False)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    short = ({"id": "synthetic_backend_smoke_not_QA"}, ids[0].tolist(),
             tokenizer.encode("二。<|im_end|>", add_special_tokens=False).ids)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-5)
    optimizer.zero_grad(set_to_none=True)
    old = {k: p.detach().clone() for k, p in adapter.prefix.items()}
    loss, stats = backward(model, adapter, control, short)
    gate["short_loss"] = loss.item()
    gate["short_gradient_stats"] = stats
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
    optimizer.step()
    gate["updated_parameter_blocks"] = [
        k for k, p in adapter.prefix.items() if not torch.equal(p, old[k])]
    require(len(gate["updated_parameter_blocks"]) == len(adapter.prefix),
            "Some layer prefixes did not update")
    del optimizer, old, loss
    adapter.zero_grad(set_to_none=True)
    model.eval()
    # Disable checkpointing for actual cached inference; train re-enables below.
    model.gradient_checkpointing_disable()
    gate["trained_cache_comparison"] = cache_check(model, adapter, control, ids)
    atomic_json(out / "gate.json", gate)
    for result in gate["trained_cache_comparison"]:
        require(result["top1_equal"] and result["max_abs"] <= 0.5
                and result["probability_max_abs"] <= 0.02 and result["kl"] <= 0.01,
                f"Cached decode does not agree with full recomputation: {result}")
    emit(out, "gate_short_backward_and_cache_passed",
         short_loss=gate["short_loss"], cache=gate["trained_cache_comparison"])
    # Remove gate-only optimizer changes: task training starts from INITIAL prompt.
    restored = DeepPrefix.load(out / "initial_adapter", adapter.spec, manifest["source_sha256"])
    adapter.load_state_dict(restored.state_dict())
    del restored
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    longest = max(rows["train"] + rows["val"], key=lambda item: len(item[1]) + len(item[2]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    emit(out, "gate_longest_backward_started", id=longest[0]["id"],
         prompt_tokens=len(longest[1]), target_tokens=len(longest[2]), evidence_count=20)
    loss, stats = backward(model, adapter, control, longest)
    gate.update(longest_id=longest[0]["id"], longest_prompt_tokens=len(longest[1]),
                longest_target_tokens=len(longest[2]), longest_loss=loss.item(),
                longest_gradient_stats=stats,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    del loss
    adapter.zero_grad(set_to_none=True)
    require(checksum_backbone(model) == before, "Backbone changed in training gate")
    gate.update(completed=True, frozen_backbone_unchanged=True,
                backbone_sha256=before, all_64_layer_prefixes_received_gradients=True)
    atomic_json(out / "gate.json", gate)
    emit(out, "gate_passed", peak_cuda_allocated_bytes=gate["peak_cuda_allocated_bytes"])
    return gate


def save_checkpoint(out, step, adapter, optimizer, source):
    import torch
    # Adapter + optimizer + RNG in ONE committed directory. Pointer advances last.
    # Unique transaction name allows safe replay after a crash between directory
    # commit and pointer update; an orphan is retained rather than overwritten.
    name = f"checkpoint-{step}-{time.time_ns()}"
    staging = out / f"{name}.pending"
    target = out / name
    adapter.save(staging, source)
    torch.save(dict(step=step, optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()),
               staging / "training_state.pt")
    os.replace(staging, target)
    atomic_json(out / "latest_checkpoint.json", {
        "step": step, "directory": target.name,
        "training_state_sha256": sha(target / "training_state.pt"),
        "adapter_sha256": sha(target / "deep_prefix.safetensors"),
    })


def train(out, model, manifest, adapter, control, tokenizer, rows, gate, args, identity):
    import torch
    from full_gate import checksum_backbone
    from ptuning_v2_hybrid import DeepPrefix, METHOD
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    require({id(p) for g in optimizer.param_groups for p in g["params"]}
            == {id(p) for p in adapter.parameters()}, "Optimizer isolation failed")
    require(not ({id(p) for p in adapter.parameters()} &
                 {id(p) for p in model.parameters()}), "Backbone in optimizer")
    if args.resume:
        run = json.loads((out / "run.json").read_text())
        require(run["identity"] == identity and run["learning_rate"] == args.lr
                and run["epochs"] == args.epochs, "Resume identity/settings changed")
        require(not run["training_complete"], "Training is already complete")
        pointer = json.loads((out / "latest_checkpoint.json").read_text())
        path = out / pointer["directory"]
        require(re.fullmatch(rf"checkpoint-{pointer['step']}-[0-9]+", path.name)
                and path.parent == out, "Invalid checkpoint pointer")
        require(sha(path / "training_state.pt") == pointer["training_state_sha256"]
                and sha(path / "deep_prefix.safetensors") == pointer["adapter_sha256"],
                "Checkpoint integrity failed")
        restored = DeepPrefix.load(path, adapter.spec, manifest["source_sha256"])
        adapter.load_state_dict(restored.state_dict())
        del restored
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        require(state["step"] == pointer["step"], "Checkpoint step mismatch")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        step = state["step"]
        del state
        # A crash after committing a checkpoint but before logging its event is
        # safe: sample position is determined by the committed optimizer step.
        emit(out, "resume", committed_step=step)
    else:
        run = dict(
            method=METHOD, identity=identity, training_complete=False, optimizer_steps=0,
            train_examples=len(rows["train"]), val_examples=len(rows["val"]),
            epochs=args.epochs, learning_rate=args.lr, prefix_length=adapter.length,
            trainable_parameters=sum(p.numel() for p in adapter.parameters()),
            full_attention_layers=adapter.spec["layer_types"].count("full_attention"),
            linear_attention_layers=adapter.spec["layer_types"].count("linear_attention"),
            backbone_frozen=True, no_truncation=True, full_evidence=True,
            answer_validation_enabled=False, activation_offload=True,
            initialized_from="old_INITIAL_shallow_adapter_layer_activation_calibration",
            initialized_from_final_trained_adapter=False,
            evaluation_kind="fixed_historical_development_evidence_not_blind_not_end_to_end",
            gguf_server_injection_validated=False, production_modified=False,
        )
        step = 0
        atomic_json(out / "run.json", run)
        save_checkpoint(out, 0, adapter, optimizer, manifest["source_sha256"])
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    start = time.monotonic()
    for epoch in range(args.epochs):
        order = list(range(len(rows["train"])))
        random.Random(20261012 + epoch).shuffle(order)
        for position, index in enumerate(order):
            absolute = epoch * len(order) + position
            if absolute < step:
                continue
            item = rows["train"][index]
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, stats = backward(model, adapter, control, item)
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0).item()
            require(norm > 0, "Zero overall gradient")
            optimizer.step()
            step += 1
            # Every step is recoverable. Retaining all snapshots also permits
            # audit of the committed prefix/optimizer pair after interruption.
            save_checkpoint(out, step, adapter, optimizer, manifest["source_sha256"])
            run.update(optimizer_steps=step, training_complete=False)
            atomic_json(out / "run.json", run)
            emit(out, "optimizer_step", step=step, epoch=epoch, id=item[0]["id"],
                 loss=loss.item(), prefix_gradient_norm=norm,
                 minimum_layer_gradient_norm=min(s["norm"] for s in stats.values()),
                 prompt_tokens=len(item[1]), target_tokens=len(item[2]),
                 evidence_count=20, truncated=False,
                 elapsed_seconds=time.monotonic() - tick,
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            del loss
    optimizer.zero_grad(set_to_none=True)
    require(checksum_backbone(model) == gate["backbone_sha256"], "Backbone changed")
    if (out / "final_adapter").exists():
        # Recovery after final save but before completion receipt.
        restored = DeepPrefix.load(out / "final_adapter", adapter.spec, manifest["source_sha256"])
        require(all(torch.equal(v, restored.state_dict()[k])
                    for k, v in adapter.state_dict().items()), "Existing final adapter differs")
        del restored
    else:
        adapter.save(out / "final_adapter", manifest["source_sha256"])
    run.update(training_complete=True, optimizer_steps=step,
               backbone_bytewise_unchanged=True,
               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
               training_seconds_this_invocation=time.monotonic() - start)
    atomic_json(out / "run.json", run)
    emit(out, "training_complete", optimizer_steps=step)


def evaluate(out, model, adapter, control, tokenizer, rows, max_new_tokens):
    import torch
    from ptuning_v2_hybrid import greedy_generate
    model.eval()
    model.gradient_checkpointing_disable()
    eos = {tokenizer.token_to_id(s) for s in ("<|im_end|>", "<|endoftext|>")}
    eos.discard(None)
    path = out / "generated_native46.jsonl"
    # Preflight ALL budgets before writing any output: never silently shorten.
    require(all(adapter.length + len(p) + max_new_tokens <= 32768
                for _, p, _ in rows["val"]), "Generation budget exceeds context limit")
    with path.open("x") as stream:
        for row, p, _ in rows["val"]:
            tick = time.monotonic()
            ids = torch.tensor([p], device="cuda")
            tokens = greedy_generate(model, adapter, control, ids, max_new_tokens, eos)
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            try:
                parsed = json.loads(text)
                valid_json = True
            except ValueError:
                parsed, valid_json = None, False
            result = dict(
                id=row["id"], question=row["question"], raw_text=text, token_ids=tokens,
                parsed=parsed, valid_json=valid_json,
                prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                evidence_count=20, prompt_tokens=len(p), prefix_tokens=adapter.length,
                truncated=False, output_limit_reached=len(tokens) == max_new_tokens
                and tokens[-1] not in eos,
                elapsed_seconds=time.monotonic() - tick,
            )
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            emit(out, "generation", id=row["id"], valid_json=valid_json,
                 generated_tokens=len(tokens), output_limit_reached=result["output_limit_reached"])
    atomic_json(out / "generation_receipt.json", dict(
        completed=True, count=len(rows["val"]), output_sha256=sha(path),
        no_input_truncation=True, answer_validation_enabled=False,
        semantic_accuracy_not_yet_reviewed=True,
        historical_development_evidence=True, blind_test=False, end_to_end=False,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT /
                        "targeted_pt_v2_20261014_attempt02/merged_data")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifact")
    parser.add_argument("--initial-shallow", type=Path, default=ROOT /
                        "targeted_pt_v2_20261014_attempt02/trained_v2/initial_adapter")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(args.epochs >= 1 and args.lr > 0, "Invalid training settings")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    with (out / "workflow.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment(out)
        status = dict(stage="preflight", pid=os.getpid(), started_unix=time.time(),
                      production_modified=False)
        atomic_json(out / "workflow.json", status)
        try:
            import torch
            from backend import load_backend
            from ptuning_v2_hybrid import DeepPrefix, PrefixController, architecture
            torch.manual_seed(20261012)
            require(torch.cuda.device_count() == 1, "Expose exactly one selected GPU")
            free, total = torch.cuda.mem_get_info()
            require(free >= 19500 * 1024**2, f"Need 19500 MiB free; got {free / 1024**2:.0f}")
            # Reserve at least 1.5 GiB for existing jobs; never consume all free memory.
            fraction = min(0.90, (free - 1536 * 1024**2) / total)
            torch.cuda.set_per_process_memory_fraction(fraction)
            emit(out, "resource_preflight", free_gpu_bytes=free,
                 process_memory_cap_bytes=int(fraction * total),
                 gpu_name=torch.cuda.get_device_name())
            data_manifest, tokenizer, rows = load_data(args.data, args.length)
            identity = dict(
                code_sha256={name: sha(ROOT / name) for name in CODE_FILES},
                data_sha256=data_manifest["data_sha256"], prefix_length=args.length,
                initial_adapter_sha256=sha(args.initial_shallow / "prompt_encoder.safetensors"),
                initial_adapter_config_sha256=sha(args.initial_shallow / "adapter_config.json"),
                tokenizer_sha256=sha(ROOT / "official/tokenizer.json"),
                artifact_manifest_sha256=sha(args.artifact / "manifest.json"),
            )
            emit(out, "data_audit_passed", train=len(rows["train"]), val=len(rows["val"]),
                 data_sha256=identity["data_sha256"])
            model, manifest = load_backend(args.artifact, dtype=torch.bfloat16)
            adapter = DeepPrefix(architecture(model.config), args.length).cuda()
            control = PrefixController(model, adapter)
            if args.resume:
                gate = json.loads((out / "gate.json").read_text())
                require(gate["completed"] and gate["identity"] == identity,
                        "Resume gate incomplete or changed")
                from full_gate import checksum_backbone
                require(checksum_backbone(model) == gate["backbone_sha256"],
                        "Backbone changed since gate")
            else:
                status["stage"] = "gate"
                atomic_json(out / "workflow.json", status)
                gate = run_gate(out, model, manifest, adapter, control, tokenizer,
                                rows, args.initial_shallow, identity)
            if not args.gate_only:
                status["stage"] = "training"
                atomic_json(out / "workflow.json", status)
                train(out, model, manifest, adapter, control, tokenizer, rows, gate, args, identity)
                status["stage"] = "generation"
                atomic_json(out / "workflow.json", status)
                evaluate(out, model, adapter, control, tokenizer, rows, args.max_new_tokens)
            control.close()
            status.update(stage="complete", finished_unix=time.time())
            atomic_json(out / "workflow.json", status)
        except BaseException as exc:
            status.update(stage="failed", failed_stage=status["stage"],
                          exception_type=type(exc).__name__, error=str(exc),
                          failed_unix=time.time())
            atomic_json(out / "workflow.json", status)
            raise


if __name__ == "__main__":
    main()
