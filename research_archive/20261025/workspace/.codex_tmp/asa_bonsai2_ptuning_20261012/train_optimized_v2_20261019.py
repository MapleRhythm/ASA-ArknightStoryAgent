"""Isolated optimizer experiment; unchanged deep injection/backbone/data/code.

Peak LR 1e-4, 5% linear warmup, cosine decay to 20%, no weight decay.
This is a preregistered experimental setting, not a demonstrated optimum.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import run_ptuning_v2_hybrid as deep
import train_curated_deep_20261019 as curated

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "ptuning_input_curated_20261019_attempt01"
SETTINGS = dict(
    optimizer="AdamW", peak_learning_rate=1e-4, betas=[0.9, 0.999], eps=1e-8,
    weight_decay=0.0, warmup_fraction=0.05, final_lr_fraction=0.2,
    scheduler="linear_warmup_cosine", gradient_clip_norm=1.0,
    epochs=1, seed=20261012,
    experimental_not_proven_optimal=True,
    changed_factors=["learning_rate", "learning_rate_schedule", "weight_decay"],
    single_factor_causal_attribution=False,
)


def learning_rate(step, total):
    """One-indexed step; reaches peak after warmup and floor on final step."""
    if not 1 <= step <= total:
        raise ValueError("Invalid optimization step")
    warmup = max(1, math.ceil(total * SETTINGS["warmup_fraction"]))
    if step <= warmup:
        return SETTINGS["peak_learning_rate"] * step / warmup
    fraction = (step - warmup) / max(1, total - warmup)
    decay = SETTINGS["final_lr_fraction"] + (1 - SETTINGS["final_lr_fraction"]) * (
        1 + math.cos(math.pi * fraction)) / 2
    return SETTINGS["peak_learning_rate"] * decay


def optimizer_for(adapter):
    import torch
    return torch.optim.AdamW(
        adapter.parameters(), lr=SETTINGS["peak_learning_rate"],
        betas=tuple(SETTINGS["betas"]), eps=SETTINGS["eps"], weight_decay=0.0)


def optimized_gate(out, model, manifest, adapter, control, tokenizer, rows, initial_path, identity):
    """Keep the original real-backbone gate, then test actual experiment LR."""
    import torch
    from backend import PromptEncoder
    from ptuning_v2_hybrid import DeepPrefix

    gate = deep.run_gate_original(
        out, model, manifest, adapter, control, tokenizer, rows, initial_path, identity)
    # Test whether the completed INPUT-ONLY baseline can be represented without
    # changing the deep adapter's fixed input seed. Never use it to initialize training.
    initial = DeepPrefix.load(out / "initial_adapter", adapter.spec, manifest["source_sha256"])
    model.eval()
    model.gradient_checkpointing_disable()
    item = rows["val"][0]
    ids = torch.tensor([item[1]], device="cuda")
    reference_shallow = PromptEncoder.load(BASELINE / "trained/final_adapter")
    config = json.loads((BASELINE / "trained/final_adapter/adapter_config.json").read_text())
    deep.require(config["source_sha256"] == manifest["source_sha256"], "Baseline backbone mismatch")
    with torch.no_grad():
        seed = reference_shallow().detach()
        control.calibrate(seed, ids)
        adapter.input_seed.copy_(initial.input_seed)
        with control.activate(prefill=False):
            reference = model(inputs_embeds=reference_shallow.inject(
                model.model.embed_tokens(ids)), use_cache=False, logits_to_keep=1).logits
        with control.activate():
            got = model(inputs_embeds=adapter.prepare(model.model.embed_tokens(ids)),
                        use_cache=False, logits_to_keep=1).logits
        difference = (reference.float() - got.float()).abs()
        gate["trained_input_baseline_mapping"] = dict(
            id=item[0]["id"], prompt_tokens=len(item[1]), evidence_count=20,
            fixed_initial_input_seed=True, initialized_training_from_this_mapping=False,
            baseline_adapter_sha256=deep.sha(BASELINE / "trained/final_adapter/prompt_encoder.safetensors"),
            max_abs=difference.max().item(), mean_abs=difference.mean().item(),
            top1_equal=bool(reference[0, -1].argmax() == got[0, -1].argmax()))
        # Restore irrespective of whether the diagnostic agrees. A disagreement
        # is recorded, not silently "fixed" by changing training initialization.
        adapter.load_state_dict(initial.state_dict())
    del reference_shallow, reference, got, seed, difference
    deep.atomic_json(out / "gate.json", gate)
    deep.emit(out, "trained_input_mapping_diagnostic", **gate["trained_input_baseline_mapping"])
    deep.require(gate["trained_input_baseline_mapping"]["max_abs"] <= 0.01,
                 "Real-input shallow-to-deep mapping disagreed; investigate before training")

    # Probe the actual peak LR on the longest TRAIN example, never val labels.
    longest = max(rows["train"], key=lambda item: len(item[1]) + len(item[2]))
    optimizer = optimizer_for(adapter)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer.zero_grad(set_to_none=True)
    before = {k: p.detach().clone() for k, p in adapter.prefix.items()}
    loss, stats = deep.backward(model, adapter, control, longest)
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), SETTINGS["gradient_clip_norm"]).item()
    optimizer.step()
    deep.require(all(torch.isfinite(p).all() for p in adapter.parameters()),
                 "Nonfinite adapter after peak-LR gate")
    change = {}
    for branch in ("gdn", "k", "v"):
        names = [k for k in adapter.prefix if k.endswith("_" + branch)]
        total = sum(adapter.prefix[k].numel() for k in names)
        change[branch] = dict(
            elements=total,
            bf16_changed_fraction=sum(
                int((adapter.prefix[k].detach().bfloat16() != before[k].bfloat16()).sum())
                for k in names) / total)
    gate["actual_optimizer_gate"] = dict(
        settings=SETTINGS, id=longest[0]["id"], train_labels_only=True,
        loss_before_update=loss.item(), gradient_norm=norm, bf16_change=change,
        all_parameters_finite=True, reset_to_original_initial_state=True)
    del optimizer, before, loss
    adapter.zero_grad(set_to_none=True)
    adapter.load_state_dict(initial.state_dict())
    deep.require(all(torch.equal(value, initial.state_dict()[key])
                     for key, value in adapter.state_dict().items()),
                 "Peak-LR gate changes leaked into training")
    del initial
    deep.atomic_json(out / "gate.json", gate)
    deep.emit(out, "actual_optimizer_gate_passed", **gate["actual_optimizer_gate"])
    return gate


def train(out, model, manifest, adapter, control, tokenizer, rows, gate, args, identity):
    import torch
    from full_gate import checksum_backbone
    from ptuning_v2_hybrid import DeepPrefix, METHOD

    deep.require(args.lr == SETTINGS["peak_learning_rate"] and args.epochs == SETTINGS["epochs"],
                 "CLI settings differ from registered optimizer experiment")
    optimizer = optimizer_for(adapter)
    parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    deep.require(parameter_ids == {id(p) for p in adapter.parameters()}, "Wrong optimizer parameters")
    deep.require(not parameter_ids & {id(p) for p in model.parameters()}, "Backbone in optimizer")
    total = len(rows["train"]) * args.epochs
    baseline = json.loads((BASELINE / "trained/run.json").read_text())
    deep.require(baseline["training_complete"] and baseline["backbone_bytewise_unchanged"],
                 "Input-only reference training is incomplete")
    deep.require(baseline["data_sha256"] == identity["data_sha256"], "Paired dataset mismatch")
    deep.require(baseline["optimizer_steps"] == total, "Paired training steps mismatch")
    if args.resume:
        run = json.loads((out / "run.json").read_text())
        deep.require(run["identity"] == identity and run["optimizer_settings"] == SETTINGS
                     and not run["training_complete"], "Unsafe resume")
        pointer = json.loads((out / "latest_checkpoint.json").read_text())
        import re
        path = out / pointer["directory"]
        deep.require(re.fullmatch(rf"checkpoint-{pointer['step']}-[0-9]+", path.name)
                     and path.parent == out, "Invalid checkpoint target")
        deep.require(deep.sha(path / "training_state.pt") == pointer["training_state_sha256"]
                     and deep.sha(path / "deep_prefix.safetensors") == pointer["adapter_sha256"],
                     "Corrupt checkpoint")
        restored = DeepPrefix.load(path, adapter.spec, manifest["source_sha256"])
        adapter.load_state_dict(restored.state_dict())
        del restored
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        deep.require(state["step"] == pointer["step"], "Checkpoint step mismatch")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        step = state["step"]
        del state
        deep.emit(out, "resume", committed_step=step)
    else:
        step = 0
        run = dict(
            method=METHOD, identity=identity, optimizer_settings=SETTINGS,
            training_complete=False, optimizer_steps=0,
            train_examples=len(rows["train"]), val_examples=len(rows["val"]),
            epochs=args.epochs, learning_rate=args.lr, prefix_length=adapter.length,
            trainable_parameters=sum(p.numel() for p in adapter.parameters()),
            full_attention_layers=adapter.spec["layer_types"].count("full_attention"),
            linear_attention_layers=adapter.spec["layer_types"].count("linear_attention"),
            backbone_frozen=True, no_truncation=True, full_evidence=True,
            answer_validation_enabled=False, activation_offload=True,
            initialized_from="same_old_INITIAL_shallow_calibration_as_historical_deep",
            initialized_from_final_trained_adapter=False, input_baseline=str(BASELINE),
            input_baseline_run_sha256=deep.sha(BASELINE / "trained/run.json"),
            evaluation_kind="historical46_paired_full_evidence_not_blind_not_end_to_end",
            production_modified=False, gguf_server_injection_validated=False,
            experiment="optimizer_bundle_not_single_factor_ablation")
        deep.atomic_json(out / "run.json", run)
        deep.save_checkpoint(out, 0, adapter, optimizer, manifest["source_sha256"])
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    start = time.monotonic()
    for epoch in range(args.epochs):
        order = list(range(len(rows["train"])))
        random.Random(20261012 + epoch).shuffle(order)
        for position, index in enumerate(order):
            if epoch * len(order) + position < step:
                continue
            item = rows["train"][index]
            tick = time.monotonic()
            lr = learning_rate(step + 1, total)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, stats = deep.backward(model, adapter, control, item)
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0).item()
            deep.require(norm > 0 and math.isfinite(norm), "Invalid overall gradient")
            optimizer.step()
            deep.require(all(torch.isfinite(p).all() for p in adapter.parameters()),
                         "Nonfinite adapter after optimization")
            step += 1
            deep.save_checkpoint(out, step, adapter, optimizer, manifest["source_sha256"])
            run.update(optimizer_steps=step)
            deep.atomic_json(out / "run.json", run)
            deep.emit(out, "optimizer_step", step=step, epoch=epoch, id=item[0]["id"],
                      loss=loss.item(), prefix_gradient_norm=norm, learning_rate=lr,
                      minimum_layer_gradient_norm=min(s["norm"] for s in stats.values()),
                      prompt_tokens=len(item[1]), target_tokens=len(item[2]),
                      evidence_count=20, truncated=False,
                      elapsed_seconds=time.monotonic() - tick,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            del loss
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    deep.require(checksum_backbone(model) == gate["backbone_sha256"], "Backbone changed")
    if (out / "final_adapter").exists():
        restored = DeepPrefix.load(out / "final_adapter", adapter.spec, manifest["source_sha256"])
        deep.require(all(torch.equal(v, restored.state_dict()[k])
                         for k, v in adapter.state_dict().items()), "Existing final adapter differs")
        del restored
    else:
        adapter.save(out / "final_adapter", manifest["source_sha256"])
    model.eval()
    model.gradient_checkpointing_disable()
    validation = []
    for item in rows["val"]:
        with torch.no_grad(), control.activate():
            nll = deep.loss_for(model, adapter, item).item()
        deep.require(math.isfinite(nll), "Invalid validation loss")
        entry = dict(id=item[0]["id"], nll=nll)
        validation.append(entry)
        deep.emit(out, "validation", **entry)
    run.update(training_complete=True, optimizer_steps=step, backbone_bytewise_unchanged=True,
               final_validation=validation, validation_is_not_QA_accuracy=True,
               final_mean_validation_nll=sum(x["nll"] for x in validation) / len(validation),
               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
               training_seconds_this_invocation=time.monotonic() - start)
    deep.atomic_json(out / "run.json", run)
    deep.emit(out, "training_complete", optimizer_steps=step)


def paired_report(out):
    """Integrity/statistics only. Semantic judgments remain explicitly pending."""
    own_run = json.loads((out / "run.json").read_text())
    base_run = json.loads((BASELINE / "trained/run.json").read_text())
    own_receipt = json.loads((out / "generation_receipt.json").read_text())
    base_receipt = json.loads((BASELINE / "generation_receipt.json").read_text())
    deep.require(own_run["training_complete"] and base_run["training_complete"],
                 "Both final models must be trained")
    deep.require(own_run["identity"]["data_sha256"] == base_run["data_sha256"],
                 "Data identity mismatch")
    deep.require(own_receipt["completed"] and base_receipt["completed"], "Generation incomplete")
    own_path, base_path = out / "generated_native46.jsonl", BASELINE / "generated_native46.jsonl"
    deep.require(deep.sha(own_path) == own_receipt["output_sha256"]
                 and deep.sha(base_path) == base_receipt["output_sha256"], "Generation SHA mismatch")
    own, base = curated.read_rows(own_path), curated.read_rows(base_path)
    deep.require(len(own) == len(base) == 46, "Require exactly 46 final outputs")
    pairs = []
    for position, (a, b) in enumerate(zip(own, base), 1):
        deep.require(a["id"] == b["id"] and a["question"] == b["question"]
                     and a["prompt_sha256"] == b["prompt_sha256"], "Paired prompt mismatch")
        deep.require(a["evidence_count"] == b["evidence_count"] == 20
                     and not a["truncated"] and not b["truncated"], "Evidence mismatch")
        pairs.append(dict(position=position, id=a["id"], question=a["question"],
                          prompt_sha256=a["prompt_sha256"],
                          optimized_v2_raw=a["raw_text"], input_only_raw=b["raw_text"],
                          semantic_review_pending=True))
    with (out / "paired_final46_for_content_review.jsonl").open("x") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    baseline_nll = sum(r["nll"] for r in base_run["final_validation"]) / 46
    deep.require([r["id"] for r in own_run["final_validation"]]
                 == [r["id"] for r in base_run["final_validation"]], "NLL task order mismatch")
    deep.atomic_json(out / "PAIRED_FINAL46_STATISTICS.json", dict(
        count=46, same_train_and_val=True, same_training_order_and_steps=True,
        input_baseline=str(BASELINE), optimized_v2=str(out),
        paired_outputs_sha256=deep.sha(out / "paired_final46_for_content_review.jsonl"),
        generation_sha256={"input_only": deep.sha(base_path), "optimized_v2": deep.sha(own_path)},
        mean_teacher_forced_nll={
            "input_only": baseline_nll, "optimized_v2": own_run["final_mean_validation_nll"]},
        generated_json_valid={
            "input_only": sum(r["valid_json"] for r in base),
            "optimized_v2": sum(r["valid_json"] for r in own)},
        semantic_accuracy_not_yet_reviewed=True, correctness_rate=None,
        structural_and_nll_metrics_are_not_QA_accuracy=True,
        blind_test=False, end_to_end=False, production_modified=False))


def launch(data, out, gpu):
    curated.verify(data)
    deep.require(not out.exists(), "Output already exists")
    baseline = json.loads((BASELINE / "trained/run.json").read_text())
    deep.require(baseline["training_complete"], "Input baseline not trained")
    deep.require(baseline["data_sha256"] == {
        s: deep.sha(data / f"{s}.jsonl") for s in ("train", "val")}, "Baseline data mismatch")
    info = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid, free = info[0].strip(), int(info[1])
    deep.require(free >= 22000, "Insufficient free GPU memory")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "workflow",
               "--data", str(data), "--out", str(out), "--lr", "0.0001", "--epochs", "1"]
    receipt = dict(command=command, gpu=gpu, gpu_uuid=uuid, settings=SETTINGS,
                   free_gpu_mib=free, production_modified=False, other_jobs_stopped=False)
    with out.with_suffix(".launch.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS="8",
               TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True")
    with out.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt.update(pid=process.pid, launched_unix=time.time())
    deep.atomic_json(out.with_suffix(".launch.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "workflow":
        sys.argv.pop(1)
        deep.load_data = curated.load_data
        deep.CODE_FILES += ("train_curated_deep_20261019.py", Path(__file__).name)
        deep.run_gate_original = deep.run_gate
        deep.run_gate = optimized_gate
        deep.train = train
        out_index = sys.argv.index("--out") + 1
        out = Path(sys.argv[out_index]).resolve()
        deep.main()
        try:
            paired_report(out)
        except FileNotFoundError as exc:
            deep.atomic_json(out / "paired_report_pending.json", dict(
                reason=str(exc), input_baseline=str(BASELINE), semantic_review_pending=True))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "paired-report"))
    parser.add_argument("--data", type=Path, default=ROOT / "curated_shallow_data_20261019_v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.command == "launch":
        launch(args.data.resolve(), args.out.resolve(), args.gpu)
    else:
        paired_report(args.out.resolve())


if __name__ == "__main__":
    main()
