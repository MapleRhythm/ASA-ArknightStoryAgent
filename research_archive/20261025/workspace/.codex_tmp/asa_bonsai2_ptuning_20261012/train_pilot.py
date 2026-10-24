"""Train only the external P-tuning encoder, on complete evidence packs."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from backend import PromptEncoder, load_backend
from full_gate import checksum_backbone


def emit(path, payload):
    with path.open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot_data_v2")
    ap.add_argument("--artifact", default="artifact")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--gate", default="full_gate_bfloat16.json")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--validation-limit", type=int, default=0)
    ap.add_argument("--skip-initial-validation", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--memory-gate")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    gate = json.loads(Path(args.gate).read_text())
    assert gate["training_completed"] and gate["frozen_backbone_unchanged"]
    assert gate["reload_logits_max_abs"] == 0 and all(n > 0 for n in gate["grad_norms"])
    data_dir, out = Path(args.data), Path(args.out)
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    assert data_manifest.get("evidence_count_per_example") == 20
    assert data_manifest.get("truncated_documents") == 0
    assert data_manifest.get("train_val_story_overlap", []) == []
    assert not data_manifest.get("review_required_before_training", False)
    if data_manifest.get("kind", "").startswith("fresh_GLM"):
        assert data_manifest["complete"] and data_manifest["new_api_calls"] > 0
        assert not data_manifest["old_labels_used"]
        assert data_manifest["train_val_source_family_overlap"] == []
        assert data_manifest["train_val_document_overlap"] == []
    out.mkdir(exist_ok=args.resume)
    torch.manual_seed(20261012)
    # GPU1 currently has >20 GiB free; keep a material headroom guard while
    # allowing long 20-document prompts to complete backward.
    free_mib = int(torch.cuda.mem_get_info()[0] / (1024 * 1024))
    if free_mib < 20000:
        raise RuntimeError(f"insufficient safe headroom before training: {free_mib} MiB")
    torch.cuda.set_per_process_memory_fraction(0.97)
    tokenizer = Tokenizer.from_file(str(root / "official/tokenizer.json"))
    rows = {split: [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "val")}
    encoded = {}
    for split in rows:
        encoded[split] = []
        for row in rows[split]:
            assert row["evidence_count"] == 20 and len(row["documents"]) == 20 and not row["truncated"]
            assert all(d["text"] in row["prompt"] for d in row["documents"])
            p = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
            t = tokenizer.encode(row["target"] + "<|im_end|>", add_special_tokens=False).ids
            # Reject instead of truncating any sample.
            assert len(p) + len(t) + 16 <= 32768, (row["id"], len(p) + len(t))
            encoded[split].append((row, p, t))
    model, manifest = load_backend(args.artifact, dtype=torch.bfloat16)
    before_hash = checksum_backbone(model)
    encoder = PromptEncoder(16, model.config.hidden_size).cuda()
    init_ids = tokenizer.encode("只根据给定剧情证据输出事实，不复制引文，不使用模型记忆。",
                                add_special_tokens=False).ids
    init_ids = (init_ids * 16)[:16]
    with torch.no_grad():
        encoder.embedding.weight.copy_(model.model.embed_tokens(
            torch.tensor(init_ids, device="cuda")).float())
    if not args.resume:
        encoder.save(out / "initial_adapter", manifest["source_sha256"])
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimizer_ids == {id(p) for p in encoder.parameters()}
    assert not optimizer_ids & {id(p) for p in model.parameters()}
    run = {
        "method": "p_tuning_external_residual_mlp", "source_sha256": manifest["source_sha256"],
        "train_examples": len(rows["train"]), "val_examples": len(rows["val"]),
        "epochs": args.epochs, "learning_rate": args.lr,
        "trainable_parameters": sum(p.numel() for p in encoder.parameters()),
        "backbone_frozen": True, "full_evidence": True, "answer_validation_enabled": False,
        "no_truncation": True, "training_complete": False,
        "encoded_lengths": {s: [{"id": r["id"], "prompt": len(p), "target": len(t)}
                               for r, p, t in encoded[s]] for s in encoded},
        "data_sha256": {s: hashlib.sha256((data_dir / f"{s}.jsonl").read_bytes()).hexdigest()
                        for s in rows},
        "evaluation_kind": "heldout_activity_teacher_forced_nll_not_QA_accuracy",
        "historical_development_questions": True, "production_modified": False,
        "activation_offload": args.offload, "validation_limit": args.validation_limit,
        "backend_sha256": hashlib.sha256((root / "backend.py").read_bytes()).hexdigest(),
        "gate_path": str(args.gate),
    }
    if args.memory_gate:
        memory_gate = json.loads(Path(args.memory_gate).read_text())
        assert memory_gate["completed"]
        assert memory_gate["backend_sha256"] == run["backend_sha256"]
        assert memory_gate["data_sha256"] == run["data_sha256"]
        assert memory_gate["activation_offload"] == args.offload
        run["memory_gate"] = str(args.memory_gate)
    if args.resume:
        previous = json.loads((out / "run.json").read_text())
        for key in ("data_sha256", "source_sha256", "learning_rate", "backend_sha256",
                    "activation_offload", "epochs"):
            assert previous[key] == run[key], f"resume mismatch: {key}"
        run = previous
        state = torch.load(out / "training_state.pt", map_location="cuda", weights_only=True)
        restored = PromptEncoder.load(out / f"checkpoint-{state['step']}")
        encoder.load_state_dict(restored.state_dict())
        del restored
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        resumed_step = state["step"]
    else:
        resumed_step = 0
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))

    def loss_for(item, prompt_encoder):
        row, p, t = item
        ids = torch.tensor([p + t], device="cuda")
        embedded = model.model.embed_tokens(ids).detach()
        offset = 0
        if prompt_encoder is not None:
            embedded = prompt_encoder.inject(embedded)
            offset = prompt_encoder.length
        positions = torch.arange(offset + len(p) - 1, offset + len(p) + len(t) - 1, device="cuda")
        logits = model(inputs_embeds=embedded, use_cache=False, logits_to_keep=positions).logits[0]
        return F.cross_entropy(logits.float(), torch.tensor(t, device="cuda"))

    def evaluate(stage, prompt_encoder):
        model.eval()
        values = []
        with torch.no_grad():
            selected = encoded["val"][:args.validation_limit] if args.validation_limit else encoded["val"]
            for item in selected:
                start = time.monotonic()
                loss = loss_for(item, prompt_encoder).item()
                assert torch.isfinite(torch.tensor(loss))
                entry = {"event": "validation", "stage": stage, "id": item[0]["id"],
                         "question": item[0]["question"], "nll": loss,
                         "elapsed_seconds": time.monotonic() - start}
                emit(out / "progress.jsonl", entry)
                values.append(entry)
        return values

    if not args.resume:
        if not args.skip_initial_validation:
            run["base_validation"] = evaluate("frozen_no_prompt", None)
            run["initial_validation"] = evaluate("initial_prompt", encoder)
        else:
            run["initial_validation_skipped"] = True
        (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    step, start_all = resumed_step, time.monotonic()
    for epoch in range(args.epochs):
        order = list(range(len(encoded["train"])))
        random.Random(20261012 + epoch).shuffle(order)
        for order_position, index in enumerate(order):
            if epoch * len(order) + order_position < resumed_step:
                continue
            model.train()
            item = encoded["train"][index]
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if args.offload else nullcontext()
            with offload:
                loss = loss_for(item, encoder)
                assert torch.isfinite(loss)
                loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0).item()
            assert norm > 0
            optimizer.step()
            step += 1
            emit(out / "progress.jsonl", {
                "event": "optimizer_step", "step": step, "epoch": epoch, "id": item[0]["id"],
                "loss": loss.item(), "prompt_grad_norm": norm, "evidence_count": 20,
                "prompt_tokens": len(item[1]), "target_tokens": len(item[2]),
                "truncated": False, "elapsed_seconds": time.monotonic() - start,
                "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            })
            encoder.save(out / f"checkpoint-{step}", manifest["source_sha256"])
            state_path = out / "training_state.pt"
            torch.save({"step": step, "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state()}, out / "training_state.pt.tmp")
            os.replace(out / "training_state.pt.tmp", state_path)
    run["final_validation"] = evaluate("trained_prompt", encoder)
    assert checksum_backbone(model) == before_hash
    encoder.save(out / "final_adapter", manifest["source_sha256"])
    restored = PromptEncoder.load(out / "final_adapter")
    assert torch.equal(restored().detach(), encoder().detach())
    run.update(training_complete=True, optimizer_steps=step,
               backbone_parameters_and_buffers_sha256=before_hash,
               backbone_bytewise_unchanged=True, adapter_reload_embeddings_exact=True,
               train_elapsed_seconds=time.monotonic() - start_all,
               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
    print(json.dumps(run, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
