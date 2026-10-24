"""Full 27B fidelity, external encoder gradient, and save/reload gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from backend import PromptEncoder, load_backend


def checksum_backbone(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        assert not param.requires_grad and param.grad is None
        h.update(name.encode())
        h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    for name, value in model.named_buffers():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifact")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--fidelity-only", action="store_true")
    ap.add_argument("--out", help="New gate receipt path; leave historical receipts untouched")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.out) if args.out else root / f"full_gate_{args.dtype}.json"
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    t0 = time.monotonic()
    print("LOADING_FROZEN_BACKBONE", flush=True)
    model, manifest = load_backend(args.artifact, dtype=dtype)
    print(f"BACKBONE_LOADED {time.monotonic() - t0:.2f}s", flush=True)
    results = {"dtype": args.dtype, "source_sha256": manifest["source_sha256"],
               "fidelity": [], "training_completed": False}
    reference = json.loads((root / "reference.json").read_text())
    for row in reference:
        ids = torch.tensor([row["tokens"]], device="cuda")
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        probs = logits.softmax(-1)
        values, indices = probs.topk(20)
        entry = {"prompt": row["prompt"], "top20": [
            {"id": int(i), "p": float(v)} for i, v in zip(indices, values)],
            "native": row["native"], "finite": bool(torch.isfinite(logits).all()),
            "elapsed_seconds": time.monotonic() - t0}
        results["fidelity"].append(entry)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    if args.fidelity_only:
        return
    # Enforce native next-token equality and close native top20 probabilities.
    for entry in results["fidelity"]:
        native = entry["native"]
        nt = native.get("tokens", [])
        if len(nt) != 1 or entry["top20"][0]["id"] != nt[0]:
            raise RuntimeError("Fidelity gate failed: native top token mismatch")
        distribution = native["completion_probabilities"][0]["top_logprobs"]
        ref_probs = {int(p["id"]): math.exp(float(p["logprob"])) for p in distribution}
        got = {p["id"]: p["p"] for p in entry["top20"]}
        error = max(abs(got.get(i, 0) - p) for i, p in ref_probs.items())
        entry["native_top20_max_probability_error"] = error
        if error > 0.035:
            raise RuntimeError(f"Fidelity gate failed: probability error {error}")
    print("FIDELITY_GATE_PASSED", flush=True)
    before_hash = checksum_backbone(model)
    encoder = PromptEncoder(16, model.config.hidden_size).cuda()
    # Synthetic arithmetic is a backend smoke, NOT storyline training/evaluation.
    tokenizer = Tokenizer.from_file(str(root / "official/tokenizer.json"))
    prefix = "<|im_start|>user\n一加一等于几？<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    target_ids = tokenizer.encode("二。<|im_end|>", add_special_tokens=False).ids
    ids = torch.tensor([input_ids + target_ids], device="cuda")
    embedded = model.model.embed_tokens(ids).detach()
    positions = torch.arange(encoder.length + len(input_ids) - 1,
                             encoder.length + len(input_ids) + len(target_ids) - 1, device="cuda")
    target = torch.tensor(target_ids, device="cuda")
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(encoder.parameters(), lr=1e-4)
    trainable = sum(p.numel() for p in encoder.parameters())
    losses, norms = [], []
    init_prompt = encoder().detach().clone()
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        x = encoder.inject(embedded)
        logits = model(inputs_embeds=x, use_cache=False, logits_to_keep=positions).logits[0].float()
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        grad_stats = {name: {
            "present": p.grad is not None,
            "finite": bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
            "max_abs": p.grad.abs().max().item() if p.grad is not None else None,
        } for name, p in encoder.named_parameters()}
        print(json.dumps({"step": step, "gradient_stats": grad_stats}), flush=True)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
        norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0).item()
        assert norm > 0 and all(p.grad is None for p in model.parameters())
        opt.step()
        losses.append(loss.item())
        norms.append(norm)
        print(json.dumps({"step": step, "loss": loss.item(), "prompt_grad_norm": norm}), flush=True)
    delta = (encoder().detach() - init_prompt).abs().max().item()
    assert delta > 0 and checksum_backbone(model) == before_hash
    adapter_path = output.parent / f"{output.stem}_adapter"
    encoder.save(adapter_path, manifest["source_sha256"])
    restored = PromptEncoder.load(adapter_path)
    assert torch.equal(encoder().detach(), restored().detach())
    model.eval()
    with torch.no_grad():
        l1 = model(inputs_embeds=encoder.inject(embedded), use_cache=False, logits_to_keep=positions).logits
        l2 = model(inputs_embeds=restored.inject(embedded), use_cache=False, logits_to_keep=positions).logits
    reload_diff = (l1 - l2).abs().max().item()
    assert reload_diff == 0
    results.update(training_completed=True, smoke_only=True, trainable_parameters=trainable,
                   frozen_backbone_unchanged=True, losses=losses, grad_norms=norms,
                   prompt_max_abs_update=delta, reload_logits_max_abs=reload_diff,
                   elapsed_seconds=time.monotonic() - t0,
                   peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
