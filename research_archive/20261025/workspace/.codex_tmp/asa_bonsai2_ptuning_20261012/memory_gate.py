"""Actual longest full-context forward/backward, isolated from task training."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from backend import PromptEncoder, load_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--compare-short", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    root = Path(__file__).resolve().parent
    tokenizer = Tokenizer.from_file(str(root / "official/tokenizer.json"))
    examples = []
    for split in ("train", "val"):
        for line in (Path(args.data) / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert len(row["documents"]) == 20 and not row["truncated"]
            assert all(d["text"] in row["prompt"] for d in row["documents"])
            p = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
            t = tokenizer.encode(row["target"] + "<|im_end|>", add_special_tokens=False).ids
            examples.append((row, p, t))
    row, p, t = max(examples, key=lambda x: len(x[1]) + len(x[2]))
    assert len(p) + len(t) + 16 <= 32768
    torch.manual_seed(42)
    start = time.monotonic()
    model, manifest = load_backend("artifact", dtype=torch.bfloat16)
    encoder = PromptEncoder(16, model.config.hidden_size).cuda()
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    receipt = {"kind": "memory_gradient_diagnostic_not_QA_accuracy",
               "source_sha256": manifest["source_sha256"], "id": row["id"],
               "prompt_tokens": len(p), "target_tokens": len(t),
               "evidence_count": 20, "truncated": False,
               "source_data_integrity_not_independently_certified": True,
               "activation_offload": args.offload, "completed": False}
    receipt["data_sha256"] = {s: hashlib.sha256((Path(args.data) / f"{s}.jsonl").read_bytes()).hexdigest()
                             for s in ("train", "val")}
    receipt["backend_sha256"] = hashlib.sha256((root / "backend.py").read_bytes()).hexdigest()
    out.write_text(json.dumps(receipt, indent=2))

    def run(prompt_ids, target_ids, offload):
        encoder.zero_grad(set_to_none=True)
        ids = torch.tensor([prompt_ids + target_ids], device="cuda")
        embedded = model.model.embed_tokens(ids).detach()
        positions = torch.arange(16 + len(prompt_ids) - 1,
                                 16 + len(prompt_ids) + len(target_ids) - 1, device="cuda")
        context = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
        with context:
            logits = model(inputs_embeds=encoder.inject(embedded), use_cache=False,
                           logits_to_keep=positions).logits[0]
            loss = F.cross_entropy(logits.float(), torch.tensor(target_ids, device="cuda"))
            loss.backward()
        assert torch.isfinite(loss)
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in encoder.parameters())
        assert all(not x.requires_grad and x.grad is None for x in model.parameters())
        grads = torch.cat([x.grad.detach().flatten().cpu() for x in encoder.parameters()])
        assert grads.norm().item() > 0
        return loss.item(), grads

    if args.compare_short:
        ids = tokenizer.encode("只根据证据回答：一加一等于几？", add_special_tokens=False).ids
        target = tokenizer.encode("二。", add_special_tokens=False).ids
        a_loss, a_grad = run(ids, target, False)
        repeat_loss, repeat_grad = run(ids, target, False)
        b_loss, b_grad = run(ids, target, True)
        receipt["short_comparison"] = {
            "losses": [a_loss, repeat_loss, b_loss],
            "gpu_repeat_gradient_max_abs": (a_grad - repeat_grad).abs().max().item(),
            "offload_gradient_max_abs": (a_grad - b_grad).abs().max().item(),
            "gpu_repeat_gradient_relative_l2": ((a_grad - repeat_grad).norm() / a_grad.norm()).item(),
            "offload_gradient_relative_l2": ((a_grad - b_grad).norm() / a_grad.norm()).item(),
            "offload_gradients_exact": torch.equal(a_grad, b_grad),
        }
        out.write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt["short_comparison"]), flush=True)
        assert a_loss == repeat_loss == b_loss
        assert receipt["short_comparison"]["offload_gradient_relative_l2"] < 0.01
        assert torch.allclose(a_grad, b_grad, atol=1e-5, rtol=0.01)
        print("SHORT_OFFLOAD_GRADIENT_COMPARISON_PASSED", flush=True)
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loss, grads = run(p, t, args.offload)
    receipt.update(completed=True, loss=loss, prompt_gradient_norm=grads.norm().item(),
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   elapsed_seconds=time.monotonic() - start)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
