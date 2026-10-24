"""Differentiable input path for the *same* frozen Prism ternary checkpoint.

Packed matrices stay frozen. Dense BF16/FP16 matrices exist only during each
linear operation and are reconstructed in backward, not kept as activations.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
import triton
import triton.language as tl


@triton.jit
def _unpack_kernel(P, S, B, Y, N: tl.constexpr, WIDTH: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = i < N
    word = tl.load(P + i // 16, valid, other=0).to(tl.uint32)
    code = (word >> ((i % 16) * 2)) & 3
    scale = tl.load(S + i // 128, valid, other=0).to(tl.float32)
    bias = tl.load(B + i // 128, valid, other=0).to(tl.float32)
    tl.store(Y + i, code.to(tl.float32) * scale + bias, valid)


def dequantize(packed, scales, biases, dtype):
    rows, words = packed.shape
    width = words * 16
    out = torch.empty((rows, width), dtype=dtype, device=packed.device)
    _unpack_kernel[(triton.cdiv(rows * width, 1024),)](
        packed, scales, biases, out, rows * width, width, 1024)
    return out


def _fwht_bounded(x, block, signs, inverse, chunk_rows):
    shape, dtype = x.shape, x.dtype
    if block <= 0 or block & (block - 1) or shape[-1] % block:
        raise ValueError("Hadamard block must divide width")
    if chunk_rows <= 0 or signs.shape != (shape[-1],):
        raise ValueError("Invalid transform chunk/sign width")
    flat = x.reshape(-1, shape[-1])
    out = torch.empty_like(flat)
    for start in range(0, flat.shape[0], chunk_rows):
        # Own this storage, even for FP32 input: never mutate the caller.
        z = flat[start:start + chunk_rows].float().clone()
        if not inverse:
            z.mul_(signs)
        z = z.reshape(-1, block)
        step = 1
        while step < block:
            y = z.reshape(z.shape[0], -1, 2, step)
            left, right = y.unbind(-2)
            z = torch.stack((left + right, left - right), dim=-2).reshape(z.shape)
            step *= 2
        z = (z / math.sqrt(block)).reshape(-1, shape[-1])
        if inverse:
            z.mul_(signs)
        out[start:start + chunk_rows].copy_(z.to(dtype))
    return out.reshape(shape)


class FrozenHadamardFunction(torch.autograd.Function):
    """Linear signed orthogonal transform; its transpose needs no activations."""
    @staticmethod
    def forward(ctx, x, block, signs, inverse, chunk_rows):
        if signs.requires_grad:
            raise ValueError("Hadamard signs must be frozen")
        ctx.save_for_backward(signs)
        ctx.block, ctx.inverse, ctx.chunk_rows = block, inverse, chunk_rows
        return _fwht_bounded(x, block, signs, inverse, chunk_rows)

    @staticmethod
    def backward(ctx, grad):
        (signs,) = ctx.saved_tensors
        # Forward H D has transpose D H; inverse D H has transpose H D.
        return (_fwht_bounded(grad, ctx.block, signs, not ctx.inverse, ctx.chunk_rows),
                None, None, None, None)


def fwht(x, block, signs, inverse=False, chunk_rows=512):
    return FrozenHadamardFunction.apply(x, block, signs, inverse, chunk_rows)


class FrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scales, biases, chunk):
        if any(t.requires_grad for t in (packed, scales, biases)):
            raise ValueError("Packed backbone buffers must be frozen")
        # Immutable backbone buffers already live on GPU. Do not feed them to
        # saved-activation CPU offload hooks (they are not activations).
        ctx.frozen_buffers = (packed, scales, biases)
        ctx.input_shape, ctx.dtype, ctx.chunk = x.shape, x.dtype, chunk
        rows = packed.shape[0]
        flat = x.reshape(-1, x.shape[-1])
        out = torch.empty((flat.shape[0], rows), device=x.device, dtype=x.dtype)
        for lo in range(0, rows, chunk):
            hi = min(lo + chunk, rows)
            w = dequantize(packed[lo:hi], scales[lo:hi], biases[lo:hi], x.dtype)
            out[:, lo:hi] = F.linear(flat, w)
        return out.reshape(*x.shape[:-1], rows)

    @staticmethod
    def backward(ctx, grad):
        packed, scales, biases = ctx.frozen_buffers
        rows = packed.shape[0]
        flat = grad.reshape(-1, rows)
        out = torch.empty((flat.shape[0], packed.shape[1] * 16),
                          device=grad.device, dtype=ctx.dtype)
        for start in range(0, flat.shape[0], 2048):
            end = min(start + 2048, flat.shape[0])
            acc = torch.zeros((end - start, out.shape[1]),
                              device=grad.device, dtype=torch.float32)
            for lo in range(0, rows, ctx.chunk):
                hi = min(lo + ctx.chunk, rows)
                w = dequantize(packed[lo:hi], scales[lo:hi], biases[lo:hi], ctx.dtype)
                acc.add_(torch.matmul(flat[start:end, lo:hi].to(ctx.dtype), w).float())
            out[start:end].copy_(acc.to(ctx.dtype))
        return out.reshape(ctx.input_shape), None, None, None, None


class Packed(nn.Module):
    def __init__(self, tensors, record, signs, dtype):
        super().__init__()
        for name in ("packed", "scales", "biases"):
            self.register_buffer(name, tensors[name])
        self.register_buffer("signs", signs)
        self.block, self.embedding = record["block"], record["inverse"]
        self.shape, self.dtype = record["shape"], dtype
        self.chunk = 8192

    def forward(self, x):
        if self.embedding:
            # Embedding rows are frozen; only the inserted external prompt trains.
            idx = x.reshape(-1)
            # CUDA uint32 indexing is not supported in some torch versions.
            p = self.packed.view(torch.int32).index_select(0, idx).view(torch.uint32)
            s, b = self.scales.index_select(0, idx), self.biases.index_select(0, idx)
            y = dequantize(p, s, b, self.dtype).reshape(*x.shape, self.shape[1])
            return fwht(y, self.block, self.signs, inverse=True) if self.block else y
        if self.block:
            x = fwht(x, self.block, self.signs)
        return FrozenLinearFunction.apply(x, self.packed, self.scales, self.biases, self.chunk)


def locate(model, target):
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def load_backend(artifact, dtype=torch.bfloat16, device="cuda"):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding,
    )
    artifact = Path(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text())
    cfg = Qwen3_5TextConfig(**manifest["config"], dtype=dtype, use_cache=False)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    expected = {name: tuple(p.shape) for name, p in model.named_parameters()}
    assigned = set()
    sign_map = load_file(str(artifact / "signs.safetensors"), device=device)
    for record in manifest["tensors"]:
        target = record["target"]
        if target not in expected or target in assigned:
            raise ValueError(f"Unknown/duplicate tensor: {target}")
        if tuple(record["shape"]) != expected[target]:
            raise ValueError(f"Shape mismatch {target}: {record['shape']} != {expected[target]}")
        data = load_file(str(artifact / record["file"]), device=device)
        if record["kind"] == "packed":
            path = target.removesuffix(".weight")
            parent, key = locate(model, path)
            signs = sign_map[str(record["sign_width"])] if record["block"] else None
            setattr(parent, key, Packed(data, record, signs, dtype))
        else:
            parent, key = locate(model, target)
            value = data["value"]
            # Preserve precision for scalar dynamics and effective RMSNorm weights.
            if value.ndim >= 2:
                value = value.to(dtype)
            setattr(parent, key, nn.Parameter(value, requires_grad=False))
        assigned.add(target)
    if set(expected) != assigned:
        raise ValueError(f"Unassigned tensors: {set(expected) - assigned}")
    # RoPE buffers constructed on meta need to be explicitly re-created.
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg, device=device)
    model.requires_grad_(False)
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
        raise ValueError("Remaining meta storage")
    model.eval()
    return model, manifest


class PromptEncoder(nn.Module):
    """P-tuning: trainable prompt embeddings passed through an external MLP."""
    def __init__(self, length, hidden_size, bottleneck=256):
        super().__init__()
        self.length, self.hidden_size, self.bottleneck = length, hidden_size, bottleneck
        self.embedding = nn.Embedding(length, hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, bottleneck), nn.Tanh(),
            nn.Linear(bottleneck, hidden_size),
        )
        nn.init.normal_(self.embedding.weight, std=0.02)
        # Residual MLP initialized small, with nonzero gradients through both layers.
        nn.init.normal_(self.encoder[-1].weight, std=0.001)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self):
        base = self.embedding.weight
        return base + self.encoder(base)

    def inject(self, frozen_embeddings):
        prompt = self().to(frozen_embeddings.dtype).unsqueeze(0)
        return torch.cat((prompt.expand(frozen_embeddings.shape[0], -1, -1),
                          frozen_embeddings), dim=1)

    def save(self, path, source_sha256):
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(path / "prompt_encoder.safetensors"))
        (path / "adapter_config.json").write_text(json.dumps({
            "method": "p_tuning_external_residual_mlp", "length": self.length,
            "hidden_size": self.hidden_size, "bottleneck": self.bottleneck,
            "source_sha256": source_sha256, "base_frozen": True,
            "gguf_server_injection_validated": False,
        }, indent=2))

    @classmethod
    def load(cls, path, device="cuda"):
        path = Path(path)
        cfg = json.loads((path / "adapter_config.json").read_text())
        obj = cls(cfg["length"], cfg["hidden_size"], cfg["bottleneck"]).to(device)
        obj.load_state_dict(load_file(str(path / "prompt_encoder.safetensors"), device=device))
        return obj
