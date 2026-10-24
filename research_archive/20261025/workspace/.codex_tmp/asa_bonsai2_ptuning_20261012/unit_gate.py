"""Gate the packed linear input gradient against independent dense autograd."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from backend import FrozenLinearFunction, dequantize, fwht

sys.path.insert(0, str(Path(__file__).parent / "official"))
from codec import unpack

torch.manual_seed(42)
rows, width = 37, 256
p = torch.randint(0, 2**31 - 1, (rows, width // 16), dtype=torch.int32, device="cuda").view(torch.uint32)
s = torch.rand(rows, width // 128, device="cuda", dtype=torch.float16) / 20
b = -s
x = torch.randn(2, 3, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
ref = torch.from_numpy(unpack(p.cpu().numpy(), s.cpu().numpy(), b.cpu().numpy())).cuda().bfloat16()
decoded = dequantize(p, s, b, torch.bfloat16)
assert torch.equal(decoded, ref)
y = FrozenLinearFunction.apply(x, p, s, b, 19)
grad = torch.randn_like(y)
y.backward(grad)
x2 = x.detach().clone().requires_grad_()
y2 = torch.nn.functional.linear(x2, ref)
y2.backward(grad)
forward_diff = (y - y2).abs().max().item()
grad_diff = (x.grad - x2.grad).abs().max().item()
assert forward_diff == 0
assert torch.allclose(x.grad, x2.grad, atol=0.003, rtol=0.02)
signs = torch.where(torch.arange(width, device="cuda") % 2 == 0, 1.0, -1.0)
z = torch.randn(2, width, device="cuda", requires_grad=True)
roundtrip = fwht(fwht(z, 128, signs), 128, signs, inverse=True)
assert torch.allclose(z, roundtrip, atol=1e-6, rtol=1e-5)
roundtrip.sum().backward()
assert torch.allclose(z.grad, torch.ones_like(z), atol=1e-6)


def reference_fwht(x, block, signs, inverse=False):
    shape = x.shape
    z = x.float()
    if not inverse:
        z = z * signs
    z = z.reshape(-1, block)
    step = 1
    while step < block:
        left, right = z.reshape(z.shape[0], -1, 2, step).unbind(-2)
        z = torch.stack((left + right, left - right), dim=-2).reshape(z.shape)
        step *= 2
    z = (z / math.sqrt(block)).reshape(shape)
    if inverse:
        z = z * signs
    return z.to(x.dtype)


transform_cases = []
for width in (256, 5120, 17408):
    signs = torch.where(torch.arange(width, device="cuda") % 3 == 0, 1.0, -1.0)
    for dtype in (torch.float32, torch.bfloat16):
        for inverse in (False, True):
            # More than one chunk, noncontiguous input, multiple Hadamard blocks.
            value = torch.randn(2, width, 263, device="cuda", dtype=dtype).transpose(1, 2)
            before = value.clone()
            a = value.detach().requires_grad_()
            b_ref = value.detach().clone().requires_grad_()
            got = fwht(a, 128, signs, inverse=inverse)
            expected = reference_fwht(b_ref, 128, signs, inverse=inverse)
            assert torch.equal(value, before)
            assert torch.equal(got, expected)
            upstream = torch.randn_like(got)
            got.backward(upstream)
            expected.backward(upstream)
            error = (a.grad - b_ref.grad).abs().max().item()
            tolerance = 1e-6 if dtype == torch.float32 else 0.03125
            assert torch.allclose(a.grad, b_ref.grad, atol=tolerance, rtol=0.02), error
            transform_cases.append({"width": width, "dtype": str(dtype),
                                    "inverse": inverse, "grad_max_abs": error})

# Exercise the sequence-row chunk in packed-linear backward.
long_x = torch.randn(2051, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
long_ref = long_x.detach().clone().requires_grad_()
long_y = FrozenLinearFunction.apply(long_x, p, s, b, 19)
long_dense = torch.nn.functional.linear(long_ref, ref)
upstream = torch.randn_like(long_y)
long_y.backward(upstream)
long_dense.backward(upstream)
long_forward_diff = (long_y - long_dense).abs().max().item()
assert torch.allclose(long_y, long_dense, atol=0.016, rtol=0.02), long_forward_diff
assert torch.allclose(long_x.grad, long_ref.grad, atol=0.004, rtol=0.02)
# FP32 isolates BF16 GEMM tiling/rounding from implementation mistakes.
fp_x = long_x.detach().float().requires_grad_()
fp_ref = fp_x.detach().clone().requires_grad_()
fp_w = dequantize(p, s, b, torch.float32)
fp_y = FrozenLinearFunction.apply(fp_x, p, s, b, 19)
fp_dense = torch.nn.functional.linear(fp_ref, fp_w)
fp_upstream = upstream.float()
fp_y.backward(fp_upstream)
fp_dense.backward(fp_upstream)
assert torch.allclose(fp_y, fp_dense, atol=1e-6, rtol=1e-5)
assert torch.allclose(fp_x.grad, fp_ref.grad, atol=1e-6, rtol=1e-5)
print(json.dumps({"dequant_exact": True, "forward_max_abs": forward_diff,
                  "input_grad_max_abs": grad_diff, "fwht_roundtrip": True,
                  "bounded_transform_cases": transform_cases,
                  "long_linear_forward_max_abs": long_forward_diff,
                  "long_linear_grad_max_abs": (long_x.grad - long_ref.grad).abs().max().item()}), flush=True)
