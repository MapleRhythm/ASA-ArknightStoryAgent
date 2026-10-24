"""Check the pinned runtime's gradient-graph support without GPU allocation.

This is an operator compatibility gate, not an actual soft-prompt training run.
The SSM_CONV operator lies on Qwen35's input-to-logits path; frozen kernel
weights do not remove the need for gradients with respect to its input.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import resource
import subprocess
import sys


class InitParams(ct.Structure):
    _fields_ = [("mem_size", ct.c_size_t), ("mem_buffer", ct.c_void_p), ("no_alloc", ct.c_bool)]


def child(lib_path, operation):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lib = ct.CDLL(str(lib_path))
    pointer = ct.c_void_p
    signatures = {
        "ggml_init": ([InitParams], pointer),
        "ggml_new_tensor_3d": ([pointer, ct.c_int, ct.c_int64, ct.c_int64, ct.c_int64], pointer),
        "ggml_new_tensor_2d": ([pointer, ct.c_int, ct.c_int64, ct.c_int64], pointer),
        "ggml_set_param": ([pointer], None),
        "ggml_set_loss": ([pointer], None),
        "ggml_add": ([pointer, pointer, pointer], pointer),
        "ggml_ssm_conv": ([pointer, pointer, pointer], pointer),
        "ggml_sum": ([pointer, pointer], pointer),
        "ggml_new_graph_custom": ([pointer, ct.c_size_t, ct.c_bool], pointer),
        "ggml_build_forward_expand": ([pointer, pointer], None),
        "ggml_build_backward_expand": ([pointer, pointer, pointer], None),
        "ggml_free": ([pointer], None),
    }
    for name, (args, returns) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, returns
    ctx = lib.ggml_init(InitParams(16 * 1024 * 1024, None, False))
    assert ctx
    # F32=0. Mark only activations trainable; convolution backbone is frozen.
    x = lib.ggml_new_tensor_3d(ctx, 0, 5, 2, 2)
    lib.ggml_set_param(x)
    if operation == "control_add":
        y = lib.ggml_add(ctx, x, x)
    else:
        frozen = lib.ggml_new_tensor_2d(ctx, 0, 4, 2)
        y = lib.ggml_ssm_conv(ctx, x, frozen)
    loss = lib.ggml_sum(ctx, y)
    lib.ggml_set_loss(loss)
    graph = lib.ggml_new_graph_custom(ctx, 4096, True)
    lib.ggml_build_forward_expand(graph, loss)
    lib.ggml_build_backward_expand(ctx, graph, None)
    lib.ggml_free(ctx)
    print("backward_graph_constructed", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lib", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--child", choices=["control_add", "ssm_conv"])
    args = p.parse_args()
    if args.child:
        child(args.lib, args.child)
        return
    results = {}
    for operation in ("control_add", "ssm_conv"):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--lib", str(args.lib),
             "--child", operation], capture_output=True, text=True, timeout=30,
        )
        results[operation] = {
            "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
        }
    control_ok = results["control_add"]["returncode"] == 0
    missing = "unsupported ggml op for backward pass: SSM_CONV" in results["ssm_conv"]["stderr"]
    report = {
        "kind": "operator_backward_gate_not_full_model_training",
        "control_passed": control_ok, "ssm_conv_backward_missing": missing,
        "direct_p_tuning_gate_passed": control_ok and results["ssm_conv"]["returncode"] == 0,
        "results": results,
    }
    if args.out:
        with args.out.open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
