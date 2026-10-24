"""Lossless, chunked Prism GGUF -> frozen affine-2bit PyTorch artifact.

Uses the pinned official codec, retaining signed Hadamard metadata. This is
not a dense approximation or a different base checkpoint.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent / "official"))
from codec import transcode, unpack
from gguf import GGUFReader

EXPECTED_SHA = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
MAPPING = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}
GLOBAL = {"output.weight": "lm_head.weight", "output_norm.weight": "model.norm.weight",
          "token_embd.weight": "model.embed_tokens.weight"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite completed conversion")
    with open(args.gguf, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    assert digest == EXPECTED_SHA, digest
    reader = GGUFReader(args.gguf)
    fields = {k: v.contents() for k, v in reader.fields.items()}
    assert fields["general.architecture"] == "qwen35"
    assert fields["prism.hadamard.sign_mode"] == "explicit"
    g = lambda k: fields["qwen35." + k]
    nv, nk = int(g("ssm.time_step_rank")), int(g("ssm.group_count"))
    hd, hk = int(g("ssm.inner_size")) // nv, int(g("ssm.state_size"))
    assert nv % nk == 0
    tensors = {t.name: t for t in reader.tensors}
    block = int(fields["prism.hadamard.block_size"])
    assert block in (512, 1024, 2048, 4096)
    folded = set(fields["prism.hadamard.weight_names"])
    inverse = set(fields.get("prism.hadamard.inverse_weight_names", []))
    assert inverse == {"token_embd.weight"} and not folded & inverse
    assert (folded | inverse) <= tensors.keys()
    if nv != nk:
        assert fields["prism.hadamard.gdn_v_grouped"]
    signs, offset = {}, 0
    for width in fields["prism.hadamard.sign_widths"]:
        width = int(width)
        a = np.array(fields["prism.hadamard.sign_values"][offset:offset + width], dtype=np.float32)
        assert len(a) == width and np.isin(a, [-1, 1]).all()
        signs[str(width)] = torch.from_numpy(a)
        offset += width
    assert offset == len(fields["prism.hadamard.sign_values"])
    save_file(signs, str(out / "signs.safetensors"))
    cfg = dict(
        hidden_size=int(g("embedding_length")), intermediate_size=int(g("feed_forward_length")),
        num_hidden_layers=int(g("block_count")), num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")), head_dim=int(g("attention.key_length")),
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        vocab_size=int(tensors["token_embd.weight"].shape[1]),
        max_position_embeddings=int(g("context_length")),
        linear_num_value_heads=nv, linear_num_key_heads=nk,
        linear_value_head_dim=hd, linear_key_head_dim=hk,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")), tie_word_embeddings=False,
        layer_types=["full_attention" if (i + 1) % int(g("full_attention_interval")) == 0
                     else "linear_attention" for i in range(int(g("block_count")))],
        rope_parameters={"rope_type": "default", "rope_theta": float(g("rope.freq_base")),
                         "partial_rotary_factor": int(g("rope.dimension_count")) / int(g("attention.key_length"))},
    )

    def vperm(unit):
        return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)

    def reorder(a, stem):
        if nv == nk:
            return a
        if stem in ("attn_qkv.weight", "ssm_conv1d.weight"):
            qk = 2 * nk * hk
            return np.concatenate((a[:qk], a[qk:][vperm(hd)]), axis=0)
        if stem == "attn_gate.weight":
            return a[vperm(hd)]
        if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias"):
            return a[vperm(1)]
        return a

    manifest = {"source_sha256": digest, "config": cfg, "block": block, "tensors": [],
                "codec_revision": "fcba37d2117a7077eac6b613b2668d14d9779edd"}
    for i, (name, t) in enumerate(tensors.items()):
        stem = name
        if name in GLOBAL:
            target = GLOBAL[name]
        else:
            _, layer, stem = name.split(".", 2)
            target = f"model.layers.{layer}." + MAPPING[stem]
        shape = tuple(map(int, t.shape[::-1]))
        filename = f"{i:04d}.safetensors"
        record = {"gguf_name": name, "target": target, "shape": shape, "file": filename}
        if t.tensor_type.name in ("PTQ1_0", "PQ2_0"):
            rows, width = shape
            arrays = [np.empty((rows, width // 16), dtype=np.uint32),
                      np.empty((rows, width // 128), dtype=np.float16),
                      np.empty((rows, width // 128), dtype=np.float16)]
            raw = t.data.reshape(rows, -1)
            for start in range(0, rows, 512):
                end = min(start + 512, rows)
                chunk = transcode(raw[start:end].tobytes(), (end - start, width), t.tensor_type.name)
                for dest, src in zip(arrays, chunk):
                    dest[start:end] = src
            arrays = [np.ascontiguousarray(reorder(a, stem)) for a in arrays]
            # Independent affine codec round-trip: no illegal fourth code.
            test = unpack(*(a[:2] for a in arrays))
            assert np.isfinite(test).all()
            words = arrays[0][:2]
            assert all((((words >> (2 * lane)) & 3) <= 2).all() for lane in range(16))
            data = {k: torch.from_numpy(a) for k, a in zip(("packed", "scales", "biases"), arrays)}
            active = name in folded or name in inverse
            record.update(kind="packed", block=block if active else 0,
                          inverse=name in inverse, sign_width=width if active else 0)
        else:
            assert t.tensor_type.name in ("F32", "F16", "BF16")
            assert name not in folded | inverse
            if t.tensor_type.name == "BF16":
                # Reader represents BF16 as bytes, not native NumPy floats.
                bits = np.frombuffer(t.data.tobytes(), dtype="<u2").astype(np.uint32) << 16
                a = bits.view(np.float32).reshape(shape)
            else:
                a = np.asarray(t.data).copy()
            a = reorder(a, stem)
            if stem == "ssm_a":
                assert (a < 0).all()
                a = np.log(-a)
            elif stem == "ssm_conv1d.weight":
                a = a[:, None, :]
            elif (target.endswith(("input_layernorm.weight", "post_attention_layernorm.weight",
                                   "q_norm.weight", "k_norm.weight")) or target == "model.norm.weight"):
                # HF uses (1 + weight); GGUF/MLX already store effective weight.
                a = a.astype(np.float32) - 1.0
            assert np.isfinite(a).all()
            data = {"value": torch.from_numpy(np.ascontiguousarray(a))}
            record.update(kind="float", shape=list(a.shape))
        save_file(data, str(out / filename))
        record["sha256"] = hashlib.sha256((out / filename).read_bytes()).hexdigest()
        manifest["tensors"].append(record)
        print(json.dumps({"index": i, "target": target, "shape": record["shape"], "kind": record["kind"]}), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Export GGUF tokenizer data for exact token IDs; no tokenizer substitution.
    tokenizer_fields = {k: v for k, v in fields.items() if k.startswith("tokenizer.")}
    (out / "tokenizer_gguf.json").write_text(json.dumps(tokenizer_fields, ensure_ascii=False))
    print("CONVERSION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
