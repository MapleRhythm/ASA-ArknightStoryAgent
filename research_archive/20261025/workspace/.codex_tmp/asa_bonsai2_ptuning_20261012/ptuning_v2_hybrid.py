"""External deep prefixes for frozen Qwen3.5 hybrid attention.

Full attention: independent, directly optimized K/V prefixes (pre-RoPE K).
Gated DeltaNet: independent normalized-input prefixes drive convolution and
recurrent state. This is a hybrid adaptation of P-tuning v2, not conventional
KV prefix tuning applied to a nonexistent GDN KV cache.

Reserve prefix positions ONCE. No hooks run on cached decode. The activate()
scope MUST include backward when gradient checkpointing recomputes forwards.
No monkey-patched Transformers source and no trainable backbone parameters.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file


METHOD = "p_tuning_v2_hybrid_direct_kv_and_gdn_prefix"


def architecture(config):
    result = {key: getattr(config, key) for key in (
        "model_type", "hidden_size", "num_hidden_layers", "num_key_value_heads",
        "head_dim", "layer_types", "num_attention_heads", "linear_num_key_heads",
        "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
        "linear_conv_kernel_dim",
    )}
    result["layer_types"] = list(result["layer_types"])
    if result["model_type"] != "qwen3_5_text":
        raise ValueError("Only verified Qwen3.5 text architecture is supported")
    if len(result["layer_types"]) != result["num_hidden_layers"]:
        raise ValueError("Layer count mismatch")
    if set(result["layer_types"]) - {"linear_attention", "full_attention"}:
        raise ValueError("Unsupported attention type")
    # Include ALL remaining configuration, including RoPE and normalization.
    # Exclude runtime-only flags changed by training/evaluation.
    full = config.to_dict()
    for key in ("use_cache", "output_attentions", "output_hidden_states",
                "_attn_implementation_internal", "transformers_version",
                "dtype", "torch_dtype", "_name_or_path"):
        full.pop(key, None)
    result["full_config_sha256"] = hashlib.sha256(
        json.dumps(full, sort_keys=True, default=str).encode()).hexdigest()
    return result


class DeepPrefix(nn.Module):
    def __init__(self, spec, length=16):
        super().__init__()
        if length < 1:
            raise ValueError("Prefix length must be positive")
        self.spec, self.length = spec, length
        self.register_buffer("input_seed", torch.zeros(length, spec["hidden_size"]))
        params = {}
        for index, kind in enumerate(spec["layer_types"]):
            if kind == "linear_attention":
                params[f"layer_{index:03d}_gdn"] = nn.Parameter(
                    torch.zeros(length, spec["hidden_size"]))
            elif kind == "full_attention":
                for branch in ("k", "v"):
                    params[f"layer_{index:03d}_{branch}"] = nn.Parameter(torch.zeros(
                        length, spec["num_key_value_heads"], spec["head_dim"]))
            else:
                raise ValueError(kind)
        self.prefix = nn.ParameterDict(params)

    def prepare(self, embedded):
        if embedded.ndim != 3 or embedded.shape[0] != 1:
            raise ValueError("Only unpadded batch size one is currently supported")
        if embedded.shape[-1] != self.spec["hidden_size"]:
            raise ValueError("Embedding width mismatch")
        seed = self.input_seed.to(device=embedded.device, dtype=embedded.dtype)
        return torch.cat((seed.unsqueeze(0), embedded), dim=1)

    def gradient_stats(self):
        result = {}
        for name, p in self.prefix.items():
            result[name] = {
                "present": p.grad is not None,
                "finite": bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
                "norm": p.grad.float().norm().item() if p.grad is not None else 0.0,
            }
        return result

    def assert_gradients(self):
        stats = self.gradient_stats()
        if not all(s["present"] and s["finite"] and s["norm"] > 0 for s in stats.values()):
            raise RuntimeError(f"Missing/nonfinite/zero deep-prefix gradient: {stats}")
        return stats

    def save(self, path, source_sha256):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        weights = path / "deep_prefix.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(weights))
        (path / "adapter_config.json").write_text(json.dumps({
            "method": METHOD, "length": self.length, "architecture": self.spec,
            "source_sha256": source_sha256, "base_frozen": True,
            "weight_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "gguf_server_injection_validated": False,
        }, indent=2))

    @classmethod
    def load(cls, path, spec, source_sha256, device="cuda"):
        path = Path(path)
        config = json.loads((path / "adapter_config.json").read_text())
        if (config["method"] != METHOD or config["architecture"] != spec
                or config["source_sha256"] != source_sha256):
            raise ValueError("Adapter method/architecture/backbone mismatch")
        weights = path / "deep_prefix.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != config["weight_sha256"]:
            raise ValueError("Corrupt adapter weights")
        obj = cls(spec, config["length"]).to(device)
        obj.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
        return obj


class PrefixController:
    """Removable hooks; external adapter is deliberately NOT a model submodule."""
    def __init__(self, model, adapter):
        if architecture(model.config) != adapter.spec:
            raise ValueError("Model architecture does not match adapter")
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Backbone must already be frozen")
        if getattr(model, "_asa_deep_prefix_attached", False):
            raise ValueError("Duplicate deep-prefix controller")
        model._asa_deep_prefix_attached = True
        self.model, self.adapter, self.mode = model, adapter, None
        self.handles = []
        try:
            for index, layer in enumerate(model.model.layers):
                if adapter.spec["layer_types"][index] == "linear_attention":
                    self.handles.append(layer.linear_attn.register_forward_pre_hook(
                        self._gdn_hook(index), with_kwargs=True))
                else:
                    self.handles.append(layer.self_attn.k_norm.register_forward_hook(
                        self._kv_hook(index, "k")))
                    self.handles.append(layer.self_attn.v_proj.register_forward_hook(
                        self._kv_hook(index, "v")))
        except BaseException:
            self.close()
            raise

    def _replace(self, value, name, flatten=False):
        if self.mode != "prefill":
            return value
        length = self.adapter.length
        if value.shape[0] != 1 or value.shape[1] < length:
            raise ValueError("Invalid prefill shape; never prefix cached decode")
        prompt = self.adapter.prefix[name]
        if flatten:
            prompt = prompt.flatten(1)
        if tuple(value.shape[2:]) != tuple(prompt.shape[1:]):
            raise ValueError(f"Prefix shape mismatch at {name}: {value.shape}, {prompt.shape}")
        prompt = prompt.to(device=value.device, dtype=value.dtype).unsqueeze(0)
        return torch.cat((prompt, value[:, length:]), dim=1)

    def _kv_hook(self, index, branch):
        def hook(module, args, output):
            return self._replace(output, f"layer_{index:03d}_{branch}", branch == "v")
        return hook

    def _gdn_hook(self, index):
        def hook(module, args, kwargs):
            if self.mode != "prefill":
                return None
            cache = kwargs.get("cache_params")
            if cache is not None and cache.has_previous_state(index):
                raise ValueError("Refusing prefix reinjection into populated GDN cache")
            if args:
                raise ValueError("Expected named GDN hidden_states (verified Transformers API)")
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = self._replace(
                kwargs["hidden_states"], f"layer_{index:03d}_gdn")
            return args, kwargs
        return hook

    @contextmanager
    def activate(self, prefill=True):
        if not self.handles:
            raise RuntimeError("Controller has been closed")
        previous = self.mode
        self.mode = "prefill" if prefill else "decode"
        try:
            yield self
        finally:
            self.mode = previous

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.mode = None
        self.model._asa_deep_prefix_attached = False

    @torch.no_grad()
    def calibrate(self, seed, token_ids):
        """Copy the old INITIAL shallow prompt's per-layer activations.

        Uses no labels and no final-trained adapter. At initialization the deep
        model must reproduce this shallow prompt (checked by the gate).
        """
        if seed.shape != self.adapter.input_seed.shape:
            raise ValueError("Calibration seed shape mismatch")
        self.adapter.input_seed.copy_(seed.float())
        handles, captured = [], set()

        def capture(name, value, flatten=False):
            if name in captured:
                raise RuntimeError("Calibration hook called twice")
            value = value[0, :self.adapter.length]
            if flatten:
                value = value.reshape_as(self.adapter.prefix[name])
            self.adapter.prefix[name].copy_(value.float())
            captured.add(name)

        for index, layer in enumerate(self.model.model.layers):
            if self.adapter.spec["layer_types"][index] == "linear_attention":
                name = f"layer_{index:03d}_gdn"

                def pre(module, args, kwargs, name=name):
                    capture(name, kwargs["hidden_states"])
                handles.append(layer.linear_attn.register_forward_pre_hook(pre, with_kwargs=True))
            else:
                for branch, module in (("k", layer.self_attn.k_norm),
                                       ("v", layer.self_attn.v_proj)):
                    name = f"layer_{index:03d}_{branch}"

                    def post(module, args, output, name=name, branch=branch):
                        capture(name, output, branch == "v")
                    handles.append(module.register_forward_hook(post))
        self.model.eval()
        try:
            with self.activate(prefill=False):
                embedded = self.model.model.embed_tokens(token_ids).detach()
                logits = self.model(inputs_embeds=self.adapter.prepare(embedded),
                                    use_cache=False, logits_to_keep=1).logits.detach()
        finally:
            for handle in handles:
                handle.remove()
        if captured != set(self.adapter.prefix):
            raise RuntimeError("Not every layer was calibrated")
        return logits


@torch.no_grad()
def greedy_generate(model, adapter, controller, input_ids, max_new_tokens, eos_ids,
                    context_limit=32768):
    """Explicit prefill/decode; prefix and all evidence are retained."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
        raise ValueError("Require one nonempty, unpadded prompt")
    if max_new_tokens < 1 or adapter.length + input_ids.shape[1] + max_new_tokens > context_limit:
        raise ValueError("Context budget exceeded; refusing to truncate")
    model.eval()
    with controller.activate(prefill=True):
        output = model(inputs_embeds=adapter.prepare(model.model.embed_tokens(input_ids)),
                       use_cache=True, logits_to_keep=1)
    generated = []
    for _ in range(max_new_tokens):
        next_id = int(output.logits[0, -1].argmax())
        generated.append(next_id)
        if next_id in eos_ids or len(generated) == max_new_tokens:
            break
        with controller.activate(prefill=False):
            output = model(input_ids=torch.tensor([[next_id]], device=input_ids.device),
                           past_key_values=output.past_key_values,
                           use_cache=True, logits_to_keep=1)
    return generated
