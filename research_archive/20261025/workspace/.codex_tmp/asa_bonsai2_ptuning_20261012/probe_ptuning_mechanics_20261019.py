"""CPU-only functional diagnostics; no 27B loading, training, or model changes.

Uses the same Transformers version as the historical remote run, but tiny
random Qwen3.5 weights. These are implementation tests, NOT QA measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from ptuning_v2_hybrid import DeepPrefix, PrefixController, architecture, greedy_generate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b):
    a, b = a.float(), b.float()
    return {
        "max_abs": (a - b).abs().max().item(),
        "mean_abs": (a - b).abs().mean().item(),
        "argmax_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
    }


def model_fixture(dtype):
    torch.manual_seed(20261019)
    config = Qwen3_5TextConfig(
        vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=8,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, max_position_embeddings=4096,
        layer_types=["linear_attention"] * 3 + ["full_attention"]
        + ["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM(config).to(dtype).eval().requires_grad_(False)
    adapter = DeepPrefix(architecture(config), 16)
    control = PrefixController(model, adapter)
    return model, adapter, control


@torch.no_grad()
def run_dtype(dtype, long_lengths):
    model, adapter, control = model_fixture(dtype)
    old_seed = torch.randn(16, 32) * 0.02
    # A second arbitrary prompt, not a trained adapter and not a claimed optimum.
    changed_seed = old_seed + torch.randn_like(old_seed) * 0.008
    short = torch.randint(3, 96, (1, 9))

    def logits(ids, seed=None):
        embedded = model.model.embed_tokens(ids)
        if seed is None:
            embedded = adapter.prepare(embedded)
        else:
            embedded = torch.cat((seed.to(dtype).unsqueeze(0), embedded), dim=1)
        # Compare ONLY real-token positions; prefix hidden outputs may differ.
        positions = torch.arange(16, embedded.shape[1])
        with control.activate(prefill=seed is None):
            return model(inputs_embeds=embedded, use_cache=False,
                         logits_to_keep=positions).logits

    try:
        control.calibrate(old_seed, short)
        initial_state = {k: p.clone() for k, p in adapter.state_dict().items()}
        control.calibrate(changed_seed, short)
        changed_state = {k: p.clone() for k, p in adapter.state_dict().items()}
        # Key test: keep the original, FIXED input seed of the deep run.
        adapter.input_seed.copy_(old_seed)
        mapped = {}
        for length in long_lengths:
            ids = torch.randint(3, 96, (1, length))
            reference = logits(ids, seed=changed_seed)
            mapped[str(length)] = compare(reference, logits(ids))
            torch.testing.assert_close(reference, logits(ids), rtol=0, atol=1e-6)

        # Independently calibrated prefix tensors must not depend on future text.
        control.calibrate(changed_seed, ids)
        prefix_diffs = {
            k: (p - changed_state[k]).abs().max().item()
            for k, p in adapter.state_dict().items() if k.startswith("prefix.")
        }
        assert max(prefix_diffs.values()) == 0
        adapter.input_seed.copy_(old_seed)
        reference = logits(ids, seed=changed_seed)
        arbitrary_seed = torch.randn_like(old_seed) * 2
        adapter.input_seed.copy_(arbitrary_seed)
        seed_invariance = compare(reference, logits(ids))
        torch.testing.assert_close(reference, logits(ids), rtol=0, atol=1e-6)
        adapter.input_seed.copy_(old_seed)

        branch_results = {}
        for branch in ("gdn", "kv"):
            for key, parameter in adapter.prefix.items():
                selected = key.endswith("_gdn") == (branch == "gdn")
                state = changed_state if selected else initial_state
                parameter.copy_(state["prefix." + key])
            branch_results[branch + "_only"] = compare(reference, logits(ids))
        for key, parameter in adapter.prefix.items():
            parameter.copy_(changed_state["prefix." + key])

        # Test a genuinely non-calibrated deep adapter, not just the identity map.
        for parameter in adapter.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.006)
        cache_ids = ids[:, :min(ids.shape[1], 768)].clone()
        with control.activate():
            output = model(inputs_embeds=adapter.prepare(model.model.embed_tokens(cache_ids)),
                           use_cache=True, logits_to_keep=1)
        cache_checks = []
        for _ in range(16):
            token = output.logits[:, -1:].argmax(-1)
            cache_ids = torch.cat((cache_ids, token), dim=1)
            with control.activate(prefill=False):
                output = model(input_ids=token, past_key_values=output.past_key_values,
                               use_cache=True, logits_to_keep=1)
            full = logits(cache_ids)[:, -1:]
            cache_checks.append(compare(full, output.logits))
        tolerance = 1e-5 if dtype == torch.float32 else 0.03
        assert max(x["max_abs"] for x in cache_checks) < tolerance

        # HF generate versus explicit decode for the SAME perturbed deep adapter.
        # This temporary model-level hook selects prefill/decode before layer hooks.
        def select_phase(module, args, kwargs):
            cache = kwargs.get("past_key_values")
            previous = cache is not None and cache.get_seq_length() > 0
            control.mode = "decode" if previous else "prefill"

        generation_ids = ids[:, :256]
        embedded = adapter.prepare(model.model.embed_tokens(generation_ids))
        handle = model.register_forward_pre_hook(select_phase, with_kwargs=True)
        try:
            hf_tokens = model.generate(
                inputs_embeds=embedded,
                attention_mask=torch.ones(embedded.shape[:2], dtype=torch.long),
                do_sample=False, max_new_tokens=12, use_cache=True,
                eos_token_id=None, pad_token_id=0,
            )[0].tolist()
        finally:
            handle.remove()
            control.mode = None
        explicit_tokens = greedy_generate(
            model, adapter, control, generation_ids, 12, set(), context_limit=4096)
        assert hf_tokens == explicit_tokens
        return {
            "dtype": str(dtype), "long_prompt_calibration_mapping": mapped,
            "fixed_original_input_seed_preserved_during_mapping": True,
            "short_vs_long_calibration_max_abs": max(prefix_diffs.values()),
            "arbitrary_input_seed_with_fixed_prefixes": seed_invariance,
            "partial_transplant_diagnostic_not_accuracy": branch_results,
            "perturbed_deep_cache_vs_recompute": cache_checks,
            "hf_vs_explicit_generated_ids_equal": hf_tokens == explicit_tokens,
            "generation_steps": len(explicit_tokens),
        }
    finally:
        control.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != "5.6.2":
        raise RuntimeError("Require the historical Transformers 5.6.2")
    torch.set_num_threads(2)
    start = time.monotonic()
    result = {
        "kind": "tiny_random_qwen35_cpu_functional_diagnostic_not_27B_QA",
        "training_performed": False, "gpu_used": False,
        "production_or_training_settings_modified": False,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "script_sha256": sha(__file__),
        "injection_script_sha256": sha(Path(__file__).with_name("ptuning_v2_hybrid.py")),
        "transformers_modeling_sha256": sha(modeling_qwen3_5.__file__),
        "local_reference_modeling_sha256": sha(
            Path(__file__).with_name("qwen35_model_reference_5_6_2.py")),
        "limitations": [
            "Tiny random weights and CPU kernels, not historical 27B trained weights or GPU kernels.",
            "Calibration mapping tests representability, not successful optimization.",
            "Partial transplant logit differences are not branch QA contributions.",
            "Cache tolerance is numerical, not proof of long-27B generation equivalence.",
        ],
    }
    result["results"] = [
        run_dtype(torch.float32, (512, 2048)),
        run_dtype(torch.bfloat16, (512,)),
    ]
    result["elapsed_seconds"] = time.monotonic() - start
    result["passed"] = True
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
