"""Small actual-model GPU loss/gradient equivalence test; no data training."""
from copy import deepcopy
import json

import prepare


def run():
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

    assert not (prepare.HERE / "kernel_equivalence.json").exists()
    torch.cuda.set_per_process_memory_fraction(0.10, 0)
    torch.manual_seed(123)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": 64,
            "layer_types": ["full_attention"], "max_position_embeddings": 128,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.25, "mrope_section": [2, 3, 3]},
            "use_cache": False,
        },
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 64},
    )
    original_forward = Qwen3_5ForConditionalGeneration.forward
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        Qwen3_5ForConditionalGeneration.forward = original_forward
        original = Qwen3_5ForConditionalGeneration(config).to(device="cuda", dtype=dtype).train()
        candidate = deepcopy(original)
        ids = torch.randint(4, 120, (1, 24), device="cuda")
        labels = ids.clone()
        labels[:, :17] = -100
        labels[:, 20] = -100
        old_loss = original(input_ids=ids, labels=labels, use_cache=False).loss
        old_loss.backward()
        apply_liger_kernel_to_qwen3_5(
            rope=False, cross_entropy=False, fused_linear_cross_entropy=True,
            rms_norm=False, swiglu=False)
        new_output = candidate(input_ids=ids, labels=labels, use_cache=False)
        assert new_output.logits is None, "fused path not active"
        new_loss = new_output.loss
        new_loss.backward()
        atol = 3e-6 if dtype == torch.float32 else 0.012
        rtol = 3e-5 if dtype == torch.float32 else 0.04
        torch.testing.assert_close(new_loss, old_loss, atol=atol, rtol=rtol)
        max_difference, compared = 0.0, 0
        for (name, old), (new_name, new) in zip(original.named_parameters(), candidate.named_parameters()):
            assert name == new_name and (old.grad is None) == (new.grad is None), name
            if old.grad is None:
                continue
            torch.testing.assert_close(new.grad, old.grad, atol=atol, rtol=rtol, msg=name)
            max_difference = max(max_difference, (new.grad.float() - old.grad.float()).abs().max().item())
            compared += 1
        # Evaluation remains native-logit loss, with the same full inputs.
        candidate.eval()
        with torch.no_grad():
            evaluation = candidate(input_ids=ids, labels=labels, use_cache=False)
            assert evaluation.logits.shape == (1, 24, 128)
            torch.testing.assert_close(evaluation.loss, old_loss, atol=atol, rtol=rtol)
        rows.append({"dtype": str(dtype), "original_loss": old_loss.item(),
                     "fused_loss": new_loss.item(), "gradient_tensors_compared": compared,
                     "max_abs_gradient_difference": max_difference,
                     "atol": atol, "rtol": rtol, "evaluation_full_logits_checked": True})
        del original, candidate, old_loss, new_loss, new_output, evaluation
        torch.cuda.empty_cache()
    Qwen3_5ForConditionalGeneration.forward = original_forward
    receipt = {"passed": True, "model_forward_gradient_cases": len(rows), "cases": rows,
               "created_at_utc": prepare.now(), "training_dataset_used": False}
    prepare.save(prepare.HERE / "kernel_equivalence.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    run()
