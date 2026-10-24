"""Small real Qwen3.5 CPU tests; no 27B loading or GPU allocation."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from ptuning_v2_hybrid import DeepPrefix, PrefixController, architecture, greedy_generate


class HybridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        torch.set_num_threads(2)
        cfg = Qwen3_5TextConfig(
            vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_conv_kernel_dim=4, max_position_embeddings=128,
            layer_types=["linear_attention", "full_attention",
                         "linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                             "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
        )
        cfg._attn_implementation = "sdpa"
        self.model = Qwen3_5ForCausalLM(cfg).eval().requires_grad_(False)
        self.adapter = DeepPrefix(architecture(cfg), 3)
        self.control = PrefixController(self.model, self.adapter)
        self.ids = torch.tensor([[5, 8, 11, 13, 17, 19]])
        self.seed = torch.randn(3, 32) * 0.02
        self.reference = self.control.calibrate(self.seed, self.ids)

    def tearDown(self):
        self.control.close()

    def logits(self):
        with self.control.activate():
            return self.model(inputs_embeds=self.adapter.prepare(
                self.model.model.embed_tokens(self.ids)), use_cache=False,
                logits_to_keep=1).logits

    def test_calibration_and_independent_layers(self):
        torch.testing.assert_close(self.logits(), self.reference, rtol=0, atol=1e-6)
        self.assertEqual(len({p.data_ptr() for p in self.adapter.parameters()}), 6)
        embedded = self.model.model.embed_tokens(self.ids)
        self.assertTrue(torch.equal(self.adapter.prepare(embedded)[:, 3:], embedded))

    def test_all_gradients_checkpoint_and_frozen_backbone(self):
        plain = self.logits().float().square().sum()
        plain.backward()
        expected = {k: p.grad.clone() for k, p in self.adapter.prefix.items()}
        self.adapter.assert_gradients()
        self.adapter.zero_grad(set_to_none=True)
        self.model.train()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        with self.control.activate():
            loss = self.model(inputs_embeds=self.adapter.prepare(
                self.model.model.embed_tokens(self.ids)), use_cache=False,
                logits_to_keep=1).logits.float().square().sum()
            loss.backward()
        self.adapter.assert_gradients()
        for name, p in self.adapter.prefix.items():
            torch.testing.assert_close(p.grad, expected[name], rtol=1e-5, atol=1e-7)
        self.assertTrue(all(p.grad is None and not p.requires_grad
                            for p in self.model.parameters()))
        self.assertEqual({id(p) for p in self.adapter.parameters()} &
                         {id(p) for p in self.model.parameters()}, set())

    def test_cached_decode_matches_full_recompute(self):
        # Deliberately change calibrated prompts; test trained, not only identity.
        with torch.no_grad():
            for p in self.adapter.parameters():
                p.add_(torch.randn_like(p) * 0.01)
            with self.control.activate():
                out = self.model(inputs_embeds=self.adapter.prepare(
                    self.model.model.embed_tokens(self.ids)), use_cache=True,
                    logits_to_keep=1)
            ids = self.ids.clone()
            for _ in range(3):
                new = out.logits[:, -1:].argmax(-1)
                ids = torch.cat((ids, new), dim=1)
                with self.control.activate(prefill=False):
                    out = self.model(input_ids=new, past_key_values=out.past_key_values,
                                     use_cache=True, logits_to_keep=1)
                with self.control.activate():
                    full = self.model(inputs_embeds=self.adapter.prepare(
                        self.model.model.embed_tokens(ids)), use_cache=False,
                        logits_to_keep=1).logits
                torch.testing.assert_close(out.logits, full, rtol=1e-4, atol=1e-6)
            with self.control.activate():
                with self.assertRaises(ValueError):
                    self.model(input_ids=ids, past_key_values=out.past_key_values,
                               use_cache=True)

    def test_roundtrip_strict_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "adapter"
            self.adapter.save(path, "source-test")
            loaded = DeepPrefix.load(path, self.adapter.spec, "source-test", "cpu")
            for key, tensor in self.adapter.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[key]))
            with self.assertRaises(ValueError):
                DeepPrefix.load(path, self.adapter.spec, "WRONG", "cpu")
            spec = copy.deepcopy(self.adapter.spec)
            spec["head_dim"] += 1
            with self.assertRaises(ValueError):
                DeepPrefix.load(path, spec, "source-test", "cpu")
            with self.assertRaises(FileExistsError):
                self.adapter.save(path, "source-test")

    def test_scopes_cleanup_and_context_guard(self):
        with self.assertRaises(RuntimeError):
            with self.control.activate():
                raise RuntimeError("test")
        self.assertIsNone(self.control.mode)
        with self.assertRaises(ValueError):
            PrefixController(self.model, self.adapter)
        with self.assertRaises(ValueError):
            self.adapter.prepare(torch.zeros(2, 6, 32))
        with self.assertRaises(ValueError):
            greedy_generate(self.model, self.adapter, self.control, self.ids,
                            120, set(), context_limit=128)
        self.assertEqual(len(greedy_generate(
            self.model, self.adapter, self.control, self.ids, 3, set())), 3)
        self.control.close()
        self.assertFalse(self.model._asa_deep_prefix_attached)
        replacement = PrefixController(self.model, self.adapter)
        replacement.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
