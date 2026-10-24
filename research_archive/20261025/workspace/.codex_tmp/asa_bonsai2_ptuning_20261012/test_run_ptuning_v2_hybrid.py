import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ptuning_v2_hybrid import DeepPrefix
from run_ptuning_v2_hybrid import save_checkpoint, sha


class CheckpointTests(unittest.TestCase):
    def test_committed_pair_integrity_and_safe_replay(self):
        spec = {"hidden_size": 8, "layer_types": ["linear_attention", "full_attention"],
                "num_key_value_heads": 1, "head_dim": 4}
        adapter = DeepPrefix(spec, 2)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-5)
        sum(p.square().sum() + p.sum() for p in adapter.parameters()).backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as temp, patch(
                "torch.cuda.get_rng_state", return_value=torch.get_rng_state()):
            out = Path(temp)
            save_checkpoint(out, 1, adapter, optimizer, "test")
            first = json.loads((out / "latest_checkpoint.json").read_text())
            save_checkpoint(out, 1, adapter, optimizer, "test")
            second = json.loads((out / "latest_checkpoint.json").read_text())
            self.assertNotEqual(first["directory"], second["directory"])
            self.assertTrue((out / first["directory"]).is_dir())
            folder = out / second["directory"]
            self.assertEqual(sha(folder / "training_state.pt"),
                             second["training_state_sha256"])
            loaded = DeepPrefix.load(folder, spec, "test", "cpu")
            for k, v in adapter.state_dict().items():
                self.assertTrue(torch.equal(v, loaded.state_dict()[k]))
            state = torch.load(folder / "training_state.pt", weights_only=True)
            self.assertEqual(state["step"], 1)
            self.assertTrue(state["optimizer"]["state"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
