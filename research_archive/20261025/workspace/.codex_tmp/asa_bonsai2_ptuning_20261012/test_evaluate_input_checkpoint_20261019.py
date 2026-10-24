import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_input_checkpoint_20261019 as evaluation


class SnapshotTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        trained = source / "trained"
        checkpoint = trained / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "prompt_encoder.safetensors").write_bytes(b"fixture-weights-not-real")
        (checkpoint / "adapter_config.json").write_text("{}")
        data = root / "data"
        data.mkdir()
        for split in ("train", "val"):
            (data / f"{split}.jsonl").write_text("{}\n")
        run = dict(
            data_sha256={s: evaluation.sha(data / f"{s}.jsonl") for s in ("train", "val")},
            train_examples=1117, epochs=1, training_complete=False)
        (trained / "run.json").write_text(json.dumps(run))
        (trained / "progress.jsonl").write_text(
            '{"event":"optimizer_step","step":1}\n'
            '{"event":"optimizer_step","step":2}\n')
        return argparse.Namespace(source_run=source, data=data, out=root / "evaluation", gpu=0)

    def test_snapshot_excludes_latest_possibly_uncommitted_step_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            source_hash = evaluation.sha(args.source_run / "trained/run.json")
            with patch.object(evaluation.subprocess, "check_output", return_value="GPU-test, 24000"), \
                    patch.object(evaluation.subprocess, "Popen") as process:
                process.return_value.pid = 12345
                evaluation.launch(args)
            receipt = json.loads((args.out / "snapshot_receipt.json").read_text())
            self.assertEqual(receipt["checkpoint_step"], 1)
            self.assertFalse(receipt["source_training_complete_at_snapshot"])
            self.assertTrue(receipt["original_training_unmodified"])
            self.assertEqual(evaluation.sha(args.source_run / "trained/run.json"), source_hash)
            self.assertEqual(
                receipt["adapter_sha256"]["prompt_encoder.safetensors"],
                evaluation.sha(args.out / "adapter_snapshot/prompt_encoder.safetensors"))

    def test_insufficient_memory_refuses_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            with patch.object(evaluation.subprocess, "check_output", return_value="GPU-test, 10000"):
                with self.assertRaises(AssertionError):
                    evaluation.launch(args)
            self.assertFalse(args.out.exists())

    def test_data_hash_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            (args.data / "val.jsonl").write_text('{"changed":true}\n')
            with self.assertRaises(AssertionError):
                evaluation.launch(args)
            self.assertFalse(args.out.exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args.out.mkdir()
            (args.out / "keep").write_text("user-data")
            with self.assertRaises(AssertionError):
                evaluation.launch(args)
            self.assertEqual((args.out / "keep").read_text(), "user-data")


if __name__ == "__main__":
    unittest.main()
