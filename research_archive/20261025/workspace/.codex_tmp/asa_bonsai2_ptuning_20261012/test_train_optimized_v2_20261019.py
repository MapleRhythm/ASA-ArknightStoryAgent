import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_optimized_v2_20261019 as experiment


class OptimizerTests(unittest.TestCase):
    def test_schedule_warmup_peak_decay_and_floor(self):
        total = 1117
        rates = [experiment.learning_rate(i, total) for i in range(1, total + 1)]
        warmup = math.ceil(total * 0.05)
        self.assertEqual(warmup, 56)
        self.assertAlmostEqual(rates[0], 1e-4 / 56)
        self.assertAlmostEqual(rates[55], 1e-4)
        self.assertAlmostEqual(rates[-1], 2e-5)
        self.assertTrue(all(0 < rate <= 1e-4 for rate in rates))
        self.assertTrue(all(a < b for a, b in zip(rates[:55], rates[1:56])))
        self.assertTrue(all(a >= b for a, b in zip(rates[55:-1], rates[56:])))

    def test_resume_is_schedule_deterministic(self):
        total = 1117
        continuous = [experiment.learning_rate(i, total) for i in range(1, total + 1)]
        resumed = [experiment.learning_rate(i, total) for i in range(1, 501)]
        resumed += [experiment.learning_rate(i, total) for i in range(501, total + 1)]
        self.assertEqual(continuous, resumed)

    def test_invalid_schedule_steps_rejected(self):
        for step, total in [(0, 1117), (-1, 1117), (1118, 1117), (1, 0)]:
            with self.assertRaises(ValueError):
                experiment.learning_rate(step, total)

    def test_optimizer_explicit_settings_and_parameter_isolation(self):
        import torch
        adapter = torch.nn.Linear(3, 2)
        optimizer = experiment.optimizer_for(adapter)
        self.assertEqual(
            {id(p) for g in optimizer.param_groups for p in g["params"]},
            {id(p) for p in adapter.parameters()})
        for group in optimizer.param_groups:
            self.assertEqual(group["weight_decay"], 0.0)
            self.assertEqual(group["lr"], 1e-4)
            self.assertEqual(group["betas"], (0.9, 0.999))
            self.assertEqual(group["eps"], 1e-8)


class PairedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "input"
        self.out = self.root / "deep"
        (self.baseline / "trained").mkdir(parents=True)
        self.out.mkdir()
        self.baseline_patch = patch.object(experiment, "BASELINE", self.baseline)
        self.baseline_patch.start()
        self.addCleanup(self.baseline_patch.stop)
        self.validation = [dict(id=str(i), nll=1.0) for i in range(46)]
        self.data = dict(train="same-train", val="same-val")
        self.write(self.baseline / "trained/run.json", dict(
            training_complete=True, data_sha256=self.data,
            final_validation=self.validation))
        self.write(self.out / "run.json", dict(
            training_complete=True, identity=dict(data_sha256=self.data),
            final_validation=self.validation, final_mean_validation_nll=0.9))
        self.rows = [dict(
            id=str(i), question=f"问题{i}", raw_text='{"answer":"未作正确性判断"}',
            prompt_sha256=f"same-prompt-{i}", evidence_count=20,
            truncated=False, valid_json=True) for i in range(46)]
        self.outputs(self.baseline, self.rows)
        self.outputs(self.out, self.rows)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False))

    def outputs(self, directory, rows):
        path = directory / "generated_native46.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        self.write(directory / "generation_receipt.json", dict(
            completed=True, output_sha256=experiment.deep.sha(path)))

    def test_valid_pair_never_invents_semantic_accuracy(self):
        experiment.paired_report(self.out)
        report = json.loads((self.out / "PAIRED_FINAL46_STATISTICS.json").read_text())
        self.assertEqual(report["count"], 46)
        self.assertIsNone(report["correctness_rate"])
        self.assertTrue(report["semantic_accuracy_not_yet_reviewed"])
        self.assertEqual(report["mean_teacher_forced_nll"],
                         dict(input_only=1.0, optimized_v2=0.9))

    def test_prompt_mismatch_rejected_before_comparison_artifact(self):
        rows = [dict(r) for r in self.rows]
        rows[30]["prompt_sha256"] = "different"
        self.outputs(self.out, rows)
        with self.assertRaisesRegex(RuntimeError, "Paired prompt mismatch"):
            experiment.paired_report(self.out)
        self.assertFalse((self.out / "paired_final46_for_content_review.jsonl").exists())

    def test_modified_output_without_new_receipt_is_rejected(self):
        path = self.out / "generated_native46.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "Generation SHA mismatch"):
            experiment.paired_report(self.out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
