"""Integrity/provenance tests, not independent semantic certification."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_reader_error_repairs_20261018 as b
import review_reader_errors_individually as r


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = r.load()
        cls.files = b.prepare()
        cls.revised = [json.loads(line) for line in cls.files["train.jsonl"].splitlines()]

    def test_exactly_approved_failures_changed(self):
        changed = [i for i, (x, y) in enumerate(zip(self.original, self.revised), 1) if x != y]
        self.assertEqual(changed, r.audit()["failed_positions"])
        self.assertEqual(len(self.revised), 486)
        self.assertEqual(len({x["id"] for x in self.revised}), 486)

    def test_questions_prompts_full_evidence_and_split_unchanged(self):
        for old, new in zip(self.original, self.revised):
            self.assertEqual(
                {k: v for k, v in old.items() if k not in b.MUTABLE_FIELDS},
                {k: v for k, v in new.items() if k not in b.MUTABLE_FIELDS},
            )

    def test_ambiguous_labels_remain_untouched(self):
        for position in (30, 34, 76, 77):
            self.assertEqual(self.original[position - 1], self.revised[position - 1])

    def test_corrected_labels_not_misattributed_to_glm_or_gold(self):
        for row in self.revised:
            if "label_revision" not in row:
                continue
            self.assertEqual(row["annotation_model"], "codex_assistant_evidence_repair")
            self.assertIsNone(row["annotation_response_id"])
            self.assertIsNone(row["semantic_review"]["label_supported"])
            self.assertFalse(row["semantic_review_sampled"])
            self.assertFalse(row["label_revision"]["ready_for_training"])
            self.assertEqual(row["label_revision"]["original_metadata"]["annotation_model"], "glm-5.3")

    def test_schema_and_citations(self):
        for repair in b.REPAIRS:
            b.validate_target(repair["target"], self.original[repair["position"] - 1])

    def test_bad_citation_rejected(self):
        target = copy.deepcopy(b.REPAIRS[0]["target"])
        target["supported_facts"][0]["evidence_ids"] = ["E99"]
        with self.assertRaises(AssertionError):
            b.validate_target(target, self.original[9])

    def test_unknown_fields_and_hybrid_actions_rejected(self):
        target = copy.deepcopy(b.REPAIRS[0]["target"])
        target["follow_up_hypothesis"] = {"question": "unused"}
        with self.assertRaises(AssertionError):
            b.validate_target(target, self.original[9])

    def test_unapproved_position_rejected(self):
        invalid = copy.deepcopy(b.REPAIRS)
        invalid[0]["position"] = 30
        with patch.object(b, "REPAIRS", invalid):
            with self.assertRaises(AssertionError):
                b.prepare()

    def test_manifest_keeps_training_blocked(self):
        manifest = json.loads(self.files["manifest.json"])
        for key in ("ready_for_training", "all486_semantically_reviewed",
                    "revised_labels_independently_reviewed", "production_modified",
                    "running_training_dataset_changed", "teacher_api_called"):
            self.assertFalse(manifest[key], key)
        self.assertEqual(manifest["repaired"], 20)
        self.assertEqual(manifest["unchanged"], 466)

    def test_candidate_reproducible_across_hash_seeds(self):
        command = (
            "import json; import build_reader_error_repairs_20261018 as b; "
            "print(json.dumps({k:b.digest(v) for k,v in b.prepare().items()},sort_keys=True))"
        )
        expected = {k: b.digest(v) for k, v in self.files.items()}
        for seed in ("1", "2"):
            result = subprocess.run(
                [sys.executable, "-c", command], cwd=r.ROOT,
                env={**os.environ, "PYTHONHASHSEED": seed},
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(json.loads(result.stdout), expected)

    def test_generated_files_verify_and_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="asa_label_repairs_test_") as tmp:
            output = Path(tmp) / "candidate"
            b.build(output)
            self.assertEqual(b.verify(output)["repaired"], 20)
            with self.assertRaisesRegex(AssertionError, "output exists"):
                b.build(output)
            (output / "train.jsonl").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(AssertionError, "candidate mismatch"):
                b.verify(output)


if __name__ == "__main__":
    unittest.main()
