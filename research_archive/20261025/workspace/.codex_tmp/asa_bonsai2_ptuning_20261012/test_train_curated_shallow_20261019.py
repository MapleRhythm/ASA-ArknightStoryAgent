import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_curated_shallow_20261019 as s


class DataCopyTests(unittest.TestCase):
    def test_data_copy_is_byte_identical_and_original_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "train.jsonl").write_text('{"train":1}\n')
            (source / "val.jsonl").write_text('{"val":1}\n')
            (source / "review_snapshot").mkdir()
            (source / "review_snapshot/a.json").write_text("{}")
            original = dict(kind="test", ready_for_training=True)
            target = root / "copy"
            with patch.object(s.data_audit, "verify", return_value=original) as audit:
                s.prepare(target, source)
            self.assertEqual(audit.call_count, 2)
            for name in ("train.jsonl", "val.jsonl", "review_snapshot/a.json"):
                self.assertEqual((source / name).read_bytes(), (target / name).read_bytes())
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["evidence_count_per_example"], 20)
            self.assertEqual(manifest["truncated_documents"], 0)
            self.assertFalse(manifest["review_required_before_training"])
            self.assertNotIn("evidence_count_per_example", original)
            with patch.object(s.data_audit, "verify", return_value=original):
                with self.assertRaises(FileExistsError):
                    s.prepare(target, source)


if __name__ == "__main__":
    unittest.main()
