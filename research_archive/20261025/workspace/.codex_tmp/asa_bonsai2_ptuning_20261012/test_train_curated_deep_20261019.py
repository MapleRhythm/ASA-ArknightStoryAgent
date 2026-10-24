import json
import unittest

import train_curated_deep_20261019 as t


class SelectionTests(unittest.TestCase):
    def test_real_selection_excludes_conflicts_and_uncertain_scope(self):
        rows = t.read_rows(t.ROOT / "READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl")
        snapshots = [json.loads((t.ROOT / n).read_text()) for n in t.SNAPSHOTS]
        selected = t.select(rows, snapshots)
        ids = {r["id"] for r in selected}
        self.assertEqual(len(selected), 367)
        for position in (55, 56, 229, 236, 200, 202, 203, 204, 205, 206, 330, 463):
            self.assertNotIn(rows[position - 1]["id"], ids)
        self.assertIn(rows[471]["id"], ids)
        self.assertTrue(all(r in rows for r in selected))

    def test_reject_changed_source_identity(self):
        rows = t.read_rows(t.ROOT / "READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl")
        snapshots = [json.loads((t.ROOT / n).read_text()) for n in t.SNAPSHOTS]
        snapshots[-1]["source_sha256"] = "invalid"
        with self.assertRaises(AssertionError):
            t.select(rows, snapshots)


if __name__ == "__main__":
    unittest.main()
