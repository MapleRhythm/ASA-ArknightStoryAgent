import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_inputs as b


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = b.scope_rows()

    def test_counts(self):
        self.assertEqual(len(self.rows), 240)
        self.assertEqual(len({row["diagnostic"]["family"] for row in self.rows}), 6)
        self.assertEqual(len({row["id"] for row in self.rows}), 240)

    def test_eval_only(self):
        for row in self.rows:
            self.assertTrue(row["diagnostic"]["reference"]["exclude_from_training"])
            self.assertEqual(len(row["conversations"]), 1)

    def test_rendering_no_reference_leak(self):
        for row in self.rows:
            diag = row["diagnostic"]
            fields = diag["input_fields"]
            expected = b.render_exx_user_prompt(
                question=fields["question"], hypothesis=fields["hypothesis"],
                round_value=fields["round"],
                evidence_text="\n".join(f"[E{i}]\n{doc['text']}" for i, doc in enumerate(fields["documents"], 1)))
            self.assertEqual(row["conversations"][0]["value"], expected)
            self.assertEqual(diag["reference_sha256"], b.sha(diag["reference"]))

    def test_negation_is_not_automatic_abstention(self):
        rows = [r for r in self.rows if r["diagnostic"]["family"] == "negative_approval"
                and r["diagnostic"]["factors"]["state"] in {"proof_a", "proof_b", "dual"}]
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(r["diagnostic"]["reference"]["action"] == "answer_directly" for r in rows))

    def test_missing_actions(self):
        rows = [r for r in self.rows if r["diagnostic"]["factors"]["state"] == "scope_missing"]
        self.assertEqual(Counter(r["diagnostic"]["reference"]["action"] for r in rows),
                         {"retrieve_more": 24, "abstain": 24})

    def test_sources_and_ids(self):
        for row in self.rows:
            diag = row["diagnostic"]
            for doc in diag["input_fields"]["documents"]:
                self.assertEqual(doc["text_sha256"], b.sha(doc["text"]))
            for ids in diag["reference"]["valid_support_sets"]:
                self.assertTrue(set(ids) <= {"E1", "E2"})

    def test_round_only(self):
        grouped = {}
        for row in self.rows:
            grouped.setdefault(row["id"].rsplit("_", 1)[0], []).append(row)
        for pair in grouped.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]["conversations"][0]["value"].replace("round: 1/2", "round: 2/2"),
                             pair[1]["conversations"][0]["value"])

    def test_rebuild_deterministic(self):
        self.assertEqual(self.rows, b.scope_rows())


if __name__ == "__main__":
    unittest.main()
