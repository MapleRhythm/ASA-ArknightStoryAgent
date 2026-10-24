"""Regression tests for review bookkeeping, not independent semantic grading."""
import copy
import json
import unittest
from unittest.mock import patch

import review_reader_errors_individually as r


class ReviewTests(unittest.TestCase):
    def changed_review(self, change):
        path = r.ROOT / r.REVIEW
        value = copy.deepcopy(json.loads(path.read_text()))
        change(value)
        original = r.Path.read_text

        def read_text(instance, *args, **kwargs):
            if instance == path:
                return json.dumps(value, ensure_ascii=False)
            return original(instance, *args, **kwargs)

        return patch.object(r.Path, "read_text", read_text)

    def test_real_coverage_is_partial(self):
        result = r.audit()
        self.assertEqual(result["reviewed"], 303)
        self.assertEqual(result["pending"], 183)
        self.assertEqual(result["judgments"], {"pass": 275, "fail": 24, "needs_review": 4})
        self.assertEqual(result["failed_positions"], [10, 15, 55, 56, 67, 85, 95, 107, 124, 126, 129, 135, 137, 144, 149, 152, 177, 212, 229, 236, 238, 269, 292, 463])
        self.assertEqual(result["next_pending_positions"][0], 301)
        self.assertFalse(result["all486_semantically_reviewed"])
        self.assertFalse(result["ready_for_training"])
        self.assertEqual(sum(sum(v.values()) for v in result["by_kind"].values()), 486)

    def test_first300_covered_and_history_not_double_counted(self):
        rows = r.load()
        previous = json.loads((r.ROOT / r.PREVIOUS).read_text())["cases"]
        current = json.loads((r.ROOT / r.REVIEW).read_text())["cases"]
        ids = [case["id"] for case in previous + current]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({row["id"] for row in rows[:300]} <= set(ids))
        self.assertEqual(
            [i for i, row in enumerate(rows, 1) if i > 300 and row["id"] in ids],
            [324, 335, 463],
        )

    def test_duplicate_decision_rejected(self):
        with self.changed_review(lambda v: v["cases"].append(v["cases"][0])):
            with self.assertRaisesRegex(AssertionError, "duplicate review"):
                r.audit()

    def test_overstated_completion_rejected(self):
        with self.changed_review(lambda v: v.update(all486_semantically_reviewed=True)):
            with self.assertRaises(AssertionError):
                r.audit()

    def test_training_release_rejected(self):
        with self.changed_review(lambda v: v.update(ready_for_training=True)):
            with self.assertRaises(AssertionError):
                r.audit()

    def test_incomplete_all20_read_claim_rejected(self):
        with self.changed_review(lambda v: v["cases"][0]["read_eids"].pop()):
            with self.assertRaises(AssertionError):
                r.audit()

    def test_position_mismatch_rejected(self):
        with self.changed_review(lambda v: v["cases"][0].update(position=2)):
            with self.assertRaises(AssertionError):
                r.audit()

    def test_failure_without_repair_direction_rejected(self):
        def change(value):
            next(c for c in value["cases"] if c["judgment"] == "fail").pop("repair_direction")
        with self.changed_review(change):
            with self.assertRaises(KeyError):
                r.audit()

    def test_ambiguous_case_without_resolution_rejected(self):
        def change(value):
            next(c for c in value["cases"] if c["judgment"] == "needs_review").pop("resolution_needed")
        with self.changed_review(change):
            with self.assertRaises(KeyError):
                r.audit()

    def test_ambiguous_case_with_empty_resolution_rejected(self):
        def change(value):
            next(c for c in value["cases"] if c["judgment"] == "needs_review")["resolution_needed"] = ""
        with self.changed_review(change):
            with self.assertRaisesRegex(AssertionError, "explicit resolution"):
                r.audit()


if __name__ == "__main__":
    unittest.main()
