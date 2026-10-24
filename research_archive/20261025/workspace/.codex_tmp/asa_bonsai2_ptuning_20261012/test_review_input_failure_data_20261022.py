import copy
import unittest
from unittest.mock import patch

import review_input_failure_data_20261022 as review


class Tests(unittest.TestCase):
    def test_full_integrity_and_explicit_semantic_scope(self):
        result = review.audit()
        self.assertTrue(result["integrity_passed"])
        self.assertEqual(result["structural_rows_checked"], 180)
        self.assertEqual(result["sample_state_judgments"],
                         {"accepted:pass": 13, "accepted:fail": 2, "quarantined:fail": 5})
        self.assertEqual(result["exported_not_semantically_reviewed"], 156)
        self.assertFalse(result["all180_semantically_reviewed"])
        self.assertFalse(result["ready_for_training"])
        self.assertFalse(result["remote_dataset_modified"])
        self.assertEqual(result["new_api_calls"], 0)

    def test_full20_review_cannot_silently_shrink_to_citations(self):
        original_read = review.read
        decisions = copy.deepcopy(original_read(review.DECISIONS))
        case = next(c for c in decisions["cases"] if c["position"] == 129)
        case["read_eids"] = case["source_eids"]
        with patch.object(review, "read",
                          side_effect=lambda path: decisions if path == review.DECISIONS else original_read(path)):
            with self.assertRaises(AssertionError):
                review.audit()

    def test_duplicate_review_does_not_inflate_coverage(self):
        original_read = review.read
        decisions = copy.deepcopy(original_read(review.DECISIONS))
        decisions["cases"].append(copy.deepcopy(decisions["cases"][0]))
        with patch.object(review, "read",
                          side_effect=lambda path: decisions if path == review.DECISIONS else original_read(path)):
            with self.assertRaises(AssertionError):
                review.audit()


if __name__ == "__main__":
    unittest.main()
