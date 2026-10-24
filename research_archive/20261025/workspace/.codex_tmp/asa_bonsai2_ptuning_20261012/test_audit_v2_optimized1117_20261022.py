import unittest

import audit_v2_optimized1117_20261022 as a


class Tests(unittest.TestCase):
    def test_transitions(self):
        self.assertEqual(a.transition("partial", "complete"), "gained_complete")
        self.assertEqual(a.transition("complete", "partial"), "lost_complete")
        self.assertEqual(a.transition("complete", "complete"), "both_complete")
        self.assertEqual(a.transition("wrong", "partial"), "neither_complete")

    def test_bound_review_and_counts(self):
        result = a.audit()
        self.assertTrue(result["integrity_passed"])
        self.assertEqual(len(result["cases"]), 46)
        self.assertEqual(result["optimized_v2"]["categories"],
                         dict(complete=15, partial=23, wrong=8, no_answer=0))
        self.assertEqual(result["input_only_baseline"]["categories"],
                         dict(complete=24, partial=13, wrong=9, no_answer=0))
        self.assertEqual(result["optimized_v2"]["core_complete"], 33)
        self.assertEqual(result["optimized_v2"]["valid_json"], 16)
        self.assertFalse(result["format_used_for_content_grading"])
        self.assertTrue(result["content_complete_despite_invalid_json"])
        self.assertEqual(sorted(sum(result["paired_complete_transitions"].values(), [])), list(range(1, 47)))
        for case in result["cases"]:
            self.assertEqual(len(case["output_raw_sha256"]), 64)
            if case["all20_checked_for_missing_link"]:
                self.assertEqual(len(case["source_eids"]), 20)


if __name__ == "__main__":
    unittest.main()
