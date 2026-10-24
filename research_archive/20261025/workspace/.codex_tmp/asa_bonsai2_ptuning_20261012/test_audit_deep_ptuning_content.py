"""Regression checks for read-only aggregation, identities and score separation."""
import copy
import unittest
from unittest.mock import patch

import audit_deep_ptuning_content as audit


class AuditTests(unittest.TestCase):
    def test_real_artifacts_and_counts(self):
        result = audit.aggregate()
        self.assertTrue(result["integrity_passed"])
        counts = result["summary"]["bonsai_deep750"]["all46"]["categories"]
        self.assertEqual(counts, dict(complete=22, partial=16, wrong=8, no_answer=0))
        self.assertEqual(result["spotcheck"]["failed_positions"], [11, 12, 15])
        self.assertFalse(result["threshold60_met"])
        # All valid JSON answers do not imply all semantically correct answers.
        self.assertEqual(result["generation_separate_from_semantic_accuracy"]["parsed_json"], 46)
        self.assertEqual(result["paired"]["bonsai_shallow750"]["gained_deep"]["n"], 2)
        self.assertEqual(result["paired"]["bonsai_shallow750"]["lost_deep"]["n"], 5)

    def expect_row_rejection(self, name, mutate):
        original = audit.rows

        def changed(path):
            values = original(path)
            if path.name == name:
                values = copy.deepcopy(values)
                mutate(values)
            return values

        with patch.object(audit, "rows", side_effect=changed), self.assertRaises(AssertionError):
            audit.aggregate()

    def test_duplicate_output_id_rejected(self):
        self.expect_row_rejection(
            "generated_native46.jsonl",
            lambda values: values[1].update(id=values[0]["id"]),
        )

    def test_prompt_mismatch_rejected(self):
        self.expect_row_rejection(
            "generated_native46.jsonl",
            lambda values: values[0].update(prompt_sha256="0" * 64),
        )

    def test_truncated_input_rejected(self):
        self.expect_row_rejection(
            "generated_native46.jsonl",
            lambda values: values[0].update(truncated=True),
        )

    def test_raw_and_parsed_disagreement_rejected(self):
        self.expect_row_rejection(
            "generated_native46.jsonl",
            lambda values: values[0]["parsed"].update(next_action="abstain"),
        )

    def test_evidence_tampering_rejected(self):
        self.expect_row_rejection(
            "spotcheck_requests.jsonl",
            lambda values: values[0]["documents"][0].update(text="modified"),
        )

    def test_pending_semantic_review_rejected(self):
        original = audit.read_json

        def changed(path):
            value = original(path)
            if path.name == "DEEP_PTUNING_CONTENT_REVIEW_20261017.json":
                value["semantic_adjudication_complete"] = False
            return value

        with patch.object(audit, "read_json", side_effect=changed), self.assertRaises(AssertionError):
            audit.aggregate()

    def test_offline_format_check_is_not_content_judgment(self):
        fabricated = {
            "next_action": "answer_directly",
            "supported_facts": [{"fact": "This statement is fabricated.", "evidence_ids": ["E1"]}],
        }
        self.assertTrue(audit.structure_valid(fabricated))
        fabricated["supported_facts"][0]["evidence_ids"] = ["E99"]
        self.assertFalse(audit.structure_valid(fabricated))

    def test_candidate_promotion_rejected(self):
        original = audit.read_json

        def changed(path):
            value = original(path)
            if path.name == "manifest.json" and path.parent.name == "merged_candidate":
                value["ready_for_training"] = True
            return value

        with patch.object(audit, "read_json", side_effect=changed), self.assertRaises(AssertionError):
            audit.aggregate()


if __name__ == "__main__":
    unittest.main()
