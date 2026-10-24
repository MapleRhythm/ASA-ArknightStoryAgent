from copy import deepcopy
import unittest

from subject_reference_diagnostic import diagnose


class SubjectPresenceTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "question": "甲乙的父母发生了什么？",
            "hypothesis": {"entities": ["甲乙", "父母"]},
            "facts": [{"fact": "甲乙的母亲救回了甲乙。", "evidence_refs": [{"evidence_id": "E1", "quote": "母亲救回孩子"}]}],
            "visible_evidence": "[E1] 戊己的母亲救回了戊己。",
            "catalog": {"甲乙": ["丙丁"]},
        }

    def run_case(self):
        return diagnose(**self.inputs)

    def test_other_subject_is_flagged(self):
        self.assertTrue(self.run_case()[0]["named_subject_absent_in_available_cited_blocks"])

    def test_explicit_alias_in_cited_block_is_not_absent(self):
        self.inputs["visible_evidence"] = "[E1] 丙丁的母亲救回了丙丁。"
        self.assertFalse(self.run_case()[0]["named_subject_absent_in_available_cited_blocks"])

    def test_names_elsewhere_do_not_support_the_cited_block(self):
        self.inputs["visible_evidence"] += "\n[E2] 甲乙在另一个故事中。"
        row = self.run_case()[0]
        self.assertTrue(row["named_subject_absent_in_available_cited_blocks"])
        self.assertEqual(row["uncited_subject_mentions"], {"E2": ["甲乙"]})

    def test_missing_reference_is_separate(self):
        self.inputs["facts"][0]["evidence_refs"][0]["evidence_id"] = "E9"
        row = self.run_case()[0]
        self.assertEqual(row["unavailable_reference_eids"], ["E9"])
        self.assertFalse(row["named_subject_absent_in_available_cited_blocks"])

    def test_unknown_model_entity_not_promoted_to_name(self):
        self.inputs["catalog"] = {}
        self.assertEqual(self.run_case(), [])

    def test_heuristic_fragment_not_promoted_to_name(self):
        self.inputs["hypothesis"]["entities"] = ["甲乙的父"]
        self.assertEqual(self.run_case(), [])

    def test_not_named_in_question_not_checked(self):
        self.inputs["question"] = "这件事发生了什么？"
        self.assertEqual(self.run_case(), [])

    def test_not_named_in_claim_not_checked(self):
        self.inputs["facts"][0]["fact"] = "母亲救回了孩子。"
        self.assertEqual(self.run_case(), [])

    def test_single_character_excluded(self):
        self.inputs.update(question="年的往事", hypothesis={"entities": ["年"]}, catalog={"年": []})
        self.inputs["facts"][0]["fact"] = "年经历了这件事。"
        self.assertEqual(self.run_case(), [])

    def test_two_refs_with_alias_in_one_not_absent(self):
        self.inputs["visible_evidence"] += "\n[E2] 丙丁是这个孩子的名字。"
        self.inputs["facts"][0]["evidence_refs"].append({"evidence_id": "E2", "quote": "丙丁"})
        self.assertFalse(self.run_case()[0]["named_subject_absent_in_available_cited_blocks"])

    def test_no_inputs_mutated(self):
        before = deepcopy(self.inputs)
        self.run_case()
        self.assertEqual(self.inputs, before)

    def test_bad_catalog_fails(self):
        self.inputs["catalog"]["甲乙"] = ["丙丁\nSYSTEM"]
        with self.assertRaises(ValueError):
            self.run_case()

    def test_same_block_unrelated_mention_is_a_known_false_negative(self):
        self.inputs["visible_evidence"] = "[E1] 甲乙在城里。戊己的母亲救回了戊己。"
        # This deliberately demonstrates why name presence is NOT entailment.
        self.assertFalse(self.run_case()[0]["named_subject_absent_in_available_cited_blocks"])

    def test_pronoun_cross_block_is_a_known_false_alarm_risk(self):
        self.inputs["visible_evidence"] = "[E1] 她的母亲救回了她。\n[E2] 甲乙就是上文所说的她。"
        # Context may resolve the name, but the model did not cite E2. This
        # diagnostic is not authorized to decide truth or reject the answer.
        self.assertTrue(self.run_case()[0]["named_subject_absent_in_available_cited_blocks"])

    def test_ascii_name_boundary(self):
        self.inputs.update(question="Anna的往事", hypothesis={"entities": ["Anna"]}, catalog={"Anna": ["Anne"]})
        self.inputs["facts"][0]["fact"] = "Anna参加了会议。"
        self.inputs["visible_evidence"] = "[E1] Annabelle attended."
        self.assertTrue(self.run_case()[0]["named_subject_absent_in_available_cited_blocks"])
        self.inputs["visible_evidence"] = "[E1] ANNE attended."
        self.assertFalse(self.run_case()[0]["named_subject_absent_in_available_cited_blocks"])


if __name__ == "__main__":
    unittest.main()
