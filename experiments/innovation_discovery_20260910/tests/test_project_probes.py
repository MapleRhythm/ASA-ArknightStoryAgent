import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_project_mechanisms as p


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = p.synthetic_scope_rows() + p.natural_scope_rows()
        cls.replay = p.replays()
        cls.probe = p.lexical_probe(cls.rows)

    def test_row_ids_unique(self):
        self.assertEqual(len(self.rows), len({r["id"] for r in self.rows}))

    def test_generated_counts_not_independent_question_count(self):
        self.assertEqual(sum(r["kind"] == "synthetic" for r in self.rows), 72)
        self.assertEqual(sum(r["kind"] == "natural_development" for r in self.rows), 12)
        self.assertEqual(len({r["family"] for r in self.rows if r["kind"] == "synthetic"}), 6)

    def test_all_quotes_are_visible_and_not_truncated(self):
        for row in self.rows:
            self.assertIn(row["quote"], row["source_text"])
            self.assertLessEqual(len(row["quote"]), 80)

    def test_all_source_hashes_match(self):
        for row in self.rows:
            self.assertEqual(row["source_sha256"], p.sha(row["source_text"]))

    def test_no_probe_is_training_gold(self):
        self.assertTrue(all(r["exclude_from_training"] for r in self.rows))
        self.assertTrue(all(r["label_authority"] for r in self.rows))

    def test_three_labels_balanced_within_synthetic_world_type(self):
        grouped = {}
        for row in self.rows:
            if row["kind"] == "synthetic":
                grouped.setdefault(row["id"].rsplit("_", 1)[0], set()).add(row["reference_label"])
        self.assertTrue(all(labels == {"supported", "contradicted", "not_established"} for labels in grouped.values()))

    def test_low_level_probe_does_not_mutate_data(self):
        before = copy.deepcopy(self.rows)
        p.lexical_probe(self.rows)
        self.assertEqual(self.rows, before)

    def test_all_query_replay_rows(self):
        self.assertEqual(len(self.replay["records"]), 12)
        self.assertTrue(all(r["no_retrieval_or_model_execution"] for r in self.replay["records"]))

    def test_aggregate_query_must_not_be_omitted(self):
        row = next(r for r in self.replay["records"] if r["reader"] == "qwen35_4b_base"
                   and r["case"] == "sword_identity")
        self.assertNotIn("郑愁", "\n".join(row["followup_branch_queries"]))
        self.assertIn("郑愁", row["aggregate_query"])
        self.assertIn("渡魂剑", row["aggregate_query"])
        self.assertNotIn("郑愁", row["new_terms_absent_all_pending"])
        self.assertEqual(row["all_pending_queries"][0], row["aggregate_query"])

    def test_branch_collision_does_not_imply_full_action_collision(self):
        sword = [r for r in self.replay["records"] if r["case"] == "sword_identity"]
        self.assertEqual(len({p.sha(r["followup_branch_queries"]) for r in sword}), 1)
        self.assertEqual(len({p.sha(r["all_pending_queries"]) for r in sword}), 3)

    def test_scheduler_does_not_invent_queries(self):
        for row in self.replay["records"]:
            self.assertTrue(set(row["hypothetical_scheduled_queries"]) <= set(row["all_pending_queries"]))
            self.assertIn("not the historic", row["scheduler_note"])

    def test_query_lists_are_not_changed_by_label_probe(self):
        self.assertEqual(p.replays(), self.replay)

    def test_two_validator_records_for_every_fixture(self):
        self.assertEqual(len(self.probe["details"]), 2 * len(self.rows))
        self.assertEqual({r["mode"] for r in self.probe["details"]}, {"quote", "evidence_id"})

    def test_negation_counterexample_is_recorded_not_hidden(self):
        item = next(r for r in self.probe["details"] if r["id"] == "synthetic_0_negation_1"
                    and r["mode"] == "evidence_id")
        self.assertEqual(item["reference_label"], "contradicted")
        self.assertTrue(item["no_hard_issue"])

    def test_correct_paraphrase_can_be_rejected(self):
        item = next(r for r in self.probe["details"] if r["id"] == "natural_reported_speaker_0"
                    and r["mode"] == "evidence_id")
        self.assertEqual(item["reference_label"], "supported")
        self.assertFalse(item["no_hard_issue"])


if __name__ == "__main__":
    unittest.main()
