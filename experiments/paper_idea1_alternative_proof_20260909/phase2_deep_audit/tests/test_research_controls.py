import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import research_controls as rc


class ControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = rc.load(rc.ROOT / "audit_spec.json")
        cls.natural, cls.npairs = rc.build_natural(cls.spec)
        cls.synthetic, cls.spairs, cls.truth_table = rc.build_synthetic()
        cls.old = rc.load(rc.PILOT / "data/pilot_matrix.json")[0]
        cls.pred = rc.load(rc.PILOT / "runs/nonthinking_vllm/clean_sft.predictions.json")[0]

    def test_frozen_hashes_and_all_old_outputs(self):
        audit = rc.audit_existing(self.spec)
        self.assertEqual(len(audit["details"]), 56)
        self.assertTrue(audit["all_56_prediction_prompts_exactly_bound"])

    def test_prompt_tamper_is_rejected(self):
        altered = copy.deepcopy(self.pred)
        altered["conversations"][0]["value"] += "tampered"
        with self.assertRaises(ValueError):
            rc.verify_prediction(self.old, altered)

    def test_system_tamper_is_rejected(self):
        altered = copy.deepcopy(self.pred)
        altered["system"] += "tampered"
        with self.assertRaises(ValueError):
            rc.verify_prediction(self.old, altered)

    def test_output_mirror_tamper_is_rejected(self):
        altered = copy.deepcopy(self.pred)
        altered["raw_output"] += " "
        with self.assertRaises(ValueError):
            rc.verify_prediction(self.old, altered)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            rc.index_unique([self.old, self.old])

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"next_action": "abstain", "next_action": "answer_directly"}',
                       object_pairs_hook=rc.no_duplicate_keys)

    def test_unknown_never_becomes_a_training_target(self):
        unknown = [r for r in self.natural if r["diagnostic"]["reference"]["status"] == "unknown"]
        self.assertEqual(len(unknown), 32)
        self.assertTrue(all(r["diagnostic"]["reference"]["action"] is None for r in unknown))

    def test_no_reference_in_model_messages(self):
        for row in self.natural + self.synthetic:
            self.assertEqual(len(row["conversations"]), 1)
            self.assertEqual(row["conversations"][0]["from"], "human")
            self.assertNotIn("visible_graph", row["conversations"][0]["value"])
            self.assertNotIn("minimum_edge_repairs", row["conversations"][0]["value"])

    def test_round_normalizer_does_not_mask_evidence(self):
        a = "round: 1/2\nevidence: 1/2"
        b = "round: 2/2\nevidence: 1/2"
        self.assertEqual(rc.normalize_round(a), rc.normalize_round(b))
        self.assertIn("evidence: 1/2", rc.normalize_round(a))

    def test_round_normalizer_rejects_multiple_fields(self):
        with self.assertRaises(ValueError):
            rc.normalize_round("round: 1/2\nround: 2/2")

    def test_all_control_pair_differences(self):
        for rows, pairs in ((self.natural, self.npairs), (self.synthetic, self.spairs)):
            indexed = rc.index_unique(rows)
            for p in pairs:
                self.assertEqual(rc.changed_fields(indexed[p["left"]], indexed[p["right"]]),
                                 p["changed_fields"])

    def test_uncontrolled_factor_is_rejected(self):
        a = self.natural[0]
        b = copy.deepcopy(a)
        b["diagnostic"]["input_fields"]["round"] = "3/3"
        b["diagnostic"]["input_fields"]["question"] = "different question"
        with self.assertRaises(ValueError):
            rc.pair(a, b, "round", ["round"])

    def test_truth_table_not_unique_gold_id(self):
        self.assertEqual(sum(rc.truth(m) for m in range(16)), 7)
        self.assertTrue(rc.truth(14))  # e1 missing but e3 AND e4 remain.
        self.assertFalse(rc.truth(10))  # e1/e3 missing; both proofs broken.
        self.assertTrue(rc.truth(14))  # restoring e3 repairs this state.

    def test_repairs_are_valid_and_minimal(self):
        for m in range(16):
            repairs = rc.minimum_repairs(m)
            for repair in repairs:
                self.assertTrue(rc.truth(m | sum(1 << i for i in repair)))
                for subset_size in range(len(repair)):
                    for subset in rc.itertools.combinations(repair, subset_size):
                        self.assertFalse(rc.truth(m | sum(1 << i for i in subset)))

    def test_reachability_requires_connected_identity_bridge(self):
        edges = [
            {"id": "a", "subject": "q", "relation": "routes_to", "object": "station_a"},
            {"id": "b", "subject": "station_b", "relation": "has_code", "object": "answer"}]
        self.assertEqual(rc.graph_paths(edges, "q"), [])

    def test_citations_require_a_complete_path(self):
        for row in self.synthetic:
            ref = row["diagnostic"]["reference"]
            visible = set(row["diagnostic"]["input_fields"]["evidence_ids"])
            self.assertEqual(bool(ref["all_valid_citation_pairs"]), ref["status"] == "sufficient")
            for proof in ref["all_valid_citation_pairs"]:
                self.assertEqual(len(set(proof)), 2)
                self.assertTrue(set(proof) <= visible)

    def test_synthetic_length_count_and_answer_presence_match(self):
        worlds = {}
        for row in self.synthetic:
            diag = row["diagnostic"]
            docs = diag["input_fields"]["documents"]
            self.assertEqual(len(docs), 6)
            record = (tuple(len(d["text"]) for d in docs),
                      tuple("\n".join(d["text"] for d in docs).count(a)
                            for a in diag["reference"]["candidate_answers"]))
            worlds.setdefault(diag["case"], set()).add(record)
        self.assertTrue(all(len(values) == 1 for values in worlds.values()))

    def test_builders_deterministic(self):
        n, np = rc.build_natural(self.spec)
        s, sp, t = rc.build_synthetic()
        self.assertEqual((n, np), (self.natural, self.npairs))
        self.assertEqual((s, sp, t), (self.synthetic, self.spairs, self.truth_table))

    def test_all_control_ids_and_hashes(self):
        rc.index_unique(self.natural + self.synthetic)
        for row in self.natural + self.synthetic:
            self.assertEqual(rc.digest(rc.messages(row)), row["diagnostic"]["prompt_messages_sha256"])
            self.assertEqual(rc.digest(row["diagnostic"]["reference"]), row["diagnostic"]["reference_sha256"])

    def test_counterexample_merge_can_lose_new_terms(self):
        from replay_query_processing import saturation_demo
        example = saturation_demo()
        self.assertFalse(example["new_entity_survives"])
        self.assertFalse(example["new_keyword_survives"])

    def test_frozen_sword_query_loses_named_bridge_after_merge(self):
        from replay_query_processing import (
            make_hypothesis, normalize_hypothesis_payload, merge_hypotheses,
            build_follow_up_hypothesis_queries,
        )
        case = rc.load(rc.PILOT / "input/frozen_cases.json")["cases"][0]
        row = next(r for r in rc.load(rc.PILOT / "runs/nonthinking_vllm/qwen35_4b_base.predictions.json")
                   if r["diagnostic"]["short_name"] == "sword_identity"
                   and r["diagnostic"]["condition"] == "broken_round1")
        base = make_hypothesis(case["hypothesis"])
        normalized = normalize_hypothesis_payload(
            rc.load_payload(row)["follow_up_hypothesis"], question=case["question"],
            dialogue_context="", current_intent=base.intent)
        before = build_follow_up_hypothesis_queries(case["question"], normalized)
        after = build_follow_up_hypothesis_queries(case["question"], merge_hypotheses(base, normalized))
        self.assertIn("郑愁", "\n".join(before))
        self.assertNotIn("郑愁", "\n".join(after))


if __name__ == "__main__":
    unittest.main()
