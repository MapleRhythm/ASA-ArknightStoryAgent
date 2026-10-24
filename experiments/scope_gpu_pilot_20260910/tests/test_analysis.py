import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import analyze_results as a


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = a.load(a.ROOT / "input/smoke4.json")
        cls.predictions = a.load(a.ROOT / "runs/smoke_base_v1/predictions.json")

    def test_actual_smoke_bindings(self):
        for source, pred in zip(self.sources, self.predictions, strict=True):
            payload, error = a.checked_prediction(source, pred)
            self.assertIsNone(error)
            self.assertIsInstance(payload, dict)

    def test_prompt_tamper_rejected(self):
        pred = copy.deepcopy(self.predictions[0])
        pred["conversations"][0]["value"] += "污染"
        with self.assertRaises(ValueError):
            a.checked_prediction(self.sources[0], pred)

    def test_output_tamper_rejected(self):
        pred = copy.deepcopy(self.predictions[0])
        pred["raw_output"] += " "
        with self.assertRaises(ValueError):
            a.checked_prediction(self.sources[0], pred)

    def test_reference_tamper_rejected(self):
        pred = copy.deepcopy(self.predictions[0])
        pred["diagnostic"]["reference"]["action"] = "abstain"
        with self.assertRaises(ValueError):
            a.checked_prediction(self.sources[0], pred)

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            a.no_duplicates([("next_action", "abstain"), ("next_action", "answer_directly")])

    def test_normalizer_does_not_change_source(self):
        source = copy.deepcopy(self.sources[1])
        original = copy.deepcopy(source)
        payload, _ = a.checked_prediction(source, self.predictions[1])
        a.normalize_result(source, payload)
        self.assertEqual(source, original)

    def test_split_facts_are_not_automatically_false(self):
        source = {"diagnostic": {"suite": "synthetic_controls", "reference": {
            "status": "sufficient", "answer": "ABC", "all_valid_citation_pairs": [["E1", "E2"]]}}}
        payload = {"supported_facts": [
            {"fact": "箱子指向站点。", "evidence_ids": ["E1"]},
            {"fact": "站点口令ABC。", "evidence_ids": ["E2"]}]}
        proxy = a.symbolic_proxy(source, payload)
        self.assertTrue(proxy["answer_token_and_citation_path"])
        self.assertFalse(proxy["single_fact_contains_complete_path"])
        self.assertNotIn("semantic_correct", proxy)

    def test_wrong_path_fails_proxy(self):
        source = {"diagnostic": {"suite": "synthetic_controls", "reference": {
            "status": "sufficient", "answer": "ABC", "all_valid_citation_pairs": [["E1", "E2"]]}}}
        payload = {"supported_facts": [{"fact": "口令ABC", "evidence_ids": ["E3", "E4"]}]}
        self.assertFalse(a.symbolic_proxy(source, payload)["answer_token_and_citation_path"])


if __name__ == "__main__":
    unittest.main()
