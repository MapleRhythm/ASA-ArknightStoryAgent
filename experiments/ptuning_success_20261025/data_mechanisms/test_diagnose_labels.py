"""Abstract fixtures only: no original plot, prompt or answer text."""
import json
from pathlib import Path
import tempfile
import unittest

from diagnose_labels import inspect, quantiles, run


def row(i="abstract-1", facts=None, kind="minimal_complete", action="answer_directly"):
    facts = facts or [("甲已交付。", ["E1"]), ("乙已接收。", ["E1"]),
                      ("后来另有人复述交付。", ["E2"]), ("后来另有人复述交付。", ["E2"])]
    target = {"next_action": action}
    if action == "answer_directly":
        target["supported_facts"] = [{"fact": text, "evidence_ids": refs} for text, refs in facts]
    else:
        target["reason"] = "局部未知。"
    return {"id": i, "question": "甲给谁交付了什么？", "round": "2/2",
            "targeted_kind": kind, "target": json.dumps(target, ensure_ascii=False),
            "documents": [{"evidence_id": "E1", "document_id": "fixture/scene1#chunk-0000",
                           "text": "甲向乙交付物品。"},
                          {"evidence_id": "E2", "document_id": "fixture/scene2#chunk-0000",
                           "text": "后来另一人谈到相同交付。"}]}


class TestDiagnostics(unittest.TestCase):
    def test_repeated_facts_get_distinct_offsets(self):
        source = row()
        result = inspect(source, 1)
        self.assertEqual(result["fact_count"], 4)
        spans = result["fact_spans"]
        self.assertLess(spans[2]["target_char_end"], spans[3]["target_char_start"])
        for span, fact in zip(spans, json.loads(source["target"])["supported_facts"]):
            self.assertEqual(json.loads(source["target"][span["target_char_start"]:
                                       span["target_char_end"]]), fact["fact"])
        self.assertIn("multi_stage_citations", result["flags_for_review_not_errors"])
        self.assertNotIn("semantic_error", result)

    def test_ascii_escaped_label_supported(self):
        source = row(facts=[('甲说“是\\否”。', ["E1"])])
        source["target"] = json.dumps(json.loads(source["target"]), ensure_ascii=True)
        self.assertEqual(inspect(source, 1)["fact_count"], 1)

    def test_partial_action_is_review_signal_not_error(self):
        result = inspect(row(kind="partial_support", action="abstain"), 1)
        self.assertIn("partial_action_review", result["flags_for_review_not_errors"])
        self.assertEqual(result["fact_count"], 0)

    def test_five_required_facts_never_auto_deleted(self):
        source = row(facts=[(f"子问{i}的必要结论。", ["E1"]) for i in range(1, 6)])
        before = source["target"]
        self.assertEqual(inspect(source, 1)["fact_count"], 5)
        self.assertEqual(source["target"], before)

    def test_aggregate_contains_no_plot_text_and_sources_unchanged(self):
        # All scratch stays below this owned experiment directory and is deleted.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            p = Path(temp)
            source = p / "source.jsonl"
            text = json.dumps(row(), ensure_ascii=False) + "\n"
            source.write_text(text)
            report = run([("fixture", source)], p / "private", p / "aggregate.json")
            self.assertEqual(source.read_text(), text)
            public = (p / "aggregate.json").read_text()
            self.assertNotIn("甲", public)
            self.assertNotIn("abstract-1", public)
            self.assertEqual(report["datasets"]["fixture"]["all"]["rows"], 1)
            private = json.loads((p / "private/fixture.review_bundle.jsonl").read_text())
            self.assertFalse(private["ready_for_training"])
            self.assertEqual(len(private["documents"]), 2)

    def test_quantiles(self):
        self.assertEqual(quantiles([1, 2, 3, 4])["p90_nearest_rank"], 4)


if __name__ == "__main__":
    unittest.main()
