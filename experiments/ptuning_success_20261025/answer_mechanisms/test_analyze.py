"""Synthetic provenance tests; fixtures contain no story text."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import analyze as a


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class AnalyzeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        doc = "synthetic passage\nsecond line"
        task = dict(id="c1", bonsai_prompt_sha256="p", documents=[dict(evidence_id="E1", text=doc, sha256=sha(doc))])
        before, after = dict(id="c1", raw_text="core fact"), dict(id="c1", raw_text="core fact; wrong extra fact")
        files = {"task": ("tasks.jsonl", json.dumps(task) + "\n"),
                 "input": ("in.jsonl", json.dumps(before) + "\n"),
                 "v2": ("v2.jsonl", json.dumps(after) + "\n")}
        for name, text in files.values():
            (self.root / name).write_text(text)
        self.audit = dict(
            source_paths={k: v[0] for k, v in files.items()},
            source_file_sha256={k: sha(v[1]) for k, v in files.items()},
            cases=[dict(position=1, caseID="c1", transition="lost_complete", mechanism="speaker_binding",
                        scope_status="clear_clean_prefix", minimal_adequate_clean_prefix_end_fact=1,
                        first_material_v2_error=dict(fact_index_1based=2, text="wrong extra fact",
                                                     raw_char_range_0based_halfopen=[11, 27]),
                        sources=dict(task_prompt_sha256="p", input_raw_sha256=sha(before["raw_text"]),
                                     v2_raw_sha256=sha(after["raw_text"]), input_read_char_range=[0, 9],
                                     v2_read_char_range=[0, 27],
                                     evidence_read=[dict(evidence_id="E1", sha256=sha(doc),
                                                         char_range=[0, len(doc)], line_range=[1, 2])]))])

    def tearDown(self):
        self.tmp.cleanup()

    def run_audit(self, audit):
        path = self.root / "audit.json"
        path.write_text(json.dumps(audit))
        return a.analyze(self.root, path)

    def test_counts_only_and_no_text(self):
        result = self.run_audit(self.audit)
        self.assertEqual(result["lost_scope_status_counts"], {"clear_clean_prefix": 1})
        self.assertNotIn("wrong extra fact", json.dumps(result))
        self.assertFalse(result["new_model_accuracy_measured"])

    def test_offset_and_source_tampering_raise(self):
        bad = copy.deepcopy(self.audit)
        bad["cases"][0]["first_material_v2_error"]["raw_char_range_0based_halfopen"] = [10, 26]
        with self.assertRaisesRegex(ValueError, "offset"):
            self.run_audit(bad)
        bad = copy.deepcopy(self.audit)
        bad["source_file_sha256"]["v2"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Changed source"):
            self.run_audit(bad)
        bad = copy.deepcopy(self.audit)
        bad["cases"][0]["sources"]["evidence_read"][0]["char_range"] = [0, 5]
        with self.assertRaisesRegex(ValueError, "fully read"):
            self.run_audit(bad)


if __name__ == "__main__":
    unittest.main()
