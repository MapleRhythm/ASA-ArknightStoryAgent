"""Small offline fixtures: no model, GPU, network or API charges."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import annotation_protocol as protocol
import audit_fresh_subset as audit


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.out = self.root / "reviewed"
        self.rows = {}
        receipts = []
        for split in ("train", "val"):
            docs = [{"evidence_id": f"E{i}", "document_id": f"{split}-{i}",
                     "text": f"{split} 证据 {i}：阿米娅救出了博士。",
                     "sha256": protocol.sha(f"{split} 证据 {i}：阿米娅救出了博士。")}
                    for i in range(1, 21)]
            question = f"{split}：谁救出了博士？"
            messages, prompt = protocol.prompt_for(question, "1/2", docs)
            prompt_sha = protocol.sha(protocol.compact(messages))
            target = {"next_action": "answer_directly",
                      "supported_facts": [{"fact": "阿米娅救出博士。", "evidence_ids": ["E1"]}]}
            row = dict(id=split, split=split, documents=docs, evidence_count=20,
                       truncated=False, old_labels_used=False, round="1/2",
                       question=question, source_families=[split], prompt=prompt,
                       user_prompt=messages[1]["content"], annotation_model="glm-5.3",
                       annotation_response_id=f"response-{split}",
                       source_label="fresh_GLM_API_full20", target=json.dumps(target),
                       teacher_prompt_sha256=prompt_sha, label_prompt_sha256=prompt_sha)
            self.rows[split] = [row]
            receipts.append(dict(task_id=split, response_id=f"response-{split}",
                                 phase="annotate", accepted=True,
                                 endpoint=protocol.CODING_PLAN_ENDPOINT,
                                 requested_model="glm-5.3", teacher_prompt_sha256=prompt_sha,
                                 parsed=target, content=json.dumps(target)))
            write_rows(self.source / f"{split}.jsonl", [row])
        write_rows(self.source / "api_receipts.jsonl", receipts)
        write_json(self.source / "status.json",
                   dict(stage="batch_finished", remaining_tasks=0, total_tasks=3,
                        accepted=2, quarantined=1, api_calls=2))
        write_json(self.source / "quarantine.json", [{"id": "bad"}])
        self.manifest = dict(kind="fresh_GLM_full20_annotation", run_status="batch_finished",
                             old_labels_used=False, new_api_calls=2, teacher_model="glm-5.3",
                             teacher_endpoint=protocol.CODING_PLAN_ENDPOINT,
                             evidence_count_per_example=20, truncated_documents=0,
                             train=1, val=1, planned_train=2, planned_val=1,
                             labelled_examples=2, complete=False,
                             data_sha256={s: audit.file_sha(self.source / f"{s}.jsonl")
                                          for s in self.rows})
        write_json(self.source / "manifest.json", self.manifest)
        self.review = self.root / "review.json"
        self.review_payload = dict(excluded={}, additional_retained_spot_checks={"train": "OK"},
                                   labels_manually_rewritten=0, not_QA_accuracy=True,
                                   scope="Fixture review; not semantic certification.")
        write_json(self.review, self.review_payload)

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return audit.build(self.source, self.review, self.out)

    def change_rows(self, split, transform):
        rows = copy.deepcopy(self.rows[split])
        transform(rows[0])
        write_rows(self.source / f"{split}.jsonl", rows)
        self.manifest["data_sha256"][split] = audit.file_sha(self.source / f"{split}.jsonl")
        write_json(self.source / "manifest.json", self.manifest)

    def test_incomplete_source_can_yield_explicit_subset(self):
        self.assertEqual(self.build()["train"], 1)
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertFalse(manifest["source_complete"])
        self.assertFalse(manifest["all_retained_labels_semantically_certified"])
        self.assertTrue(manifest["ready_for_training"])

    def test_no_overwrite(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.build()

    def test_selected_file_tampering(self):
        self.build()
        with (self.out / "train.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "selected data changed"):
            audit.verify(self.out)

    def test_source_file_tampering(self):
        self.build()
        with (self.out / "source/api_receipts.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "source copy changed"):
            audit.verify(self.out)

    def test_unknown_review_id(self):
        self.review_payload["excluded"] = {"unknown": "bad"}
        write_json(self.review, self.review_payload)
        with self.assertRaisesRegex(ValueError, "unknown exclusion"):
            self.build()

    def test_old_labels_rejected(self):
        self.change_rows("train", lambda r: r.update(old_labels_used=True))
        with self.assertRaisesRegex(ValueError, "lineage"):
            self.build()

    def test_prompt_tampering_rejected(self):
        self.change_rows("train", lambda r: r.update(prompt=r["prompt"][:-20]))
        with self.assertRaisesRegex(ValueError, "prompt mismatch"):
            self.build()

    def test_citation_binding_rejected(self):
        def mutate(row):
            target = json.loads(row["target"])
            target["supported_facts"][0]["evidence_ids"] = ["E21"]
            row["target"] = json.dumps(target)
        self.change_rows("train", mutate)
        with self.assertRaisesRegex(ValueError, "invalid_binding"):
            self.build()

    def test_teacher_target_mismatch_rejected(self):
        def mutate(row):
            target = json.loads(row["target"])
            target["supported_facts"][0]["fact"] = "不是原始标签。"
            row["target"] = json.dumps(target)
        self.change_rows("train", mutate)
        with self.assertRaisesRegex(ValueError, "teacher response"):
            self.build()

    def test_family_leakage_rejected(self):
        self.change_rows("val", lambda r: r.update(source_families=["train"]))
        with self.assertRaisesRegex(ValueError, "source_family leakage"):
            self.build()

    def test_finished_status_required(self):
        status = json.loads((self.source / "status.json").read_text())
        status["remaining_tasks"] = 1
        write_json(self.source / "status.json", status)
        with self.assertRaisesRegex(ValueError, "fully processed"):
            self.build()

    def test_exclusion_preserves_original(self):
        self.review_payload["excluded"] = {"val": "Explicit offline reason."}
        write_json(self.review, self.review_payload)
        result = self.build()
        self.assertEqual(result["val"], 0)
        self.assertEqual(audit.read_rows(self.out / "excluded.jsonl")[0]["source_row"],
                         self.rows["val"][0])


if __name__ == "__main__":
    unittest.main()
