import copy
import unittest

from native46_eval import check_task, parse_raw, shard_tasks, text_sha


def example():
    docs = [
        {"evidence_id": f"E{i}", "document_id": f"source#{i}",
         "text": f"角色甲：第{i}条\n完整证据。", "sha256": text_sha(f"角色甲：第{i}条\n完整证据。")}
        for i in range(1, 21)]
    bonsai = "\n".join(d["text"] for d in docs)
    fourb = "\n".join(d["document_id"] + " " + " ".join(d["text"].split()) for d in docs)
    return dict(documents=docs, round="1/2", bonsai_prompt=bonsai, fourb_prompt=fourb,
                bonsai_prompt_sha256=text_sha(bonsai), fourb_prompt_sha256=text_sha(fourb))


class Checks(unittest.TestCase):
    def test_valid(self):
        check_task(example())

    def test_hash_corruption_rejected(self):
        task = example()
        task["documents"][0]["text"] += "偷偷修改"
        with self.assertRaises(AssertionError):
            check_task(task)

    def test_missing_evidence_rejected(self):
        task = example()
        task["bonsai_prompt"] = "仅摘要"
        task["bonsai_prompt_sha256"] = text_sha(task["bonsai_prompt"])
        with self.assertRaises(AssertionError):
            check_task(task)

    def test_native_document_id_required(self):
        task = example()
        task["fourb_prompt"] = task["fourb_prompt"].replace("source#20", "unknown")
        task["fourb_prompt_sha256"] = text_sha(task["fourb_prompt"])
        with self.assertRaises(AssertionError):
            check_task(task)

    def test_targets_rejected(self):
        task = example()
        task["target"] = '{"next_action":"abstain"}'
        with self.assertRaises(AssertionError):
            check_task(task)

    def test_round_rejected(self):
        task = example()
        task["round"] = "3/2"
        with self.assertRaises(AssertionError):
            check_task(task)

    def test_prefill_not_repair(self):
        self.assertEqual(parse_raw('"next_action":"abstain"}', "{"),
                         {"next_action": "abstain"})
        self.assertIsNone(parse_raw('{"next_action":"abstain"', ""))

    def test_shards_disjoint_complete(self):
        tasks = list(range(46))
        a, b = shard_tasks(tasks, 0), shard_tasks(tasks, 1)
        self.assertEqual(len(a), 23)
        self.assertEqual(len(b), 23)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(sorted(a + b), tasks)


if __name__ == "__main__":
    unittest.main()
