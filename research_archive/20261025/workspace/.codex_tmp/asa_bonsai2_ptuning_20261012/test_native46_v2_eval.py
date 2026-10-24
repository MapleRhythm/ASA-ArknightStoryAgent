import copy
import unittest

import native46_eval as n
import native46_v2_eval as v
from test_native46_eval import example


def fixtures():
    tasks, validation = [], []
    for index in range(1, 47):
        task = example()
        task.update(position=index, id=str(index), question="问题", question_group="问题")
        row = {k: copy.deepcopy(task[k]) for k in ("id", "question", "round", "documents")}
        row.update(prompt=task["bonsai_prompt"], split="val", truncated=False)
        tasks.append(task)
        validation.append(row)
    return tasks, validation


def output(task, shard):
    return dict(
        **{k: task[k] for k in ("id", "position", "question", "round", "question_group")},
        arm="bonsai", shard=shard, evidence_count=20, input_truncated=False,
        answer_validation=False, protocol_prefill="",
        prompt_sha256=task["bonsai_prompt_sha256"], input_tokens=5000,
        output_token_ids=[1, 2], finish_reason="stop", output_reached_length_limit=False,
        raw_output='{"next_action":"abstain"}',
        parsed_output_for_offline_inspection_only={"next_action": "abstain"})


class Checks(unittest.TestCase):
    def test_same_validation(self):
        v.match_validation(*fixtures())

    def test_changed_prompt(self):
        tasks, validation = fixtures()
        validation[0]["prompt"] += "偷偷变更"
        with self.assertRaises(AssertionError):
            v.match_validation(tasks, validation)

    def test_changed_round(self):
        tasks, validation = fixtures()
        validation[0]["round"] = "2/2"
        with self.assertRaises(AssertionError):
            v.match_validation(tasks, validation)

    def test_missing_sample(self):
        tasks, validation = fixtures()
        with self.assertRaises(AssertionError):
            v.match_validation(tasks, validation[:-1])

    def test_valid_records(self):
        tasks, _ = fixtures()
        for shard in (0, 1):
            records = [output(t, shard) for t in n.shard_tasks(tasks, shard)]
            v.audit_records(tasks, records, shard)

    def test_wrong_order(self):
        tasks, _ = fixtures()
        with self.assertRaises(AssertionError):
            v.audit_records(tasks, [output(tasks[2], 0)], 0)

    def test_changed_invariants(self):
        tasks, _ = fixtures()
        for key, value in (
            ("answer_validation", True), ("input_truncated", True),
            ("protocol_prefill", "{"), ("prompt_sha256", "changed"),
            ("evidence_count", 19), ("input_tokens", 32000),
            ("output_reached_length_limit", True),
        ):
            record = output(tasks[0], 0)
            record[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                v.audit_records(tasks, [record], 0)

    def test_invalid_json_is_diagnostic_not_rejection(self):
        tasks, _ = fixtures()
        record = output(tasks[0], 0)
        record.update(raw_output='{"next_action":', parsed_output_for_offline_inspection_only=None)
        v.audit_records(tasks, [record], 0)


if __name__ == "__main__":
    unittest.main()
