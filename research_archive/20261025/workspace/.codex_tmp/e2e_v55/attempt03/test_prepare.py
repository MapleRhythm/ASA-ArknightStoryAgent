from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare
import train


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.row = {"conversations": [{"from": "human", "value": "question"},
                                      {"from": "gpt", "value": '{"answer":"value"}'}]}

    def test_messages_preserve_original(self):
        before = deepcopy(self.row)
        self.assertEqual(prepare.messages(self.row)[0], {"role": "user", "content": "question"})
        self.assertEqual(before, self.row)

    def test_bad_roles_rejected(self):
        self.row["conversations"][0]["from"] = "gpt"
        with self.assertRaises(AssertionError):
            prepare.messages(self.row)

    def test_multiturn_not_silently_dropped(self):
        self.row["conversations"].append({"from": "human", "value": "extra"})
        with self.assertRaises(AssertionError):
            prepare.messages(self.row)

    def test_empty_target_rejected(self):
        self.row["conversations"][1]["value"] = ""
        with self.assertRaises(AssertionError):
            prepare.messages(self.row)

    def test_new_artifacts_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            prepare.save(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                prepare.save(path, {"a": 2})
            self.assertEqual(prepare.load(path), {"a": 1})

    def test_snapshot_tamper_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepare.save(root / "data.json", {"a": 1})
            prepare.save(root / "snapshot.json", {"sha256": {"data.json": prepare.sha(root / "data.json")}})
            with patch.object(prepare, "HERE", root):
                prepare.verify_snapshot()
                with (root / "data.json").open("a") as stream:
                    stream.write("\n")
                with self.assertRaises(AssertionError):
                    prepare.verify_snapshot()

    def test_gpu_receipt_keeps_other_pid(self):
        with patch.object(train.subprocess, "check_output", side_effect=[
            f"{train.GPU_UUID}, 40960, 1915, 0\n", "450914, 1892\n"
        ]):
            result = train.gpu_status()
        self.assertEqual(result["processes"], [{"pid": 450914, "memory_mib": 1892}])

    def test_wrong_gpu_rejected(self):
        with patch.object(train.subprocess, "check_output", return_value="wrong, 40960, 0, 0\n"):
            with self.assertRaises(AssertionError):
                train.gpu_status()


if __name__ == "__main__":
    unittest.main()
