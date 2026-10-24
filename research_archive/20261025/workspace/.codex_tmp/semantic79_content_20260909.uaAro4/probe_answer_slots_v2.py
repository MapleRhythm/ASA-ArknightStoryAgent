"""Clarify one schema clause; preserve the frozen semantic rubric/validator."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_answer_slots_v1 as original

VERSION = original.VERSION
OLD_CLAUSE = "slots为1至6项，filled必须有真正的答案摘句，unfilled不得有answer_spans。"
NEW_CLAUSE = (
    "slots为1至6项，每个slot都必须保留answer_spans字段且值为数组；"
    "filled必须有真正的答案摘句，unfilled的answer_spans必须为[]，不得省略该字段。"
)
assert original.RUBRIC.count(OLD_CLAUSE) == 1
RUBRIC = original.RUBRIC.replace(OLD_CLAUSE, NEW_CLAUSE)
validate = original.validate


def messages(case):
    return [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": json.dumps({
            "question": case["question"], "shown_answer": case["shown_answer"],
        }, ensure_ascii=False)},
    ]
