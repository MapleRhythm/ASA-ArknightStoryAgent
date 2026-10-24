"""Candidate cause-scope clarification, evaluated independently of story gold."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_answer_slots_v2 as previous

VERSION = previous.VERSION
ANCHOR = "问原因/方式/关系时，要有实际原因/方式/关系信息，而非重复‘因为有原因’之类套话。"
CLARIFICATION = (
    "原因可以是促使被问主体行动的具体上级命令、规则约束或客观条件；"
    "在已指出这类具体原因时，不再强求解释其背后另一个主体的更深动机，除非问题明确问到。"
    "仅复述被问主体的同一行为，不构成原因。"
)
assert previous.RUBRIC.count(ANCHOR) == 1
RUBRIC = previous.RUBRIC.replace(ANCHOR, ANCHOR + CLARIFICATION)
validate = previous.validate


def messages(case):
    return [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": json.dumps({
            "question": case["question"], "shown_answer": case["shown_answer"],
        }, ensure_ascii=False)},
    ]
