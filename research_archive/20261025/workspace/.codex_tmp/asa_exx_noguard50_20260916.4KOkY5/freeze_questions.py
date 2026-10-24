"""Separate question-only runtime input from offline reference material."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = root.parent / "asa_e2e20_20260914.9KSGhY/eval50_recall_questions_for_soda.jsonl"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


assert sha(source) == "fbfa6e1a4351aa484d0b8ce2e7d100044f994fa3b2079384ca804c42e6e218af"
rows = [json.loads(line) for line in source.read_text().splitlines()]
assert len(rows) == 50
assert [row["question_key"] for row in rows] == [f"eval50_{i:03d}" for i in range(50)]
assert len(set(row["question"] for row in rows)) == 50
with (root / "questions.txt").open("x") as handle:
    handle.write("\n".join(row["question"] for row in rows) + "\n")
with (root / "offline_references.jsonl").open("x") as handle:
    handle.write(source.read_text())
freeze = {
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "count": 50, "question_order": [row["question_key"] for row in rows],
    "source_sha256": sha(source), "questions_sha256": sha(root / "questions.txt"),
    "protocol_sha256": sha(root / "PROTOCOL.md"),
    "no_gold_in_questions": True,
}
with (root / "freeze.json").open("x") as handle:
    json.dump(freeze, handle, ensure_ascii=False, indent=2)
print(json.dumps(freeze, ensure_ascii=False))
