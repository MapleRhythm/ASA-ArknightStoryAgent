import hashlib
import json
from pathlib import Path


run = Path(__file__).resolve().parent
sources = {variant: json.loads((run / f"{variant}.matrix.json").read_text())
           for variant in ("before", "after")}
paired = {variant: {row["source_id"]: row for row in doc["rows"]}
          for variant, doc in sources.items()}
if paired["before"].keys() != paired["after"].keys():
    raise ValueError("before/after question IDs differ")
rows = []
for index, question_id in enumerate(sorted(paired["before"])):
    before, after = paired["before"][question_id], paired["after"][question_id]
    for key in ("question", "question_family", "hypothesis", "prompt_evidence", "system"):
        if before[key] != after[key]:
            raise ValueError(f"non-rendering input differs: {key}")
    order = ("before", "after") if index % 2 == 0 else ("after", "before")
    rows.extend(paired[variant][question_id] for variant in order)
output = {
    "version": "asa_canonical_quote_pairs_v1",
    "evaluation_only": True, "training_ready": False,
    "question_count": len(paired["before"]),
    "source_sha256": {variant: hashlib.sha256((run / f"{variant}.matrix.json").read_bytes()).hexdigest()
                      for variant in sources},
    "rows": rows,
}
with (run / "paired.matrix.json").open("x", encoding="utf-8") as f:
    f.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"questions": output["question_count"], "paired_rows": len(rows)}))
