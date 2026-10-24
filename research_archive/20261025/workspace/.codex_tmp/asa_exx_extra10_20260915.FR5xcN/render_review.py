"""Render human-reviewed grades next to immutable raw outputs and evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


root = Path(__file__).resolve().parent
review = json.loads((root / "review.json").read_text())
rows = [json.loads(line) for line in (root / "run/results.jsonl").read_text().splitlines()]
assert sha(root / "run/results.jsonl") == review["results_sha256"]
assert sha(root / "PROTOCOL.md") == review["protocol_sha256"]
assert len(rows) == len(review["per_question"]) == 10
counts = Counter(grade["final_grade"] for grade in review["per_question"])
assert counts == Counter(strict_correct=2, abstain=8)
assert Counter(grade["last_candidate_grade"] for grade in review["per_question"]) == Counter(
    strict_correct=9, partial_or_ambiguous=1)
prompts = {}
with (root / "run/audit.jsonl").open() as handle:
    for line in handle:
        event = json.loads(line)
        if event["event"] == "prompt_evidence":
            prompts.setdefault(event["question_index"], []).append(event)
parts = [
    "# 逐题完整核对材料",
    "最终交付与候选严格区分；原始候选未成功交付时不计全链路成功。"
    "本文件由冻结输出及助手人工评分渲染，不调用模型、不修改任何答案。",
]
for index, (row, grade) in enumerate(zip(rows, review["per_question"]), 1):
    assert grade["q"] == index
    evidence_prompts = prompts[index]
    traces = row["retrieval_trace"]
    assert len(evidence_prompts) == len(traces)
    assert row["answer"] == traces[-1]["conclusion"]["answer"]
    for ref in grade["first_prompt_gold_refs"]:
        document = evidence_prompts[0]["documents"][int(ref[1:]) - 1]
        assert document["clean_text"].strip()
    parts += [
        f"## {index}. {grade['topic']}",
        f"问题：{row['question']}",
        f"实际交付：{row['answer']}",
        f"最终评分：{grade['final_grade']}；末轮候选诊断：{grade['last_candidate_grade']}。",
        f"必要要点：{grade['required']}",
        f"评审理由：{grade['reason']}",
        f"参考原文：[{grade['reference']}]({grade['reference']})",
        f"耗时：{row['elapsed_sec']}秒；检索轮数：{len(traces)}。",
    ]
    for turn, (trace, context) in enumerate(zip(traces, evidence_prompts), 1):
        conclusion = trace["conclusion"]
        diag = conclusion["generation_diagnostics"]
        raw = diag["raw_output"]
        payload = json.loads(raw)
        parts += [
            f"### 第{turn}轮候选及实际引用",
            "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```",
            f"校验结果：{diag.get('grounding_status')}；行动：{conclusion['next_action']}。",
            "告警：\n```json\n" + json.dumps(conclusion.get("grounding_warnings", []),
                                          ensure_ascii=False, indent=2) + "\n```",
        ]
        refs = set()
        for fact in payload.get("supported_facts", []) + payload.get("inferred_facts", []):
            refs.update(fact.get("evidence_ids", []))
        for ref in sorted(refs, key=lambda text: int(text[1:])):
            doc = context["documents"][int(ref[1:]) - 1]
            parts += [
                f"{ref}：`{doc['id']}`",
                "```text\n" + doc["clean_text"] + "\n```",
            ]
with (root / "DETAILS.md").open("x") as handle:
    handle.write("\n\n".join(parts) + "\n")
paths = [
    root / name for name in
    ("PROTOCOL.md", "questions.txt", "review.json", "REPORT.md", "DETAILS.md",
     "run_extra10.py", "launch_extra10.py", "audit_batch.py", "render_review.py",
     "mechanical_audit.json", "launch.json", "extra10.log")
]
paths += sorted((root / "reference_sources").rglob("*.txt"))
paths += sorted(path for path in (root / "run").iterdir() if path.is_file())
hashes = {str(path.relative_to(root)): sha(path) for path in paths}
with (root / "artifact_hashes.json").open("x") as handle:
    json.dump(hashes, handle, ensure_ascii=False, indent=2)
print(json.dumps({"review_consistent": True, "final_counts": dict(counts),
                  "last_candidate_counts": dict(Counter(
                      grade["last_candidate_grade"] for grade in review["per_question"])),
                  "details": str(root / "DETAILS.md"),
                  "hashed_files": len(hashes)}, ensure_ascii=False))
