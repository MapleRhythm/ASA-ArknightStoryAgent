"""Hash-verified content-aware summary, preserving legacy labels separately."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from answer_semantics_content_v2 import PROTOCOL, messages, validate
from judge_visible_answers import digest, validate as legacy_validate


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run = Path(__file__).resolve().parent
folder = run / "question_focus_paired18_ctx12288_v1"
analysis_path = folder / "validation_v1.json"
new_path = folder / "semantic18_content_v2.json"
old_path = folder / "semantic18_v1.json"
analysis, new_audit, old_audit = map(load, (analysis_path, new_path, old_path))
assert new_audit["protocol"] == PROTOCOL
assert new_audit["rubric_script_sha256"] == sha(run / "answer_semantics_content_v2.py")
assert new_audit["source_analysis_sha256"] == old_audit["source_analysis_sha256"] == sha(analysis_path)
sources = {r["id"]: r for r in analysis["results"]}
current = {r["id"]: r for r in new_audit["results"]}
legacy = {r["id"]: r for r in old_audit["results"]}
assert len(sources) == len(current) == len(legacy) == 18
assert sources.keys() == current.keys() == legacy.keys()
rows = []
for source in analysis["results"]:
    item, old = current[source["id"]], legacy[source["id"]]
    judgement = {}
    if item["status"] == "ok":
        expected = digest({
            "payload": {"model": new_audit["requested_model"], "messages": messages(source),
                        "temperature": 0, "max_tokens": new_audit["max_tokens"]},
            "endpoint": new_audit["endpoint"], "id": source["id"], "grounded": source["grounded"],
        })
        assert item["input_key"] == expected
        assert item["response_model"] == item["requested_model"] == new_audit["requested_model"]
        assert item["finish_reason"] == "stop"
        judgement = validate(item["judgement"], source)
    old_judgement = legacy_validate(old["judgement"], source) if old["status"] == "ok" else {}
    rows.append({
        "id": source["id"], "question": source["question"], "renderer": source["renderer"],
        "action": source["grounded"]["next_action"], "answer": source["grounded"]["answer"],
        "status": item["status"], "label": judgement.get("verdict", item["status"]),
        "answer_content": judgement.get("answer_content", "not_judged"),
        "supported_answer_spans": judgement.get("supported_answer_spans", []),
        "unsupported_spans": judgement.get("unsupported_spans", []),
        "reason": judgement.get("reason", item.get("error", "")),
        "legacy_label": old_judgement.get("verdict", old["status"]),
    })
summary = {}
for renderer in sorted({r["renderer"] for r in rows}):
    subset = [r for r in rows if r["renderer"] == renderer]
    summary[renderer] = {
        "records": len(subset), "labels": dict(Counter(r["label"] for r in subset)),
        "answer_content": dict(Counter(r["answer_content"] for r in subset)),
        "actions": dict(Counter(r["action"] for r in subset)),
        "legacy_labels": dict(Counter(r["legacy_label"] for r in subset)),
        "label_changed_ids": [r["id"] for r in subset if r["label"] != r["legacy_label"]],
    }
report = {
    "protocol": PROTOCOL, "evaluation_only": True, "training_ready": False,
    "analysis_sha256": sha(analysis_path), "new_audit_sha256": sha(new_path), "legacy_audit_sha256": sha(old_path),
    "summarizer_sha256": sha(Path(__file__)), "summary": summary, "rows": rows,
    "limitations": [
        "A rubric correction, not changed model outputs or increased answering ability.",
        "New and old counts are separate; no old result was overwritten or silently relabeled.",
        "All eighteen development records, including any judge failures, remain in the denominator.",
        "Question overlap and repeated prompt development prevent independent generalization claims.",
    ],
}
with (folder / "semantic_summary18_content_v2.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False))
