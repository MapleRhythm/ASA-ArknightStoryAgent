"""Hash-bound semantic summaries retaining all records and duplicate questions."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from judge_visible_answers import validate
from judge_visible_answers_v2 import expected_key


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_question(question):
    return re.sub(r"\s+", "", question).strip("？?。！!，,；;：:\"'")


def summarize(rows):
    labels = Counter(row["label"] for row in rows)
    answering = sum(row["action"] == "answer_directly" for row in rows)
    complete = labels["supported_complete"]
    groups = defaultdict(list)
    for row in rows:
        groups[normalized_question(row["question"])].append(row)
    return {
        "records": len(rows), "distinct_normalized_questions": len(groups),
        "duplicate_question_groups": sum(len(group) > 1 for group in groups.values()),
        "labels": dict(labels),
        "grounded_actions": dict(Counter(row["action"] for row in rows)),
        "fully_supported_complete_records": complete,
        "complete_yield_all_records": complete / len(rows),
        "complete_fraction_among_shown_answers": complete / answering if answering else None,
        "supported_complete_or_partial_yield": (complete + labels["supported_partial"]) / len(rows),
        "unsupported_exposure_all_records": labels["unsupported"] / len(rows),
        "context_sufficiency": dict(Counter(r["context_sufficiency"] for r in rows)),
        "no_answer_despite_sufficient_context": sum(
            row["label"] == "no_answer" and row["context_sufficiency"] == "sufficient" for row in rows
        ),
        "macro_question_mean_complete_fraction": sum(
            sum(r["label"] == "supported_complete" for r in group) / len(group)
            for group in groups.values()
        ) / len(groups),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    audit = json.loads(args.semantic.read_text())
    if audit["source_analysis_sha256"] != sha(args.analysis):
        raise ValueError("semantic audit is not bound to this analysis")
    by_id = {r["id"]: r for r in analysis["results"]}
    verdicts = {r["id"]: r for r in audit["results"]}
    if (set(by_id) != set(verdicts) or len(by_id) != len(analysis["results"])
            or len(verdicts) != len(audit["results"])):
        raise ValueError("missing/duplicate IDs; cannot change denominator")
    config = SimpleNamespace(
        model=audit["requested_model"], max_tokens=audit["max_tokens"], endpoint=audit["endpoint"],
    )
    rows = []
    for source in analysis["results"]:
        item = verdicts[source["id"]]
        judgement = {}
        if item["status"] == "ok":
            if source["status"] != "ok" or item["input_key"] != expected_key(source, config):
                raise ValueError("judgement input hash mismatch")
            if item.get("response_model") != config.model:
                raise ValueError("unexpected response model")
            judgement = validate(item["judgement"], source)
        rows.append({
            "id": source["id"], "source_id": source["source_id"],
            "question": source["question"], "renderer": source["renderer"],
            "action": source.get("grounded", {}).get("next_action", "pipeline_error"),
            "label": judgement.get("verdict", item["status"]),
            "context_sufficiency": judgement.get("context_sufficiency", "not_judged"),
            "action_appropriateness": judgement.get("action_appropriateness", "not_judged"),
            "unsupported_spans": judgement.get("unsupported_spans", []),
            "missing_requirements": judgement.get("missing_requirements", []),
            "reason": judgement.get("reason", item.get("error", item.get("pipeline_error", ""))),
        })
    renderers = sorted({r["renderer"] for r in rows})
    report = {
        "version": "canonical_semantic_summary_v1", "evaluation_only": True, "training_ready": False,
        "analysis_sha256": sha(args.analysis), "semantic_sha256": sha(args.semantic),
        "summarizer_sha256": sha(__file__),
        "provisional_model_judgement_not_human_gold": True,
        "summary": {name: summarize([r for r in rows if r["renderer"] == name]) for name in renderers},
        "rows": rows,
        "limitations": [
            "Model judge disagreements require evidence review; counts are not independent human gold.",
            "No-answer, uncertain, judge failures and pipeline failures stay in the full denominator.",
            "Repeated questions are not independent tests; macro averages retain all evidence states.",
            "Visible-evidence state replay is not fresh retrieval or production end-to-end accuracy.",
            "Global answer support does not prove each cited E block semantically supports its attached fact.",
        ],
    }
    with args.output.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
