"""Read-only source provenance inventory; no environment or credential output."""
import json
from collections import Counter
from pathlib import Path

root = Path("${REMOTE_ASA_ROOT}/rank_v8_coverage/data")
for name in ("exx_relabel_v1_glm53_semantic_remaining_v2_20260827",
             "exx_grounding_v3_sft_minimal559_glm53_20260903",
             "exx_grounding_v2_sft_success559_contiguous_20260828"):
    for filename in ("labels.jsonl", "tasks.json", "train.json"):
        path = root / name / filename
        if not path.exists():
            continue
        if filename.endswith(".jsonl"):
            with path.open() as stream:
                rows = [json.loads(next(stream))]
        else:
            rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            print(name, filename, "ROOT_KEYS", list(rows))
            continue
        row = rows[0]
        print(name, filename, "FIRST_KEYS", list(row))
        safe = {k: v for k, v in row.items() if k in (
            "id", "provider", "model", "status", "next_action", "split",
            "api_meta", "source", "metadata", "usage", "action", "response_meta")}
        print(json.dumps(safe, ensure_ascii=False)[:1800])
        if "conversations" in row:
            print("PROMPT_START", row["conversations"][0]["value"][:1700])
            print("TARGET", row["conversations"][-1]["value"][:2000])
        for key in ("task", "context", "result", "label", "response", "target", "payload"):
            if key in row:
                value = row[key]
                print(key, (list(value) if isinstance(value, dict) else str(value)[:1200]))


if __name__ == "__main__":
    source = root / "exx_relabel_v1_glm53_semantic_remaining_v2_20260827"
    tasks = json.loads((source / "tasks.json").read_text())
    print("TASK_EVIDENCE_COUNTS", dict(Counter(len(t["evidence"]) for t in tasks)))
    for t in tasks[:2]:
        print("TASK_DETAIL", json.dumps(t, ensure_ascii=False)[:3500])
    with (source / "labels.jsonl").open() as stream:
        label = json.loads(next(stream))
    print("API_KEYS", list(label["api"]))
    print("API_METADATA", json.dumps({k: v for k, v in label["api"].items()
                                       if k in ("model", "endpoint", "usage", "id", "provider")}))
    print("GLM_LABEL", json.dumps(label["label"], ensure_ascii=False))
    task_map = {t["task_id"]: t for t in tasks}
    docs_path = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/indexes/arknights_story/documents.jsonl")
    docs = {d["id"]: d for d in (json.loads(l) for l in docs_path.read_text().splitlines())}
    stats = Counter()
    eligible = []
    for line in (source / "labels.jsonl").read_text().splitlines():
        record = json.loads(line)
        stats["labels"] += 1
        if record["api"].get("model") != "glm-5.3":
            continue
        stats["glm53"] += 1
        if record["label"].get("next_action") != "answer_directly":
            continue
        stats["glm53_answer"] += 1
        task = task_map[record["task_id"]]
        ev = task["evidence"]
        if not all(e["doc_id"] in docs and e["text"] == docs[e["doc_id"]]["clean_text"] for e in ev):
            continue
        stats["answer_full_exact_original_docs"] += 1
        families = {"/".join(e["doc_id"].split("/")[:2]) for e in ev}
        if len(families) != 1 or any(x in families for x in ("activities/act10d5", "activities/act10mini")):
            continue
        stats["answer_exact_single_family_safe"] += 1
        eligible.append({"id": task["task_id"], "family": next(iter(families)),
                         "count": len(ev), "question": task["question"],
                         "semantic_verifier_keys": list(record.get("semantic_verifier", {}))})
    print("ELIGIBLE_EXISTING_STATS", dict(stats))
    print("ELIGIBLE_FAMILIES", dict(Counter(r["family"] for r in eligible)))
    print("ELIGIBLE_FIRST", json.dumps(eligible[:3], ensure_ascii=False))
