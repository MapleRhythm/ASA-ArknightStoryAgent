"""Materialize a GLM-labelled P-tuning dataset with 20 complete documents.

The source labels are reused offline. Quotes/final answers are not copied into
the current action target; only GLM's structured decision and atomic facts are
kept, with evidence references converted to current E-number bindings.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


MARKER = re.compile(r"(?m)^E(\d+)\s+(\S+)\s*$")
EXCLUDED = ("act10d5", "act10mini")


def parse_evidence(text):
    matches = list(MARKER.finditer(text))
    rows = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        body = body.split("\n输出格式：", 1)[0].split("\n规则：", 1)[0].strip()
        if body:
            rows.append({"old_e": f"E{match.group(1)}", "id": match.group(2), "text": body})
    return rows


def normalize_target(raw, mapping):
    action = raw.get("next_action")
    if action == "answer_directly":
        facts = []
        for item in raw.get("supported_facts", []):
            refs = item.get("evidence_refs", [])
            bound = []
            for ref in refs:
                doc = ref.get("evidence_id")
                if doc in mapping and mapping[doc] not in bound:
                    bound.append(mapping[doc])
            if item.get("fact") and bound:
                facts.append({"fact": str(item["fact"]), "evidence_ids": bound[:2]})
        if not facts:
            return None
        return {"next_action": "answer_directly", "supported_facts": facts[:8]}
    if action == "retrieve_more":
        hypo = raw.get("follow_up_hypothesis")
        if not isinstance(hypo, dict) or not hypo.get("question"):
            return None
        return {"next_action": "retrieve_more", "follow_up_hypothesis": hypo}
    if action == "abstain":
        return {"next_action": "abstain", "reason": raw.get("final_answer") or "现有证据不足以确认。"}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--documents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--val", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=False)

    docs = {}
    by_story = {}
    index_hash = hashlib.sha256()
    with open(args.documents, "rb") as stream:
        for line in stream:
            index_hash.update(line)
            row = json.loads(line)
            docs[row["id"]] = row
            by_story.setdefault(row.get("story_id") or row.get("activity_id"), []).append(row)

    source = json.load(open(args.source))
    candidates = []
    for row in source:
        human = row["conversations"][0]["value"]
        evidence = parse_evidence(human)
        if len(evidence) < 8 or any(x["id"].startswith(EXCLUDED) for x in evidence):
            continue
        if any(x["id"] not in docs for x in evidence):
            continue
        try:
            target_raw = json.loads(row["conversations"][-1]["value"])
        except Exception:
            continue
        story = docs[evidence[0]["id"]].get("story_id") or docs[evidence[0]["id"]].get("activity_id")
        pool = [x for x in by_story.get(story, []) if x["id"] not in {e["id"] for e in evidence}]
        if len(pool) < 20 - len(evidence):
            activity = docs[evidence[0]["id"]].get("activity_id")
            pool = [x for x in docs.values()
                    if x.get("activity_id") == activity and x["id"] not in {e["id"] for e in evidence}]
        def distance(item):
            chunk = int(item["id"].rsplit("#chunk-", 1)[-1])
            distances = []
            for ev in evidence:
                if "#chunk-" in ev["id"]:
                    distances.append(abs(chunk - int(ev["id"].rsplit("#chunk-", 1)[-1])))
            return min(distances or [999999]), item["id"]
        pool.sort(key=distance)
        selected = evidence + [{"old_e": None, "id": x["id"], "text": x["clean_text"]}
                               for x in pool[:20 - len(evidence)]]
        if len(selected) != 20 or any(not x["text"].strip() for x in selected):
            continue
        selected = selected[:20]
        # Bind original evidence IDs to the newly assigned E numbers.
        mapping = {x["id"]: f"E{i + 1}" for i, x in enumerate(selected)}
        target = normalize_target(target_raw, mapping)
        if target is None:
            continue
        question_match = re.search(r"(?m)^问题：(.+)$", human)
        question = question_match.group(1).strip() if question_match else row["id"]
        # Keep the original protocol text, replacing only its evidence region.
        prefix = human.split("证据：", 1)[0] + "证据：\n"
        suffix = ""
        if "\n输出格式：" in human:
            suffix = "\n输出格式：" + human.split("\n输出格式：", 1)[1]
        elif "\n规则：" in human:
            suffix = "\n规则：" + human.split("\n规则：", 1)[1]
        evidence_block = "\n".join(f"E{i + 1} {x['id']}\n{x['text']}" for i, x in enumerate(selected))
        prompt = (
            "<|im_start|>system\n你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。"
            "<|im_end|>\n<|im_start|>user\n" + prefix + evidence_block + suffix +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        assert len(re.findall(r"(?m)^E\d+ ", evidence_block)) == 20
        assert all(x["text"] in prompt for x in selected)
        candidates.append({
            "id": row["id"], "question": question, "prompt": prompt,
            "target": json.dumps(target, ensure_ascii=False),
            "documents": [{"evidence_id": f"E{i + 1}", "document_id": x["id"],
                           "text": x["text"], "sha256": hashlib.sha256(x["text"].encode()).hexdigest()}
                          for i, x in enumerate(selected)],
            "evidence_count": 20, "truncated": False,
            "story_id": story, "source_label": "GLM",
        })

    # Group split by story_id, deterministic and independent of row order.
    groups = {}
    for row in candidates:
        groups.setdefault(row["story_id"], []).append(row)
    story_ids = sorted(groups)
    random.Random(20261012).shuffle(story_ids)
    val_groups, train_groups = [], []
    for story in story_ids:
        if sum(len(groups[x]) for x in val_groups) < args.val:
            val_groups.append(story)
        else:
            train_groups.append(story)
    train_rows = [r for s in train_groups for r in groups[s]][:args.train]
    val_rows = [r for s in val_groups for r in groups[s]][:args.val]
    if len(train_rows) < args.train or len(val_rows) < args.val:
        raise RuntimeError(f"not enough clean GLM rows: candidates={len(candidates)} train={len(train_rows)} val={len(val_rows)}")
    assert not {x["story_id"] for x in train_rows} & {x["story_id"] for x in val_rows}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        (out / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    (out / "manifest.json").write_text(json.dumps({
        "source": str(args.source), "source_kind": "existing_GLМ_structured_labels",
        "source_rows": len(source), "clean_candidates": len(candidates),
        "train": len(train_rows), "val": len(val_rows), "evidence_count_per_example": 20,
        "truncated_documents": 0, "train_val_story_overlap": [],
        "new_api_calls": 0, "labels_provider": "GLM",
        "quotes_and_final_answers_removed": True,
        "target_schema": "current_grounded_action_evidence_ids_v1",
        "source_index_sha256": index_hash.hexdigest(),
        "not_human_gold": True, "not_unseen_benchmark": True,
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"source_rows": len(source), "candidates": len(candidates),
                      "train": len(train_rows), "val": len(val_rows)}), flush=True)


if __name__ == "__main__":
    main()
