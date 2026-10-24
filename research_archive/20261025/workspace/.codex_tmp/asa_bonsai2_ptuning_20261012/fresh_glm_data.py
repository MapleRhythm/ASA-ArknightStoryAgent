"""Fresh GLM supervision from full indexed evidence, not recycled old labels."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import ssl
import time
import urllib.error
import urllib.request

from build_glm_pilot import parse_evidence
from prepare_pilot import RULES

SYSTEM = "你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。"
FOLLOW = ("question", "query_type", "entities", "keywords", "expected_answer_type")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def family(doc_id):
    parts = doc_id.split("/")
    return "/".join(parts[:2])


def excluded(doc_id):
    return any(part in {"act10d5", "act10mini"} for part in doc_id.split("/"))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def prepare(args):
    out = Path(args.out)
    out.mkdir(exist_ok=False)
    docs, by_family = {}, {}
    index_hash = hashlib.sha256()
    with open(args.documents, "rb") as stream:
        for line in stream:
            index_hash.update(line)
            doc = json.loads(line)
            docs[doc["id"]] = doc
            by_family.setdefault(family(doc["id"]), []).append(doc)
    candidates, seen_questions = [], set()
    dropped = Counter()
    source = json.loads(Path(args.source).read_text())
    for row in source:
        human = row["conversations"][0]["value"]
        evidence = parse_evidence(human)
        question_match = re.search(r"(?m)^问题：(.+)$", human)
        if not question_match or not evidence:
            dropped["unparsed_question_or_evidence"] += 1
            continue
        question = question_match.group(1).strip()
        normalized = re.sub(r"\W", "", question)
        if normalized in seen_questions:
            dropped["duplicate_question"] += 1
            continue
        ids = list(dict.fromkeys(x["id"] for x in evidence))
        if len(ids) > 20 or any(i not in docs or excluded(i) for i in ids):
            dropped["missing_excluded_or_too_many_documents"] += 1
            continue
        families = {family(i) for i in ids}
        # Explicit family-scoped offline pilot. Reject whole multi-family
        # inputs rather than dropping inconvenient evidence to force a split.
        if len(families) != 1:
            dropped["multi_family_input_not_used_in_this_pilot"] += 1
            continue
        group = next(iter(families))
        pool = [x for x in by_family[group] if x["id"] not in ids and not excluded(x["id"])]
        def distance(doc):
            chapter, _, chunk = doc["id"].partition("#chunk-")
            if not chunk.isdigit():
                return (True, 999999, doc["id"])
            anchors = [i for i in ids if i.partition("#chunk-")[0] == chapter]
            delta = min((abs(int(chunk) - int(i.partition("#chunk-")[2]))
                         for i in anchors if i.partition("#chunk-")[2].isdigit()), default=999999)
            return (not bool(anchors), delta, doc["id"])
        pool.sort(key=distance)
        selected = [docs[i] for i in ids] + pool[:20 - len(ids)]
        if len(selected) != 20 or any(not x["clean_text"].strip() for x in selected):
            dropped["insufficient_complete_documents"] += 1
            continue
        random.Random(20261013 + len(candidates)).shuffle(selected)
        documents = [{"evidence_id": f"E{i+1}", "document_id": x["id"],
                      "text": x["clean_text"], "sha256": digest(x["clean_text"])}
                     for i, x in enumerate(selected)]
        assert all(d["text"] == docs[d["document_id"]]["clean_text"] for d in documents)
        body = (f"task: grounded_action_generation\nquestion: {question}\n"
                'hypothesis: {"query_type":"reasoning","expected_answer_type":"剧情解释"}\n'
                "round: 2/2\nevidence:\n" +
                "\n".join(f"[{d['evidence_id']}]\n{d['text']}" for d in documents) +
                "\noutput_schema: grounded_action_exx_v1\nrules: " + RULES +
                " follow_up_hypothesis包含question、query_type、entities、keywords、expected_answer_type。")
        prompt = ("<|im_start|>system\n" + SYSTEM +
                  "<|im_end|>\n<|im_start|>user\n" + body +
                  "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        candidates.append({"id": row["id"], "question": question, "prompt": prompt,
                           "teacher_messages": [{"role": "system", "content": SYSTEM},
                                                {"role": "user", "content": body}],
                           "documents": documents, "evidence_count": 20,
                           "truncated": False, "source_family": group, "round": "2/2"})
        seen_questions.add(normalized)
    groups = {}
    for row in candidates:
        groups.setdefault(row["source_family"], []).append(row)
    group_names = sorted(groups)
    random.Random(20261013).shuffle(group_names)
    val_groups = []
    count = 0
    for name in group_names:
        if count < args.val:
            val_groups.append(name)
            count += len(groups[name])
    splits = {
        "train": [r for g in group_names if g not in val_groups for r in groups[g]][:args.train],
        "val": [r for g in val_groups for r in groups[g]][:args.val],
    }
    if any(len(splits[s]) < getattr(args, s) for s in splits):
        raise RuntimeError(f"insufficient family-isolated candidates: {len(candidates)}, "
                           f"groups={ {g: len(v) for g,v in groups.items()} }")
    family_sets = {s: {r["source_family"] for r in rows} for s, rows in splits.items()}
    doc_sets = {s: {d["document_id"] for r in rows for d in r["documents"]}
                for s, rows in splits.items()}
    assert not family_sets["train"] & family_sets["val"]
    assert not doc_sets["train"] & doc_sets["val"]
    rows = [dict(r, split=s) for s, group in splits.items() for r in group]
    (out / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "kind": "fresh_GLM_family_scoped_offline_full_evidence_pilot",
        "source_questions": str(args.source), "source_question_rows": len(source),
        "source_index": str(args.documents), "source_index_sha256": index_hash.hexdigest(),
        "old_labels_used": False, "evidence_replaced_with_full_index_text": True,
        "candidate_count": len(candidates), "dropped": dict(dropped),
        "planned_train": args.train, "planned_val": args.val,
        "train": 0, "val": 0, "evidence_count_per_example": 20, "truncated_documents": 0,
        "train_val_story_overlap": [], "train_val_source_family_overlap": [],
        "train_val_document_overlap": [],
        "families": {s: sorted(v) for s, v in family_sets.items()},
        "excluded_families": ["act10d5", "act10mini"],
        "new_api_calls": 0, "labels_provider": None, "complete": False,
        "not_human_gold": True, "not_unseen_benchmark": True,
        "not_full_chain_retrieval_evaluation": True,
        "target_schema": "grounded_action_exx_v1",
        "data_checks_not_production_answer_validation": True,
    }
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def validate_target(value):
    if not isinstance(value, dict):
        raise ValueError("target_not_object")
    action = value.get("next_action")
    if action == "answer_directly":
        if set(value) != {"next_action", "supported_facts"}:
            raise ValueError("answer_schema")
        facts = value["supported_facts"]
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            raise ValueError("facts_length")
        for fact in facts:
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                raise ValueError("fact_schema")
            refs = fact["evidence_ids"]
            if not isinstance(fact["fact"], str) or not fact["fact"].strip():
                raise ValueError("empty_fact")
            if not isinstance(refs, list) or not 1 <= len(refs) <= 2:
                raise ValueError("reference_count")
            if any(not isinstance(x, str) or x not in {f"E{i}" for i in range(1, 21)} for x in refs):
                raise ValueError("unknown_reference")
            if len(refs) != len(set(refs)):
                raise ValueError("duplicate_reference")
    elif action == "abstain":
        if set(value) != {"next_action", "reason"} or not isinstance(value["reason"], str) or not value["reason"].strip():
            raise ValueError("abstain_schema")
    else:
        # All examples are explicitly final round; no fabricated extra round.
        raise ValueError("invalid_final_round_action")
    return value


def request(args, messages):
    key = os.environ.get("BIGMODEL_API_KEY")
    if not key:
        raise RuntimeError("BIGMODEL_API_KEY_missing")
    body = {"model": args.model, "messages": messages, "temperature": 0,
            "stream": False, "max_tokens": 8192,
            "thinking": {"type": "enabled"}, "reasoning_effort": "high",
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(args.endpoint, json.dumps(body, ensure_ascii=False).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")))
    started = time.monotonic()
    try:
        with opener.open(req, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Record only provider error code, never the raw error body/headers.
        try:
            err = json.loads(exc.read()).get("error", {})
            code = str(err.get("code", "unknown"))
            message = str(err.get("message", "")).replace(key, "[REDACTED]")[:300]
        except Exception:
            code, message = "unparsed", ""
        raise RuntimeError(f"http_{exc.code}_provider_{code}: {message}") from None
    return result, time.monotonic() - started


def label(args):
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    rows = [json.loads(l) for l in (out / "requests.jsonl").read_text().splitlines()]
    cache = {}
    cache_path = out / "api_receipts.jsonl"
    if cache_path.exists():
        cache = {r["id"]: r for r in (json.loads(l) for l in cache_path.read_text().splitlines())
                 if r.get("accepted")}
    completed = 0
    for row in rows:
        if row["id"] in cache:
            assert cache[row["id"]]["prompt_sha256"] == digest(row["prompt"])
            continue
        if args.limit and completed >= args.limit:
            break
        result, elapsed = request(args, row["teacher_messages"])
        choice = result["choices"][0]
        content = choice["message"].get("content") or ""
        record = {"id": row["id"], "provider": "bigmodel_standard_api", "endpoint": args.endpoint,
                  "requested_model": args.model, "returned_model": result.get("model"),
                  "response_id": result.get("id"), "usage": result.get("usage"),
                  "finish_reason": choice.get("finish_reason"), "elapsed_seconds": elapsed,
                  "prompt_sha256": digest(row["prompt"]), "content": content, "accepted": False}
        try:
            if choice.get("finish_reason") != "stop":
                raise ValueError("incomplete_response")
            record["target"] = validate_target(json.loads(content))
            record["accepted"] = True
        except (ValueError, TypeError) as exc:
            record["data_error"] = str(exc)[:100]
        with cache_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed += 1
        if not record["accepted"]:
            raise RuntimeError(f"invalid_teacher_response_for_{row['id']}: {record['data_error']}")
        cache[row["id"]] = record
        print(json.dumps({k: record[k] for k in ("id", "usage", "elapsed_seconds", "accepted")}), flush=True)
        materialize(out, manifest, rows, cache, args)
    materialize(out, manifest, rows, cache, args)


def materialize(out, manifest, rows, cache, args):
    splits = {"train": [], "val": []}
    for row in rows:
        if row["id"] not in cache:
            continue
        target = cache[row["id"]]["target"]
        item = {k: v for k, v in row.items() if k != "teacher_messages"}
        item["target"] = json.dumps(target, ensure_ascii=False)
        item["source_label"] = "fresh_GLM_API"
        item["label_prompt_sha256"] = cache[row["id"]]["prompt_sha256"]
        splits[row["split"]].append(item)
    for split, selected in splits.items():
        tmp = out / f"{split}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
        os.replace(tmp, out / f"{split}.jsonl")
    manifest.update(train=len(splits["train"]), val=len(splits["val"]),
                    new_api_calls=len(cache), labels_provider="bigmodel_standard_api",
                    teacher_model=args.model, teacher_endpoint=args.endpoint,
                    complete=len(cache) == len(rows),
                    action_counts={s: dict(Counter(json.loads(r["target"])["next_action"] for r in rr))
                                   for s, rr in splits.items()})
    atomic_json(out / "manifest.json", manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "label"])
    ap.add_argument("--source")
    ap.add_argument("--documents")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    # Use the ordinary API for data generation, not the coding-plan endpoint.
    ap.add_argument("--endpoint", default="https://open.bigmodel.cn/api/paas/v4/chat/completions")
    ap.add_argument("--model", default="glm-5.3")
    args = ap.parse_args()
    (prepare if args.mode == "prepare" else label)(args)


if __name__ == "__main__":
    main()
