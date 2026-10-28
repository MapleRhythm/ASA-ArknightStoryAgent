#!/usr/bin/env python3
"""Offline full20 paired plan, blinded review packets, and human-judgment analysis.

No model/library/network dependency; never generates or semantically grades text.
All run directories contain private evidence and must remain outside Git.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

# Frozen-input paths in a preregistration are relative to the experiment root.
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
ERROR_TYPES = ("actor_recipient", "identity", "phase_modality", "scene",
               "unsupported", "contradiction", "other")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, values):
    with Path(path).open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def task_identity(task):
    docs = task["documents"]
    require(len(docs) == 20, "must retain exactly 20 full documents")
    require(len({d["evidence_id"] for d in docs}) == 20, "duplicate evidence ID")
    prompt = task["bonsai_prompt"]
    for doc in docs:
        require(digest(doc["text"]) == doc["sha256"], "document digest mismatch")
        require(doc["text"] in prompt, "full document absent from prompt")
    require(digest(prompt) == task["bonsai_prompt_sha256"], "prompt digest mismatch")
    require(task["round"] in ("1/2", "2/2"), "unknown action round")
    return digest(canonical({k: task[k] for k in
                            ("id", "question", "round", "documents")}))


def check_frozen_inputs(config, rubric_path):
    """Refuse to plan unless the arm text, slot rule and slot file are the registered bytes."""
    frozen = config.get("frozen_inputs")
    if frozen is None:
        require(config["synthetic"], "real preregistration must declare frozen_inputs")
        return None
    for name in ("prompt_arm", "slot_rubric"):
        path = EXPERIMENT_ROOT / frozen[name]["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == frozen[name]["sha256"],
                name + " file differs from preregistered hash")
    source = frozen["prompt_arm"]
    arm_text = load(EXPERIMENT_ROOT / source["path"])[source["field"]]
    arms = {a["id"]: a["user_instruction_suffix"] for a in config["arms"]}
    require(arms["A"] == "" and arms[source["arm"]] == arm_text,
            "arm instruction differs from preregistered prompt_arm")
    slots = frozen["per_task_slots"]
    if slots["sha256"] is not None:
        require(rubric_path is not None, "preregistered per-task slot file required")
        require(digest(Path(rubric_path).read_text()) == slots["sha256"],
                "per-task slot file differs from preregistered hash")
    return frozen["slot_rubric"]["sha256"]


def select_panel(config, panel):
    """Each adapter panel is a separate paired experiment; panels are never pooled."""
    if panel is None:
        return config
    chosen = config["panels"][panel]
    config = {k: v for k, v in config.items() if k != "panels"}
    config.update(panel=panel, model_bundle=chosen["model_bundle"], generation=chosen["generation"])
    return config


def prepare(tasks_path, config_path, out, rubric_path=None, panel=None):
    config = load(config_path)
    rule_sha = check_frozen_inputs(config, rubric_path)
    config = select_panel(config, panel)
    tasks = rows(tasks_path)
    rubrics = rows(rubric_path) if rubric_path else []
    rubric = {r["task_id"]: r for r in rubrics}
    if rubric_path:
        require(len(rubric) == len(rubrics) and set(rubric) == {t["id"] for t in tasks},
                "rubric must predeclare all and only planned tasks")
        require(all(isinstance(r["necessary_slots"], list) and r["necessary_slots"] for r in rubrics),
                "nonempty predeclared slots required")
        require(rule_sha is None or all(r["slot_rubric_sha256"] == rule_sha for r in rubrics),
                "slots written under a different SLOT_RUBRIC")
    require(tasks and len({t["id"] for t in tasks}) == len(tasks), "empty or duplicate tasks")
    require(len(config["arms"]) == 2 and [a["id"] for a in config["arms"]] == ["A", "B"],
            "exactly preregistered A/B arms required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    config_sha = digest(canonical(config))
    requests = []
    for task in tasks:
        identity = task_identity(task)
        require(not any(k in task for k in ("target", "reference_answer")), "answer leak")
        for arm in config["arms"]:
            prompt = task["bonsai_prompt"]
            addition = arm["user_instruction_suffix"]
            if addition:
                # Add within the existing last user turn; never alter evidence or action schema.
                marker = "<|im_end|>\n<|im_start|>assistant"
                require(prompt.count(marker) == 1, "ambiguous final user boundary")
                prompt = prompt.replace(marker, "\n" + addition + marker)
            require(all(d["text"] in prompt for d in task["documents"]), "evidence changed")
            requests.append(dict(
                request_id=digest(config_sha + identity + arm["id"]), task_id=task["id"],
                arm=arm["id"], question=task["question"], round=task["round"],
                question_group=task["question_group"], documents=task["documents"],
                task_identity_sha256=identity, prompt=prompt, prompt_sha256=digest(prompt),
                original_prompt_sha256=task["bonsai_prompt_sha256"],
                generation=config["generation"], config_sha256=config_sha,
                model_bundle=config["model_bundle"], input_truncation=False,
                evidence_count=20, attempt=1, predeclared_rubric=rubric.get(task["id"])))
    random.Random(config["order_seed"]).shuffle(requests)
    write_rows(out / "requests.private.jsonl", requests)
    write(out / "config.json", config)
    write(out / "manifest.json", dict(config_sha256=config_sha,
          tasks_file_sha256=digest(Path(tasks_path).read_text()),
          requests_sha256=digest((out / "requests.private.jsonl").read_text()),
          task_count=len(tasks), question_groups=len({t["question_group"] for t in tasks}),
          requests=len(requests), executable_model_calls=False,
          slots_predeclared=bool(rubric_path),
          rubric_file_sha256=digest(Path(rubric_path).read_text()) if rubric_path else None,
          slot_rule_sha256=rule_sha, panel=panel,
          readiness="blocked_until_runtime_bundle_and_decode_gate_verified",
          synthetic=config["synthetic"], evidence_count=20))
    return {"planned_pairs": len(tasks), "model_calls": 0, "synthetic": config["synthetic"]}


def check_results(requests, values):
    by_id = {r["request_id"]: r for r in requests}
    seen = {}
    for value in values:
        rid = value["request_id"]
        require(rid in by_id and rid not in seen, "unknown or duplicate attempt/output")
        request = by_id[rid]
        for key in ("prompt_sha256", "config_sha256", "task_identity_sha256"):
            require(value[key] == request[key], key + " mismatch")
        require(value["generation"] == request["generation"], "decoding settings mismatch")
        require(value["model_bundle"] == request["model_bundle"], "model bundle mismatch")
        require(value["attempt"] == 1, "retries/best-of prohibited")
        require(value["evidence_count"] == 20 and value["input_truncated"] is False,
                "full20 invariant broken")
        require(value["status"] in ("ok", "generation_error", "timeout", "oom"), "bad status")
        if value["status"] == "ok":
            require(isinstance(value["raw_text"], str), "raw text missing")
            tokens = value["output_token_ids"]
            require(tokens and all(type(t) is int for t in tokens), "nonempty token IDs required")
            generation = request["generation"]
            require(len(tokens) <= generation["max_new_tokens"], "output budget exceeded")
            require(not any(t in generation["eos_token_ids"] for t in tokens[:-1]),
                    "decoder continued after EOS")
            reason = "stop" if tokens[-1] in generation["eos_token_ids"] else "length"
            require(value["finish_reason"] == reason, "stop condition mismatch")
            require(reason == "stop" or len(tokens) == generation["max_new_tokens"],
                    "early stop without EOS")
        seen[rid] = value
    return seen


def blind(run, results_path, out):
    run, out = Path(run), Path(out)
    requests = rows(run / "requests.private.jsonl")
    config = load(run / "config.json")
    require(digest(canonical(config)) == load(run / "manifest.json")["config_sha256"], "config changed")
    require(digest((run / "requests.private.jsonl").read_text()) ==
            load(run / "manifest.json")["requests_sha256"], "requests changed")
    seen = check_results(requests, rows(results_path))
    packets, key, forms = [], [], []
    order = list(requests)
    random.Random(config["blind_seed"]).shuffle(order)
    for index, request in enumerate(order, 1):
        rid = request["request_id"]
        value = seen.get(rid)
        review_id = "R" + str(index).zfill(5)
        failure = ("missing_output" if value is None else value["status"]
                   if value["status"] != "ok" else "length_limit"
                   if value["finish_reason"] == "length" else None)
        key.append(dict(review_id=review_id, request_id=rid, task_id=request["task_id"],
                        arm=request["arm"], question_group=request["question_group"],
                        execution_failure=failure,
                        output_sha256=digest(value["raw_text"]) if value and "raw_text" in value else None))
        # No arm, prompt variant, model identity, historic grade or reference target in reviewer packet.
        packets.append(dict(review_id=review_id, question=request["question"],
                            round=request["round"], documents=request["documents"],
                            raw_text=value.get("raw_text", "") if value else "",
                            execution_failure=failure,
                            predeclared_rubric=request["predeclared_rubric"]))
        forms.append(dict(review_id=review_id, reviewer_id=None, adjudicated=False,
                          all20_checked=False,
                          necessary_slots=(request["predeclared_rubric"] or {}).get("necessary_slots", []),
                          omitted_slots=[], unnecessary_supported=[],
                          material_errors=[], first_material_error_fact=None,
                          action_appropriate=None,
                          sufficient=None, source_eids=[], note=""))
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_rows(out / "review_packets.private.jsonl", packets)
    write_rows(out / "blind_key.private.jsonl", key)
    write_rows(out / "judgments.template.jsonl", forms)
    write(out / "manifest.json", dict(config_sha256=digest(canonical(config)),
          results_sha256=digest(Path(results_path).read_text()),
          key_sha256=digest((out / "blind_key.private.jsonl").read_text()),
          packets_sha256=digest((out / "review_packets.private.jsonl").read_text()),
          request_count=len(requests), synthetic=config["synthetic"],
          missing_outputs=sum(k["execution_failure"] == "missing_output" for k in key)))
    return {"review_items": len(key), "missing_counted_as_failure":
            sum(k["execution_failure"] == "missing_output" for k in key)}


def analyze(run, review, judgments_path):
    run, review = Path(run), Path(review)
    config = load(run / "config.json")
    manifest = load(review / "manifest.json")
    require(digest(canonical(config)) == manifest["config_sha256"], "config mismatch")
    require(digest((review / "blind_key.private.jsonl").read_text()) == manifest["key_sha256"],
            "blind key changed")
    require(digest((review / "review_packets.private.jsonl").read_text()) == manifest["packets_sha256"],
            "review evidence changed")
    key = rows(review / "blind_key.private.jsonl")
    judgments = rows(judgments_path)
    require(len(judgments) == len({j["review_id"] for j in judgments}), "duplicate judgment")
    by_review = {j["review_id"]: j for j in judgments}
    require(set(by_review) == {k["review_id"] for k in key}, "all planned items must remain")
    packet = {p["review_id"]: p for p in rows(review / "review_packets.private.jsonl")}
    pairs, group_pairs, diagnostics = defaultdict(dict), {}, {a: Counter() for a in ("A", "B")}
    for item in key:
        arm, rid = item["arm"], item["review_id"]
        judgment = by_review[rid]
        counts = diagnostics[arm]
        if item["execution_failure"]:
            complete = False
            counts[item["execution_failure"]] += 1
        else:
            require(judgment["adjudicated"] is True and bool(judgment["reviewer_id"]),
                    "human adjudication incomplete")
            require(judgment["all20_checked"] is True, "review must use all20, including missing-link claims")
            require(type(judgment["sufficient"]) is bool and type(judgment["action_appropriate"]) is bool,
                    "semantic decisions must be explicit booleans")
            require(isinstance(judgment["necessary_slots"], list) and judgment["necessary_slots"],
                    "necessary question slots must be specified")
            require(isinstance(judgment["omitted_slots"], list), "omissions must be explicit")
            predeclared = packet[rid]["predeclared_rubric"]
            if predeclared:
                require(judgment["necessary_slots"] == predeclared["necessary_slots"],
                        "predeclared slots changed after generation")
            require(set(judgment["omitted_slots"]) <= set(judgment["necessary_slots"]),
                    "omissions must identify necessary slots")
            errors = judgment["material_errors"]
            require(isinstance(errors, list), "errors must be explicit")
            for error in errors:
                require(error["scope"] in ("required", "extra"), "separate core from extra errors")
                require(error["type"] in ERROR_TYPES, "unknown factual error type")
                require(error.get("claim_locator") and error.get("explanation"), "error needs evidence rationale")
                counts[error["scope"] + "_" + error["type"]] += 1
            counts["with_extra_error"] += any(e["scope"] == "extra" for e in errors)
            counts["with_required_error"] += any(e["scope"] == "required" for e in errors)
            counts["with_omission"] += bool(judgment["omitted_slots"])
            # Supported but unnecessary facts measure scope expansion; they are not errors.
            counts["with_unnecessary_supported"] += bool(judgment.get("unnecessary_supported"))
            require(not judgment["sufficient"] or not judgment["omitted_slots"], "sufficient with omitted slot")
            valid_eids = {d["evidence_id"] for d in packet[rid]["documents"]}
            require(set(judgment["source_eids"]) <= valid_eids and judgment["source_eids"], "review source IDs missing")
            require(bool(judgment["note"]), "judgment rationale missing")
            complete = (judgment["sufficient"] and not judgment["omitted_slots"] and
                        not errors and judgment["action_appropriate"])
            # Secondary outcome: an instruction may shift answer/retrieve/abstain calibration.
            try:
                parsed = json.loads(packet[rid]["raw_text"])
                action = parsed.get("next_action") if isinstance(parsed, dict) else None
            except ValueError:
                counts["invalid_json_separate_from_content"] += 1
                action = None
            counts["action_" + (action if action in ("answer_directly", "retrieve_more", "abstain")
                                else "unparsed")] += 1
        counts["whole_answer_correct"] += complete
        counts["denominator"] += 1
        pairs[item["task_id"]][arm] = int(complete)
        group_pairs[item["task_id"]] = item["question_group"]
    require(all(set(p) == {"A", "B"} for p in pairs.values()), "paired denominator broken")
    transitions = Counter()
    groups = defaultdict(list)
    for tid, pair in pairs.items():
        transitions[("both_correct" if pair["A"] and pair["B"] else "gained" if pair["B"]
                     else "lost" if pair["A"] else "both_failed")] += 1
        groups[group_pairs[tid]].append(pair["B"] - pair["A"])
    delta = statistics.mean(p["B"] - p["A"] for p in pairs.values())
    # No p-values treating duplicate question/round variants as independent.
    return dict(synthetic=config["synthetic"], model_gain_claim_permitted=False,
                primary="all_planned_question_whole_answer_correct_paired_risk_difference",
                pairs=len(pairs), question_groups=len(groups), paired_difference_B_minus_A=delta,
                equal_question_group_weighted_difference=statistics.mean(statistics.mean(v) for v in groups.values()),
                transitions=dict(transitions), diagnostics={a: dict(v) for a, v in diagnostics.items()},
                semantic_judgments="external_human_not_automatic", independent_test=False,
                slots_predeclared=load(run / "manifest.json")["slots_predeclared"],
                config_sha256=digest(canonical(config)),
                judgments_sha256=digest(Path(judgments_path).read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tasks", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--rubric", help="private JSONL with task_id and necessary_slots, fixed before generation")
    p.add_argument("--panel", help="adapter panel from config['panels']; default is the top-level bundle")
    p.add_argument("--out", required=True)
    p = sub.add_parser("blind")
    p.add_argument("--run", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--run", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--judgments", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.tasks, args.config, args.out, args.rubric, args.panel)
    elif args.command == "blind":
        result = blind(args.run, args.results, args.out)
    else:
        result = analyze(args.run, args.review, args.judgments)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
