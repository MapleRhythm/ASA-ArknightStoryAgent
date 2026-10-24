"""Bounded Coding-Plan data expansion, offline audit, then frozen-backbone training."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import stat
import subprocess
import sys
import time

import annotation_protocol as p
from audit_fresh_subset import build, inspect_source, read_rows, require, verify as verify_subset

VERSION = "targeted_pt_v2_20261014"
KINDS = ("speaker_identity", "timeline_scope", "multi_question", "missing_evidence")
GEN_SYSTEM = (
    "你是剧情证据训练题设计者。只读当前20段全文，不用外部剧情记忆，不写答案或推理。"
    "为人物归属、时间场景、多问完整性和证据不足设计真实、自然、可泛化的新问题。"
)
REVIEW_EXTRA = (
    "\n本批重点：逐句核对说话者与受话者、人物/称号/器物、国家/组织、时间与地点，"
    "不允许把甲说的话或经历归给乙，不允许把不同时段合并。"
    "检查问题每个子问是否已覆盖；若仅有部分证据，可以回答明确支持的部分，但必须"
    "明确限定未能确认的部分，不能把遗漏当成完整回答。"
    "正确主干后追加的背景、目的、因果也必须由所绑定的证据支持。"
    "候选事实应简短、必要、无重复，通常1至3条；不要为了凑数扩写。"
    "缺证时不能因同名词、相似实体或邻近剧情而编造具体答案；1/2可补检索，2/2不可补检索。"
    "注意本批题目可能由模型设计，不能相信其隐含前提；以当前证据为准。"
    "对存在这些问题的候选将label_supported或decision_appropriate设false并说明问题。"
)
BASE_REVIEW = p.review_messages


def targeted_review(row, target):
    messages = BASE_REVIEW(row, target)
    messages[-1]["content"] += REVIEW_EXTRA
    return messages


class TargetRunner(p.Runner):
    def fingerprint(self, row):
        return p.sha(p.compact({"base": super().fingerprint(row), "version": VERSION,
                               "review_instructions_sha256": p.sha(REVIEW_EXTRA)}))


def check_questions(value, requested):
    if not isinstance(value, dict) or set(value) != {"questions"}:
        raise ValueError("question_schema")
    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != 3:
        raise ValueError("question_count")
    if Counter(q.get("kind") for q in questions if isinstance(q, dict)) != Counter(requested):
        raise ValueError("question_kinds")
    for q in questions:
        if (set(q) != {"question", "kind"} or not isinstance(q["question"], str)
                or not 5 <= len(q["question"].strip()) <= 300):
            raise ValueError("question_fields")
    if len({p.normalized_question(q["question"]) for q in questions}) != 3:
        raise ValueError("duplicate_question")
    return value


class QuestionRunner(TargetRunner):
    def process(self, row):
        p.check_row(row)
        fingerprint = self.fingerprint(row)
        prior = self.ledger.job(row["id"])
        if prior and prior[0] != fingerprint:
            raise ValueError("generation fingerprint changed")
        if prior and prior[1] in ("accepted", "quarantined"):
            return prior[1]
        kinds = row["requested_kinds"]
        messages = [
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": "完整证据：\n" +
             "\n".join(f"[{d['evidence_id']}]\n{d['text']}" for d in row["documents"]) +
             "\n请设计3个不同问题，其kind必须依次为：" + p.compact(kinds) +
             "\nspeaker_identity：围绕实际对白，区分谁说/对谁说、人物身份与称号或器物。"
             "\ntimeline_scope：区分前后阶段、发生地、经历主体；只能使用证据中的情节。"
             "\nmulti_question：一个自然问题含两个相关子问或对比两个人，当前证据足以支持两部分。"
             "\nmissing_evidence：以证据中真实实体为起点，问一个这些证据尚不能确认的"
             "具体事件/动机/关系。不要发明人名、组织名、荒诞事实或无根据的剧情前提。"
             "\n前两题必须可由当前证据完整回答，第三题应考查缺证而不是常识。"
             "\n不要使用已存在的问题：" + p.compact(row["avoid_questions"]) +
             '\n仅输出{"questions":[{"question":"新中文问题","kind":"指定类型"},...]}，'
             "不输出答案、暗示答案、解释、标签或引文。"}
        ]
        try:
            receipt = self.call(row, "question_generation", messages,
                                lambda x: check_questions(x, kinds))
            self.ledger.save_job(row["id"], fingerprint, "accepted", {"generation": receipt})
            return "accepted"
        except p.BudgetReached:
            self.ledger.save_job(row["id"], fingerprint, "pending", {"reason": "budget_or_stop"})
            self.halt(self.reason or "budget_limit")
            return "pending"
        except p.APIError as error:
            state = "pending" if error.terminal or error.retryable else "quarantined"
            self.ledger.save_job(row["id"], fingerprint, state, {"error": error.record()})
            return "failed" if error.retryable else state


def config(stage):
    return p.Settings(endpoint=p.CODING_PLAN_ENDPOINT, model="glm-5.3",
                      max_output_tokens=8192, workers=2, attempts=2, timeout=240,
                      max_api_calls=480 if stage == "questions" else 1600,
                      max_total_tokens=5_000_000 if stage == "questions" else 15_000_000,
                      review_fraction=1.0, seed=20261014, request_interval=1.0)


def prepare(base, out, packs):
    require(not out.exists(), "refusing to overwrite expansion")
    verified = verify_subset(base)
    train, val = read_rows(base / "train.jsonl"), read_rows(base / "val.jsonl")
    protected = {f for r in val for f in r["source_families"]}
    protected.update(("activities/act10d5", "activities/act10mini"))
    require(not protected & {f for r in train for f in r["source_families"]}, "base family leakage")
    grouped = {}
    for row in train:
        key = tuple(sorted(d["document_id"] for d in row["documents"]))
        grouped.setdefault(key, []).append(row)
    keys = sorted(grouped)
    random.Random(20261014).shuffle(keys)
    require(len(keys) >= packs > 0, "not enough unique evidence packs")
    seeds = []
    for i, key in enumerate(keys[:packs]):
        original = grouped[key][0]
        messages, prompt = p.prompt_for(original["question"], original["round"], original["documents"])
        # Deliberately copy no old target, action or review into question generation.
        row = {k: original[k] for k in ("question", "round", "documents", "source_families",
                                       "evidence_count", "truncated", "prompt_tokens")}
        row.update(id=f"pack-{i:04d}", split="train", teacher_messages=messages, prompt=prompt,
                   teacher_prompt_sha256=p.sha(p.compact(messages)), old_labels_used=False,
                   avoid_questions=[r["question"] for r in grouped[key]],
                   requested_kinds=[KINDS[i % 3], KINDS[(i + 1) % 3], "missing_evidence"])
        seeds.append(row)
    out.mkdir()
    qout = out / "questions"
    qout.mkdir()
    with (qout / "requests.jsonl").open("x") as stream:
        for row in seeds:
            stream.write(p.compact(row) + "\n")
    plan = dict(version=VERSION, base=str(base.resolve()), base_audit=verified,
                base_manifest_sha256=p.file_sha(base / "manifest.json"),
                base_train_sha256=p.file_sha(base / "train.jsonl"),
                base_val_sha256=p.file_sha(base / "val.jsonl"),
                packs=packs, planned_new_questions=packs * 3,
                protected_families=sorted(protected),
                qrequests_sha256=p.file_sha(qout / "requests.jsonl"),
                script_sha256=p.file_sha(__file__), protocol_sha256=p.file_sha(p.__file__),
                question_system_sha256=p.sha(GEN_SYSTEM),
                review_instructions_sha256=p.sha(REVIEW_EXTRA),
                teacher_endpoint=p.CODING_PLAN_ENDPOINT,
                old_targets_sent_to_question_generator=False,
                fixed_eval_questions_used_for_training=False, production_modified=False,
                max_combined_api_calls=2080, max_combined_accounted_tokens=20_000_000)
    p.atomic_json(out / "plan.json", plan)
    for stage in ("questions", "annotation"):
        p.atomic_json(out / f"{stage}.config.json", vars(config(stage)))
    return plan


def verify_plan(out):
    plan = json.loads((out / "plan.json").read_text())
    require(plan["script_sha256"] == p.file_sha(__file__), "expansion script changed")
    require(plan["protocol_sha256"] == p.file_sha(p.__file__), "native protocol changed")
    require(plan["qrequests_sha256"] == p.file_sha(out / "questions/requests.jsonl"), "packs changed")
    base = Path(plan["base"])
    require(plan["base_manifest_sha256"] == p.file_sha(base / "manifest.json"), "base changed")
    require(plan["base_train_sha256"] == p.file_sha(base / "train.jsonl")
            and plan["base_val_sha256"] == p.file_sha(base / "val.jsonl"), "base rows changed")
    verify_subset(base)
    return plan


def questions(out, limit):
    verify_plan(out)
    qout = out / "questions"
    rows = read_rows(qout / "requests.jsonl")
    with p.exclusive_run(qout):
        runner = QuestionRunner(qout, rows, config("questions"))
        try:
            status = runner.run(limit)
            require(not runner.reason, f"question generation stopped: {runner.reason}")
            return status
        finally:
            runner.ledger.close()


def build_requests(out, partial=False):
    plan = verify_plan(out)
    qout, aout = out / "questions", out / "annotation"
    require(not aout.exists(), "annotation already prepared")
    rows = read_rows(qout / "requests.jsonl")
    status = json.loads((qout / "status.json").read_text())
    require(partial or status["remaining_tasks"] == 0, "question generation incomplete")
    base = Path(plan["base"])
    used = {p.normalized_question(r["question"]) for split in ("train", "val")
            for r in read_rows(base / f"{split}.jsonl")}
    ledger = p.Ledger(qout / "ledger.sqlite3")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(__file__).parent / "official/tokenizer.json"))
    tasks, rejected = [], Counter()
    try:
        for seed in rows:
            job = ledger.job(seed["id"])
            if not job or job[1] != "accepted":
                continue
            receipt = job[2]["generation"]
            require(receipt["endpoint"] == p.CODING_PLAN_ENDPOINT, "wrong generation endpoint")
            for index, question in enumerate(receipt["parsed"]["questions"]):
                q = question["question"].strip()
                normalized = p.normalized_question(q)
                if normalized in used:
                    rejected["duplicate_question"] += 1
                    continue
                round_value = "1/2" if (int(seed["id"].split("-")[1]) + index) % 2 == 0 else "2/2"
                messages, prompt = p.prompt_for(q, round_value, seed["documents"])
                length = len(tok.encode(prompt, add_special_tokens=False).ids)
                if length + 8192 + 16 > 32768:
                    rejected["whole_sample_over_context"] += 1
                    continue
                row = dict(id=p.sha(p.compact({"q": q, "seed": seed["id"], "round": round_value}))[:24],
                           question=q, round=round_value, source_round=round_value,
                           source=str(qout), source_row_id=seed["id"], split="train",
                           source_families=seed["source_families"], documents=seed["documents"],
                           evidence_count=20, truncated=False, teacher_messages=messages,
                           teacher_prompt_sha256=p.sha(p.compact(messages)), prompt=prompt,
                           prompt_tokens=length, old_labels_used=False, targeted_kind=question["kind"],
                           question_generation_response_id=receipt["response_id"])
                require(not set(row["source_families"]) & set(plan["protected_families"]), "holdout leak")
                p.check_row(row)
                tasks.append(row)
                used.add(normalized)
        require(tasks, "no new questions")
        aout.mkdir()
        with (aout / "requests.jsonl").open("x") as stream:
            for row in tasks:
                stream.write(p.compact(row) + "\n")
        with (qout / "api_receipts.jsonl").open("x") as stream:
            for receipt in ledger.receipts():
                stream.write(p.compact(receipt) + "\n")
    finally:
        ledger.close()
    manifest = dict(version=p.VERSION, targeted_version=VERSION, kind="fresh_GLM_full20_annotation",
                    planned_train=len(tasks), planned_val=0, requested_train=plan["planned_new_questions"],
                    requested_val=0, train=0, val=0, complete=False, old_labels_used=False,
                    evidence_count_per_example=20, truncated_documents=0,
                    train_val_story_overlap=[], train_val_source_family_overlap=[],
                    train_val_document_overlap=[], train_val_question_overlap=[],
                    requests_sha256=p.file_sha(aout / "requests.jsonl"),
                    max_output_tokens_reserved=8192, max_context_tokens=32768,
                    families={"train": sorted({f for r in tasks for f in r["source_families"]}), "val": []},
                    protected_families=plan["protected_families"], rejected=dict(rejected),
                    targeted_counts=dict(Counter(r["targeted_kind"] for r in tasks)),
                    review_instructions_sha256=p.sha(REVIEW_EXTRA), not_human_gold=True,
                    source_index_sha256=json.loads((base / "source/manifest.json").read_text())["source_index_sha256"],
                    review_required_before_training=True, production_modified=False)
    p.atomic_json(aout / "manifest.json", manifest)
    return manifest


def annotate(out, limit):
    verify_plan(out)
    aout = out / "annotation"
    p.review_messages = targeted_review
    with p.exclusive_run(aout):
        rows = p.load_requests(aout)
        runner = TargetRunner(aout, rows, config("annotation"))
        try:
            status = runner.run(limit)
            runner.export()
            require(not runner.reason, f"annotation stopped: {runner.reason}")
            return status
        finally:
            runner.ledger.close()


def verify_reviews(source, rows):
    settings = config("annotation")
    receipts = read_rows(source / "api_receipts.jsonl")
    accepted = {(r["task_id"], r["cache_key"]): r for r in receipts
                if r.get("accepted") and r["phase"] == "review"}
    for row in rows:
        target = json.loads(row["target"])
        messages = targeted_review(row, target)
        key = p.sha(p.compact({"phase": "review", "messages": messages,
                              "settings": settings.cache_settings()}))
        receipt = accepted.get((row["id"], key))
        review = row["semantic_review"]
        require(receipt and receipt["endpoint"] == p.CODING_PLAN_ENDPOINT
                and receipt["requested_model"] == settings.model
                and receipt["teacher_prompt_sha256"] == row["teacher_prompt_sha256"]
                and receipt["parsed"] == review and json.loads(receipt["content"]) == review,
                "second-pass review receipt mismatch")
        require(row["semantic_review_sampled"] and review and review["label_supported"]
                and review["decision_appropriate"], "missing or failed semantic review")


def merge(out):
    plan = verify_plan(out)
    source = out / "annotation"
    manifest, rows, _ = inspect_source(source)
    require(manifest["review_fraction"] == 1.0, "all new labels must be reviewed")
    verify_reviews(source, rows["train"])
    counts = Counter(r["targeted_kind"] for r in rows["train"])
    require(len(rows["train"]) >= 350 and all(counts[k] >= 50 for k in KINDS),
            f"insufficient reviewed new data: {dict(counts)}")
    decisions = dict(excluded={}, additional_retained_spot_checks={},
                     labels_manually_rewritten=0, not_QA_accuracy=True,
                     scope="All new labels independently requested in a second GLM pass with targeted "
                           "semantic review instructions. Same teacher model, not independent human gold. "
                           "Only labels passing both support and action review retained; no label repair.")
    p.atomic_json(out / "review_decisions.json", decisions)
    reviewed = out / "reviewed_new"
    build(source, out / "review_decisions.json", reviewed)
    base = Path(plan["base"])
    merged = out / "merged_data"
    merged.mkdir()
    train = read_rows(base / "train.jsonl") + read_rows(reviewed / "train.jsonl")
    val = read_rows(base / "val.jsonl")
    require(len({r["id"] for r in train + val}) == len(train + val), "duplicate IDs")
    require(len({p.normalized_question(r["question"]) for r in train}) ==
            len({p.normalized_question(r["question"]) for r in read_rows(base / "train.jsonl")}) +
            len(rows["train"]), "new questions duplicated")
    p.assert_split({"train": train, "val": val})
    with (merged / "train.jsonl").open("x") as stream:
        for row in train:
            stream.write(p.compact(row) + "\n")
    shutil.copy2(base / "val.jsonl", merged / "val.jsonl")
    receipt = dict(kind="audited_targeted_merged_data_v2", ready_for_training=True,
                   base=str(base), new_data=str(reviewed), expansion_plan_sha256=p.file_sha(out / "plan.json"),
                   base_audit_sha256=verify_subset(base)["audit_sha256"],
                   new_audit_sha256=verify_subset(reviewed)["audit_sha256"],
                   train=len(train), val=len(val), new_train=len(rows["train"]),
                   targeted_counts=dict(counts),
                   data_sha256={s: p.file_sha(merged / f"{s}.jsonl") for s in ("train", "val")},
                   evidence_count_per_example=20, truncated_documents=0, train_val_story_overlap=[],
                   train_val_source_family_overlap=[], train_val_document_overlap=[],
                   train_val_question_overlap=[], review_required_before_training=False,
                   all_new_labels_second_pass_reviewed=True, not_human_gold=True,
                   all_retained_labels_semantically_certified=False, production_modified=False)
    p.atomic_json(merged / "manifest.json", receipt)
    return verify_merged(merged)


def verify_merged(data):
    manifest = json.loads((data / "manifest.json").read_text())
    require(manifest["kind"] == "audited_targeted_merged_data_v2", "wrong merged data")
    base, new = Path(manifest["base"]), Path(manifest["new_data"])
    verify_plan(data.parent)
    require(verify_subset(base)["audit_sha256"] == manifest["base_audit_sha256"], "base audit changed")
    require(verify_subset(new)["audit_sha256"] == manifest["new_audit_sha256"], "new audit changed")
    verify_reviews(new / "source", read_rows(new / "train.jsonl"))
    for split in ("train", "val"):
        require(p.file_sha(data / f"{split}.jsonl") == manifest["data_sha256"][split], "data changed")
        expected = read_rows(base / f"{split}.jsonl")
        if split == "train":
            expected += read_rows(new / "train.jsonl")
        require(read_rows(data / f"{split}.jsonl") == expected, "merged rows not originals")
    require(p.file_sha(data / "val.jsonl") == p.file_sha(base / "val.jsonl"), "fixed eval changed")
    p.assert_split({s: read_rows(data / f"{s}.jsonl") for s in ("train", "val")})
    return manifest


def train_main():
    """Same initial encoder, optimizer and one epoch as v1, with merged-data preflight."""
    import train_pilot
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--memory-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args(sys.argv[2:])
    manifest = verify_merged(args.data)
    gate = json.loads(args.memory_gate.read_text())
    require(gate["completed"] and gate["activation_offload"]
            and gate["data_sha256"] == manifest["data_sha256"]
            and gate["backend_sha256"] == p.file_sha(Path(__file__).with_name("backend.py"))
            and gate["prompt_gradient_norm"] > 0, "new-data memory/gradient gate failed")
    receipt = dict(kind="targeted_merged_training_preflight", manifest_sha256=p.file_sha(args.data / "manifest.json"),
                   trainer_sha256=p.file_sha(train_pilot.__file__), data=manifest,
                   memory_gate_sha256=p.file_sha(args.memory_gate), initialized_from_v1=False,
                   same_initialization_as_v1=True, production_modified=False)
    receipt_path = args.out.with_suffix(".preflight.json")
    if args.resume:
        require(json.loads(receipt_path.read_text()) == receipt, "resume preflight changed")
    else:
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    train_pilot.main()
    run = json.loads((args.out / "run.json").read_text())
    require(run["training_complete"] and run["backbone_bytewise_unchanged"], "training incomplete")
    run["targeted_merged_preflight"] = receipt
    p.atomic_json(args.out / "run.json", run)


def workflow(out, gpu):
    require(os.environ.get("BIGMODEL_API_KEY"), "missing API credential")
    verify_plan(out)
    with (out / ".workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(pid=os.getpid(), stage="starting", started_unix=time.time(), gpu=gpu,
                      production_modified=False, other_jobs_stopped=False)

        def save(stage, **kwargs):
            status.update(stage=stage, updated_unix=time.time(), **kwargs)
            p.atomic_json(out / "workflow.json", status)

        root = Path(__file__).resolve().parent
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="8",
                   PYTORCH_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
        try:
            save("question_generation")
            questions(out, 0)
            save("prepare_annotation")
            if not (out / "annotation").exists():
                build_requests(out)
            else:
                p.load_requests(out / "annotation")
            save("annotation_and_all_label_review")
            annotate(out, 0)
            save("offline_audit_and_merge")
            merged = (verify_merged(out / "merged_data") if (out / "merged_data").exists()
                      else merge(out))
            save("data_ready", train_examples=merged["train"], new_examples=merged["new_train"])
            free = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                       "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
            require(free >= 20500, f"GPU {gpu} insufficient headroom: {free} MiB")
            data, gate, run = out / "merged_data", out / "memory_gate.json", out / "trained_v2"
            commands = [
                ("memory_gradient_gate", [sys.executable, "-u", "memory_gate.py", "--data", str(data),
                                         "--out", str(gate), "--offload"]),
                ("training", [sys.executable, "-u", str(Path(__file__).resolve()), "train", "--data",
                              str(data), "--out", str(run), "--memory-gate", str(gate),
                              "--gate", "full_gate_bfloat16_bounded_20261013.json",
                              "--offload", "--epochs", "1", "--lr", "0.00002",
                              "--validation-limit", "0", "--skip-initial-validation"]),
            ]
            for stage, command in commands:
                if stage == "memory_gradient_gate" and gate.exists():
                    previous_gate = json.loads(gate.read_text())
                    require(previous_gate["completed"] and
                            previous_gate["data_sha256"] == merged["data_sha256"],
                            "existing memory gate failed; inspect before retry")
                    continue
                if stage == "training" and run.exists():
                    previous_run = json.loads((run / "run.json").read_text())
                    if previous_run["training_complete"]:
                        require(previous_run["backbone_bytewise_unchanged"] and
                                previous_run["data_sha256"] == merged["data_sha256"],
                                "existing completed run mismatch")
                        continue
                    require((run / "training_state.pt").exists(), "no resumable optimizer checkpoint")
                    command.append("--resume")
                with (out / f"{stage}.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    save(stage, child_pid=child.pid)
                    code = child.wait()
                require(code == 0, f"{stage} exit {code}; see stage log")
            save("training_completed", child_pid=None, completed_unix=time.time(),
                 fixed46_comparison_pending=True, qa_accuracy_not_yet_measured=True)
        except Exception as error:
            save("blocked_or_failed", failed_stage=status["stage"], error=str(error), child_pid=None)
            raise


def read_credential(path):
    """Only the pre-existing literal assignment, with no shell execution or logging."""
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
            and not info.st_mode & 0o077 and info.st_size <= 65536,
            "unsafe existing credential file permissions")
    import re
    found = []
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?BIGMODEL_API_KEY\s*=(.*)$", line)
        if match:
            values = shlex.split(match[1], comments=True)
            require(len(values) == 1 and values[0] and not any(c.isspace() for c in values[0]),
                    "credential must be one literal assignment")
            found.append(values[0])
    require(len(found) == 1, "missing or ambiguous existing GLM credential")
    return found[0]


def launch(out, gpu):
    verify_plan(out)
    require(not (out / "workflow.json").exists() and not (out / "workflow.log").exists(),
            "workflow already exists; inspect before resuming")
    require(os.environ.get("BIGMODEL_API_KEY"), "missing credential")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "workflow",
               "--out", str(out), "--gpu", str(gpu)]
    with (out / "workflow.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=Path(__file__).resolve().parent, env=dict(os.environ),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    receipt = dict(workflow_pid=proc.pid, gpu_for_later_training=gpu, out=str(out),
                   stage="launched_data_generation", credential_persisted=False,
                   production_modified=False)
    p.atomic_json(out / "launch.json", receipt)
    return receipt


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "questions", "build-requests", "annotate",
                                           "merge", "verify", "workflow", "launch"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--packs", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--credential-env-file", type=Path, default=Path("${REMOTE_HOME}/.bigmodel_env"))
    args = parser.parse_args()
    if args.command in ("questions", "annotate", "workflow", "launch"):
        if not os.environ.get("BIGMODEL_API_KEY"):
            os.environ["BIGMODEL_API_KEY"] = read_credential(args.credential_env_file)
    out = args.out.resolve()
    actions = {"prepare": lambda: prepare(args.base.resolve(), out, args.packs),
               "questions": lambda: questions(out, args.limit),
               "build-requests": lambda: build_requests(out, args.partial),
               "annotate": lambda: annotate(out, args.limit),
               "merge": lambda: merge(out), "verify": lambda: verify_merged(out),
               "workflow": lambda: workflow(out, args.gpu), "launch": lambda: launch(out, args.gpu)}
    print(p.compact(actions[args.command]()), flush=True)


if __name__ == "__main__":
    main()
