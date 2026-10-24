"""New data only, isolated from the running 750-example deep-prefix experiment.

Reuse the audited Coding Plan transport, budgets, SQLite recovery, native
protocol and subset auditor. Never mutate old scripts/datasets or start training.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import annotation_protocol as p
from audit_fresh_subset import build, inspect_source, read_rows, require, verify as verify_subset
import targeted_pt_v2 as previous


ROOT = Path(__file__).resolve().parent
VERSION = "reader_error_expansion_v3_20261017"
KINDS = ("speaker_direction", "event_state_scope", "minimal_complete",
         "partial_support", "evidence_boundary")
GEN_SYSTEM = (
    "你是中文剧情证据问答训练题设计者，只依据本次20段完整证据设计自然问题。"
    "不使用外部剧情记忆，不写答案、解释或引文，不输出思维链。"
)
GEN_GUIDE = (
    "\nspeaker_direction：针对实际对白，询问谁对谁表达何种态度、谁交付给谁或人物/称号/"
    "器物关系。应有明确证据可回答，不强迫给未标明的说话人命名。"
    "\nevent_state_scope：考查已发生/拟议/猜测/演示、发现/未发现、做过/未做过、杀伤/"
    "杀死、前后阶段及派别差异中的一种。问题应自然，不在问题里指出标准结论。"
    "\nminimal_complete：问有明确依据的核心动机/机制/关系；容易联想的额外背景不应"
    "成为所问重点，训练答案完整覆盖必要点后停止，不凑无关背景。"
    "\npartial_support：同一个自然问题有两个相关子问，当前证据明确支持一部分，"
    "但没有建立另一部分具体原因/身份/关系。两部分均围绕证据中真实实体，不发明事件。"
    "\nevidence_boundary：沿用前面可回答题中的真实实体，换问当前20段尚未建立的"
    "具体动机、因果、施受方向或身份；附近应有容易张冠李戴的相似信息，但不能"
    "凭相似信息推出答案。不要发明人名或荒诞前提，不刻意假设某事已经发生。"
    "\n所有题像用户一样直接问；不要用'证据不足'、'能否确认'等措辞泄露设计意图。"
    "类型仅用于设计和后续统计，不是答案动作金标，标注者必须重新独立判断。"
)
ANNOTATION_EXTRA = (
    "\n以下仅为离线教师质量要求，不修改学生原生提示或添加剧情证据："
    "先确保回答所问人物、对象、事件阶段和每个子问；最终只输出必要的简短原子事实，"
    "通常1至3条，确实需要时可以更多。必要内容答全后停止，不加无关背景或总结。"
    "严格区分说话者/受话者、给予者/接受者、人物/称号/器物、不同派别；"
    "区分否定与肯定、已发生与计划/猜测/演示、杀伤与杀死，不能扩大断言。"
    "同一实体出现在证据中不等于所问关系成立；不得以邻近事实替代缺失的因果。"
    "只有部分子问有据时，准确表达有据部分和未能确认的范围，遵守原协议动作和轮次；"
    "不要用全盘拒答抹掉可回答部分，也不要用完整作答口吻掩盖缺口。"
    "拒答或补检索的说明也不能添加新的剧情断言。不要输出思维链。"
)
REVIEW_EXTRA = (
    "\n本轮复核须独立读证据，不相信候选标签或问题的前提。"
    "逐句检查人物主体、受话/受事对象、施受方向、称号/器物指代、否定极性、"
    "发生/计划/演示/推测状态和时间阶段。问到谁就检查谁，不以近邻人名补全。"
    "核对每个子问是否完整覆盖；证据只支持部分时应准确限定缺失部分，而不是强答"
    "或抹掉已知信息；补检索/拒答仍需与当前轮次及原生协议一致。"
    "核心正确但增加无据背景/因果、断言过强、错配人物、漏答子问或错误拒答，"
    "都必须判不通过，不以轻微/大意正确放行；拒答理由出现无据事实同样不通过。"
    "同一实体或相似句子不构成所问具体关系的证据。"
    "没有问题时issues必须为空；存在问题时明确记入issues，不重写标签。"
)
CODE_FILES = ("expand_reader_errors_v3.py", "annotation_protocol.py",
              "audit_fresh_subset.py", "targeted_pt_v2.py")


def settings(stage, smoke=False):
    return p.Settings(
        endpoint=p.CODING_PLAN_ENDPOINT, model="glm-5.3", max_output_tokens=8192,
        workers=2, attempts=2, timeout=240, review_fraction=1.0, seed=20261017,
        request_interval=1.0,
        max_api_calls=(12 if stage == "questions" else 64) if smoke else
                      (480 if stage == "questions" else 1800),
        max_total_tokens=(200_000 if stage == "questions" else 800_000) if smoke else
                         (5_000_000 if stage == "questions" else 15_000_000),
    )


def annotation_messages(messages):
    result = [dict(m) for m in messages]
    result[0]["content"] += ANNOTATION_EXTRA
    return result


def review_messages(row, target):
    messages = p.review_messages(row, target)
    messages[-1]["content"] += REVIEW_EXTRA
    return messages


def generation_messages(row):
    return [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": "完整证据：\n" +
         "\n".join(f"[{d['evidence_id']}]\n{d['text']}" for d in row["documents"]) +
         "\n按顺序设计3道不同问题，kind分别为：" + p.compact(row["requested_kinds"]) +
         GEN_GUIDE + "\n避免以下已有问题：" + p.compact(row["avoid_questions"]) +
         '\n只输出{"questions":[{"question":"新问题","kind":"指定类型"},...]}。'},
    ]


def check_questions(value, requested):
    previous.check_questions(value, requested)
    if [q["kind"] for q in value["questions"]] != requested:
        raise ValueError("question_order")
    return value


class Runner(p.Runner):
    def fingerprint(self, row):
        return p.sha(p.compact({
            "base": super().fingerprint(row), "version": VERSION,
            "generation": p.sha(GEN_SYSTEM + GEN_GUIDE),
            "annotation": p.sha(ANNOTATION_EXTRA), "review": p.sha(REVIEW_EXTRA),
            "requested_kinds": row.get("requested_kinds"),
        }))

    def process(self, row):
        p.check_row(row)
        fingerprint = self.fingerprint(row)
        prior = self.ledger.job(row["id"])
        require(not prior or prior[0] == fingerprint, "job fingerprint changed")
        if prior and prior[1] in ("accepted", "quarantined"):
            return prior[1]
        try:
            if "requested_kinds" in row:
                response = self.call(row, "question_generation", generation_messages(row),
                                     lambda v: check_questions(v, row["requested_kinds"]))
                result, accepted = {"generation": response}, True
            else:
                label = self.call(row, "annotate", annotation_messages(row["teacher_messages"]),
                                  lambda v: p.check_target(v, row))
                review = self.call(row, "review", review_messages(row, label["parsed"]), p.check_review)
                accepted = (review["parsed"]["label_supported"]
                            and review["parsed"]["decision_appropriate"]
                            and not review["parsed"]["issues"])
                result = dict(label=label, review=review, semantic_review_sampled=True,
                              semantic_review_is_independent_human_gold=False)
            state = "accepted" if accepted else "quarantined"
            self.ledger.save_job(row["id"], fingerprint, state, result)
            return state
        except p.BudgetReached:
            self.ledger.save_job(row["id"], fingerprint, "pending",
                                 {"reason": self.reason or "budget_or_stop"})
            self.halt(self.reason or "budget_limit")
            return "pending"
        except p.APIError as error:
            state = "pending" if error.terminal or error.retryable else "quarantined"
            self.ledger.save_job(row["id"], fingerprint, state, {"error": error.record()})
            return "failed" if error.retryable else state


def kind_schedule(index):
    return ["speaker_direction" if index % 2 == 0 else "event_state_scope",
            "minimal_complete" if (index // 2) % 2 == 0 else "partial_support",
            "evidence_boundary"]


def make_seeds(train, val, packs):
    protected = {f for row in val for f in row["source_families"]}
    protected.update(("activities/act10d5", "activities/act10mini"))
    val_docs = {d["document_id"] for row in val for d in row["documents"]}
    grouped = {}
    for row in train:
        require(not set(row["source_families"]) & protected, "protected family in training")
        require(not {d["document_id"] for d in row["documents"]} & val_docs, "held-out document")
        grouped.setdefault(tuple(sorted(d["document_id"] for d in row["documents"])), []).append(row)
    keys = sorted(grouped)
    random.Random(20261017).shuffle(keys)
    # Favor less-questioned packs; round-robin families limits domination by a
    # few heavily represented activities, without inspecting evaluation stories.
    buckets = {}
    for key in keys:
        family = tuple(sorted(grouped[key][0]["source_families"]))
        buckets.setdefault(family, []).append(key)
    for family in buckets:
        buckets[family].sort(key=lambda k: len(grouped[k]))
    order = []
    while buckets:
        for family in list(buckets):
            order.append(buckets[family].pop(0))
            if not buckets[family]:
                del buckets[family]
    require(0 < packs <= len(order), "not enough distinct training evidence packs")
    seeds = []
    for index, key in enumerate(order[:packs]):
        original = grouped[key][0]
        # Explicit allow-list: no old answers, reviews or evaluation notes.
        row = {k: original[k] for k in ("question", "round", "documents", "source_families",
                                       "evidence_count", "truncated", "prompt_tokens")}
        messages, prompt = p.prompt_for(row["question"], row["round"], row["documents"])
        row.update(id=f"pack-{index:04d}", split="train",
                   teacher_messages=messages, prompt=prompt,
                   teacher_prompt_sha256=p.sha(p.compact(messages)), old_labels_used=False,
                   avoid_questions=[r["question"] for r in grouped[key]],
                   requested_kinds=kind_schedule(index))
        p.check_row(row)
        seeds.append(row)
    return seeds, sorted(protected), len(order)


def prepare(base, out, packs, smoke=False):
    require(not out.exists(), "refusing to overwrite expansion")
    previous.verify_merged(base)
    train, val = (read_rows(base / f"{s}.jsonl") for s in ("train", "val"))
    seeds, protected, available = make_seeds(train, val, packs)
    out.mkdir()
    (out / "questions").mkdir()
    with (out / "questions/requests.jsonl").open("x") as stream:
        for row in seeds:
            stream.write(p.compact(row) + "\n")
    # Only generic error categories reach the teacher. No fixed-eval questions,
    # answers, entities, documents or adjudication notes are supplied.
    plan = dict(
        version=VERSION, base=str(base), mode="smoke" if smoke else "formal",
        packs=packs, available_unique_training_packs=available, planned_new_questions=packs * 3,
        requested_kind_counts=dict(Counter(k for r in seeds for k in r["requested_kinds"])),
        base_data_sha256={s: p.file_sha(base / f"{s}.jsonl") for s in ("train", "val")},
        base_manifest_sha256=p.file_sha(base / "manifest.json"),
        qrequests_sha256=p.file_sha(out / "questions/requests.jsonl"),
        code_sha256={name: p.file_sha(ROOT / name) for name in CODE_FILES},
        protected_families=protected, teacher_endpoint=p.CODING_PLAN_ENDPOINT,
        diagnosis="750-example shallow prompt run; not yet the running deep-prefix result",
        old_targets_sent_to_teacher=False, fixed_eval_content_sent_to_teacher=False,
        questions_not_new_story_corpus=True, production_modified=False,
        running_training_dataset_changed=False, auto_training_enabled=False,
        budget={stage: vars(settings(stage, smoke)) for stage in ("questions", "annotation")},
    )
    p.atomic_json(out / "plan.json", plan)
    return plan


def verify_plan(out):
    plan = json.loads((out / "plan.json").read_text())
    require(plan["version"] == VERSION, "wrong expansion version")
    for name, digest in plan["code_sha256"].items():
        require(p.file_sha(ROOT / name) == digest, f"code changed: {name}")
    require(plan["qrequests_sha256"] == p.file_sha(out / "questions/requests.jsonl"),
            "question seeds changed")
    base = Path(plan["base"])
    require(plan["base_manifest_sha256"] == p.file_sha(base / "manifest.json"), "base manifest changed")
    require(plan["base_data_sha256"] == {s: p.file_sha(base / f"{s}.jsonl")
                                        for s in ("train", "val")}, "base data changed")
    previous.verify_merged(base)
    for stage in ("questions", "annotation"):
        require(plan["budget"][stage] == vars(settings(stage, plan["mode"] == "smoke")),
                "budget/settings changed")
    return plan


def run_stage(out, stage, limit=0):
    plan = verify_plan(out)
    directory = out / stage
    rows = read_rows(directory / "requests.jsonl") if stage == "questions" else p.load_requests(directory)
    with p.exclusive_run(directory):
        runner = Runner(directory, rows, settings(stage, plan["mode"] == "smoke"))
        try:
            status = runner.run(limit)
            if stage == "annotation":
                runner.export()
            require(not runner.reason, f"{stage} stopped: {runner.reason}")
            return status
        finally:
            runner.ledger.close()


def build_requests(out):
    from tokenizers import Tokenizer
    plan = verify_plan(out)
    qout, aout = out / "questions", out / "annotation"
    require(not aout.exists(), "annotation requests already exist")
    status = json.loads((qout / "status.json").read_text())
    require(status["remaining_tasks"] == 0 and not status["stop_reason"], "questions unfinished")
    base = Path(plan["base"])
    used = {p.normalized_question(r["question"]) for s in ("train", "val")
            for r in read_rows(base / f"{s}.jsonl")}
    tokenizer = Tokenizer.from_file(str(ROOT / "official/tokenizer.json"))
    tasks, rejected = [], Counter()
    ledger = p.Ledger(qout / "ledger.sqlite3")
    try:
        for seed in read_rows(qout / "requests.jsonl"):
            job = ledger.job(seed["id"])
            if not job or job[1] != "accepted":
                continue
            receipt = job[2]["generation"]
            expected_key = p.sha(p.compact(dict(
                phase="question_generation", messages=generation_messages(seed),
                settings=settings("questions", plan["mode"] == "smoke").cache_settings())))
            require(receipt["cache_key"] == expected_key
                    and receipt["endpoint"] == p.CODING_PLAN_ENDPOINT, "wrong generation request")
            check_questions(receipt["parsed"], seed["requested_kinds"])
            for index, question in enumerate(receipt["parsed"]["questions"]):
                q = question["question"].strip()
                normalized = p.normalized_question(q)
                if normalized in used:
                    rejected["duplicate_question"] += 1
                    continue
                round_value = "1/2" if (int(seed["id"].split("-")[1]) + index) % 2 == 0 else "2/2"
                messages, prompt = p.prompt_for(q, round_value, seed["documents"])
                tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
                if tokens + 8192 + 16 > 32768:
                    rejected["whole_sample_over_context"] += 1
                    continue
                row = dict(
                    id=p.sha(p.compact(dict(version=VERSION, q=q, seed=seed["id"],
                                           round=round_value)))[:24],
                    question=q, round=round_value, source_round=round_value,
                    source=str(qout), source_row_id=seed["id"], split="train",
                    source_families=seed["source_families"], documents=seed["documents"],
                    evidence_count=20, truncated=False, teacher_messages=messages,
                    teacher_prompt_sha256=p.sha(p.compact(messages)), prompt=prompt,
                    prompt_tokens=tokens, old_labels_used=False, targeted_kind=question["kind"],
                    question_generation_response_id=receipt["response_id"],
                )
                require(not set(row["source_families"]) & set(plan["protected_families"]),
                        "held-out family leaked")
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
    # Retain the original source-index provenance from the audited base lineage.
    old_plan = previous.verify_plan(base.parent)
    source_manifest = json.loads((Path(old_plan["base"]) / "source/manifest.json").read_text())
    manifest = dict(
        version=p.VERSION, targeted_version=VERSION, kind="fresh_GLM_full20_annotation",
        planned_train=len(tasks), planned_val=0, requested_train=plan["planned_new_questions"],
        requested_val=0, train=0, val=0, complete=False, old_labels_used=False,
        evidence_count_per_example=20, truncated_documents=0,
        train_val_story_overlap=[], train_val_source_family_overlap=[],
        train_val_document_overlap=[], train_val_question_overlap=[],
        requests_sha256=p.file_sha(aout / "requests.jsonl"),
        max_output_tokens_reserved=8192, max_context_tokens=32768,
        families=dict(train=sorted({f for row in tasks for f in row["source_families"]}), val=[]),
        protected_families=plan["protected_families"], rejected=dict(rejected),
        targeted_counts=dict(Counter(r["targeted_kind"] for r in tasks)),
        source_index_sha256=source_manifest["source_index_sha256"],
        review_required_before_training=True, not_human_gold=True, production_modified=False,
        annotation_instructions_sha256=p.sha(ANNOTATION_EXTRA),
        review_instructions_sha256=p.sha(REVIEW_EXTRA),
    )
    p.atomic_json(aout / "manifest.json", manifest)
    return manifest


def verify_reviews(source, rows, smoke=False):
    receipts = read_rows(source / "api_receipts.jsonl")
    accepted = {(r["task_id"], r["cache_key"]): r for r in receipts if r.get("accepted")}
    config = settings("annotation", smoke).cache_settings()
    for row in rows:
        native, _ = p.prompt_for(row["question"], row["round"], row["documents"])
        for phase, messages in (
            ("annotate", annotation_messages(native)),
            ("review", review_messages(row, json.loads(row["target"]))),
        ):
            key = p.sha(p.compact(dict(phase=phase, messages=messages, settings=config)))
            receipt = accepted.get((row["id"], key))
            require(receipt and receipt["phase"] == phase
                    and receipt["endpoint"] == p.CODING_PLAN_ENDPOINT
                    and receipt["requested_model"] == "glm-5.3"
                    and receipt["teacher_prompt_sha256"] == row["teacher_prompt_sha256"],
                    f"{phase} request lineage mismatch")
            expected = json.loads(row["target"]) if phase == "annotate" else row["semantic_review"]
            require(receipt["parsed"] == expected and json.loads(receipt["content"]) == expected,
                    "receipt does not match retained label/review")
            if phase == "annotate":
                require(receipt["response_id"] == row["annotation_response_id"], "response id mismatch")
        review = row["semantic_review"]
        require(row["semantic_review_sampled"] and review["label_supported"]
                and review["decision_appropriate"] and not review["issues"], "failed/warned review")


def audit_and_merge(out):
    plan = verify_plan(out)
    require(plan["mode"] == "formal", "smoke data must never merge")
    source = out / "annotation"
    _, rows, _ = inspect_source(source)
    verify_reviews(source, rows["train"])
    counts = Counter(row["targeted_kind"] for row in rows["train"])
    require(len(rows["train"]) >= 350 and all(counts[k] >= 40 for k in KINDS),
            f"Insufficient reviewed coverage: {dict(counts)}")
    decisions = dict(
        excluded={}, additional_retained_spot_checks={}, labels_manually_rewritten=0,
        not_QA_accuracy=True, scope="All labels re-read against complete evidence by a second "
        "GLM request. Require support, appropriate action and empty issues. No label repair; "
        "same-model review is not independent human gold. Separate assistant spot-check pending.")
    p.atomic_json(out / "review_decisions.json", decisions)
    reviewed = out / "reviewed_new"
    build(source, out / "review_decisions.json", reviewed)
    base = Path(plan["base"])
    train = read_rows(base / "train.jsonl") + read_rows(reviewed / "train.jsonl")
    val = read_rows(base / "val.jsonl")
    require(len({r["id"] for r in train + val}) == len(train + val), "duplicate IDs")
    old_questions = {p.normalized_question(r["question"]) for r in read_rows(base / "train.jsonl")}
    new_questions = [p.normalized_question(r["question"]) for r in rows["train"]]
    require(len(set(new_questions)) == len(new_questions) and
            not set(new_questions) & old_questions, "duplicate new questions")
    p.assert_split(dict(train=train, val=val))
    merged = out / "merged_candidate"
    merged.mkdir()
    with (merged / "train.jsonl").open("x") as stream:
        for row in train:
            stream.write(p.compact(row) + "\n")
    shutil.copy2(base / "val.jsonl", merged / "val.jsonl")
    receipt = dict(
        kind=VERSION, ready_for_training=False, candidate_data_only=True,
        assistant_spot_check_pending=True, base=str(base), new_data=str(reviewed),
        expansion_plan_sha256=p.file_sha(out / "plan.json"),
        new_audit_sha256=verify_subset(reviewed)["audit_sha256"],
        train=len(train), val=len(val), old_train=750, new_train=len(rows["train"]),
        targeted_counts=dict(counts),
        new_action_counts=dict(Counter(json.loads(r["target"])["next_action"] for r in rows["train"])),
        new_round_counts=dict(Counter(r["round"] for r in rows["train"])),
        data_sha256={s: p.file_sha(merged / f"{s}.jsonl") for s in ("train", "val")},
        evidence_count_per_example=20, truncated_documents=0,
        train_val_story_overlap=[], train_val_source_family_overlap=[],
        train_val_document_overlap=[], train_val_question_overlap=[],
        all_new_labels_second_pass_reviewed=True, not_human_gold=True,
        all_retained_labels_semantically_certified=False, production_modified=False,
        running_training_dataset_changed=False, review_required_before_training=True,
    )
    require(receipt["data_sha256"]["val"] == plan["base_data_sha256"]["val"], "fixed eval changed")
    p.atomic_json(merged / "manifest.json", receipt)
    # Reproducible, stratified read-only spot-check queue, never labeled "passed".
    selected = []
    for kind in KINDS:
        subset = [r for r in rows["train"] if r["targeted_kind"] == kind]
        random.Random(20261017).shuffle(subset)
        selected.extend(subset[:3])
    with (out / "spotcheck_requests.jsonl").open("x") as stream:
        for row in selected:
            stream.write(p.compact(row) + "\n")
    return receipt


def workflow(out):
    verify_plan(out)
    require(os.environ.get("BIGMODEL_API_KEY"), "missing credential")
    with (out / "workflow.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(pid=os.getpid(), stage="starting", started_unix=time.time(),
                      no_gpu_usage=True, running_training_dataset_changed=False,
                      production_modified=False, auto_training_enabled=False)

        def save(stage, **fields):
            status.update(stage=stage, updated_unix=time.time(), **fields)
            p.atomic_json(out / "workflow.json", status)
        try:
            save("question_generation")
            run_stage(out, "questions")
            save("prepare_annotation")
            if not (out / "annotation").exists():
                build_requests(out)
            else:
                p.load_requests(out / "annotation")
            save("annotation_and_all_label_review")
            run_stage(out, "annotation")
            save("offline_audit_and_candidate_merge")
            receipt = audit_and_merge(out)
            save("candidate_ready_awaiting_spotcheck", train_examples=receipt["train"],
                 new_examples=receipt["new_train"], completed_unix=time.time())
            return status
        except Exception as exc:
            save("blocked_or_failed", failed_stage=status["stage"], error=str(exc))
            raise


def launch(out):
    verify_plan(out)
    require(not (out / "workflow.json").exists(), "workflow exists; inspect before resuming")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "workflow", "--out", str(out)]
    with (out / "workflow.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    result = dict(pid=proc.pid, out=str(out), no_gpu_usage=True,
                  auto_training_enabled=False, running_training_dataset_changed=False,
                  credential_persisted=False)
    p.atomic_json(out / "launch.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "questions", "build-requests", "annotate",
                                           "merge", "verify-plan", "workflow", "launch"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT /
                        "targeted_pt_v2_20261014_attempt02/merged_data")
    parser.add_argument("--packs", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.command in ("questions", "annotate", "workflow", "launch"):
        if not os.environ.get("BIGMODEL_API_KEY"):
            os.environ["BIGMODEL_API_KEY"] = previous.read_credential(Path("${REMOTE_HOME}/.bigmodel_env"))
    # Annotation runs are CPU/network only, not consumers of training GPU memory.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    out = args.out.resolve()
    actions = {
        "prepare": lambda: prepare(args.base.resolve(), out, args.packs, args.smoke),
        "questions": lambda: run_stage(out, "questions", args.limit),
        "build-requests": lambda: build_requests(out),
        "annotate": lambda: run_stage(out, "annotation", args.limit),
        "merge": lambda: audit_and_merge(out), "verify-plan": lambda: verify_plan(out),
        "workflow": lambda: workflow(out), "launch": lambda: launch(out),
    }
    print(p.compact(actions[args.command]()), flush=True)


if __name__ == "__main__":
    main()
