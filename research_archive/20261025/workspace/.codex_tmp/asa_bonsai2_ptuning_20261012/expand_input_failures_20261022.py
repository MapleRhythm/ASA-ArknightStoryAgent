"""Isolated GLM Coding Plan expansion from generic input-prompt failure families.

No evaluation text/answers sent to the teacher, no promotion, no training.
Reuse the existing transport, ledger, protocol and full20 evidence seed selection.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import annotation_protocol as p
import expand_reader_errors_v3 as old

ROOT = Path(__file__).resolve().parent
VERSION = "input_failure_types_20261022_v1"
FILES = ("expand_input_failures_20261022.py", "expand_reader_errors_v3.py",
         "annotation_protocol.py", "targeted_pt_v2.py", "audit_fresh_subset.py")
GUIDE = """
本次仅针对通用错误类型出题，不使用任何评测题、评测答案、人物清单或评分笔记。
speaker_direction：角色/受话人/行为人和受事者绑定，尤其人名、称号、器物之别，
引语属于谁、给谁，别把对甲说的话归给乙。问题要有可判断的核心关系。
event_state_scope：同人物的不同事件阶段、计划/已经发生、演示/真实经过、
不同阵营的行动、做过/没有做过；不可把阶段混成一个事实。
minimal_complete：问具体动机或话语含义，要求解释为什么，而非抄附近机制和背景。
必要点答全即可，不诱导输出无关背景。
partial_support：两个相关子问，一部分有据，另一部分没有确切关系；
不要把一个未知子问升级成全部未知，更不能只复制引文不保留已知答案。
evidence_boundary：实体存在但所问具体因果/身份/场景缺失；附近应有真实的
相似事件而不是随机噪声。不能凭实体同名或地点相近就补出关系。
问题不要泄露正确动作，不在问题里使用“证据不足”“能否确认”等提示语。
"""
LABEL_EXTRA = """
本次重点复核：每条事实的说话者/行为人/受事人、人物与称号和器物，
不同事件和先后阶段，不得错配。问动机就答直接动机，问话语含义就解释含义，
不能用一段相关背景代替所问。依据允许作有约束推理，不强迫逐字直述。
对复合问题逐项检查：只要当前证据能答一部分，优先answer_directly，
supported_facts保留可答部分并明确剩余哪一问未知；禁止添加非法混合动作字段。
不能仅因另一子问缺证就把全部转为retrieve_more或abstain。
仅在没有任何可答核心部分、且仍有轮次时retrieve_more，末轮才abstain。
对缺证的判断必须读全部20段，不据世界知识补剧情。简短概括，通常1至3条；
不要长篇复制引文、输出推理链或凑背景。检索假设只提有依据的线索和中性问题。
"""


def schedule(index):
    return ["speaker_direction",
            "minimal_complete" if index % 2 == 0 else "event_state_scope",
            "partial_support" if (index // 2) % 2 == 0 else "evidence_boundary"]


def question_round(seed_index, question_index):
    return "1/2" if (3 * seed_index + question_index) % 2 == 0 else "2/2"


def settings(stage):
    return p.Settings(
        endpoint=p.CODING_PLAN_ENDPOINT, model="glm-5.3",
        max_output_tokens=8192, workers=2, attempts=2, timeout=240,
        review_fraction=1, seed=20261022, request_interval=1,
        max_api_calls=140 if stage == "questions" else 460,
        max_total_tokens=1_200_000 if stage == "questions" else 4_800_000,
    )


def generation_messages(row):
    return [
        {"role": "system", "content": old.GEN_SYSTEM},
        {"role": "user", "content": "完整证据：\n" +
         "\n".join(f"[{d['evidence_id']}]\n{d['text']}" for d in row["documents"]) +
         "\n按顺序设计3道新题，类型为：" + p.compact(row["requested_kinds"]) +
         GUIDE + "\n避免以下已有训练问题：" + p.compact(row["avoid_questions"]) +
         '\n只输出{"questions":[{"question":"自然问题","kind":"指定类型"},...]}。'},
    ]


def annotation_messages(row):
    messages = old.annotation_messages(row["teacher_messages"])
    messages[0]["content"] += LABEL_EXTRA
    return messages


def review_messages(row, target):
    messages = old.review_messages(row, target)
    messages[-1]["content"] += LABEL_EXTRA
    return messages


class Runner(p.Runner):
    def fingerprint(self, row):
        return p.sha(p.compact(dict(base=super().fingerprint(row), version=VERSION,
                                   script=p.file_sha(__file__))))

    def process(self, row):
        p.check_row(row)
        fingerprint = self.fingerprint(row)
        prior = self.ledger.job(row["id"])
        assert not prior or prior[0] == fingerprint, "task changed"
        if prior and prior[1] in ("accepted", "quarantined"):
            return prior[1]
        try:
            if "requested_kinds" in row:
                receipt = self.call(
                    row, "question_generation", generation_messages(row),
                    lambda value: old.check_questions(value, row["requested_kinds"]))
                result, passed = dict(generation=receipt), True
            else:
                label = self.call(row, "annotate", annotation_messages(row),
                                  lambda value: p.check_target(value, row))
                review = self.call(row, "review", review_messages(row, label["parsed"]),
                                   p.check_review)
                verdict = review["parsed"]
                passed = verdict["label_supported"] and verdict["decision_appropriate"] and not verdict["issues"]
                result = dict(label=label, review=review, semantic_review_sampled=True,
                              semantic_review_is_independent_human_gold=False)
            state = "accepted" if passed else "quarantined"
            self.ledger.save_job(row["id"], fingerprint, state, result)
            return state
        except p.BudgetReached:
            self.ledger.save_job(row["id"], fingerprint, "pending",
                                 dict(reason=self.reason or "budget_limit"))
            self.halt(self.reason or "budget_limit")
            return "pending"
        except p.APIError as error:
            # Truncation/network errors are not semantic failures of labels.
            self.ledger.save_job(row["id"], fingerprint, "pending", dict(error=error.record()))
            return "failed"


def restore_messages(row):
    row = dict(row)
    row["teacher_messages"], _ = p.prompt_for(row["question"], row["round"], row["documents"])
    p.check_row(row)
    return row


def prepare(base, out, packs, avoid_sources):
    assert 0 < packs <= 60, "bounded batch: at most 60 packs"
    assert not out.exists(), "refuse overwrite"
    train = [restore_messages(row) for row in p.json_rows(base / "train.jsonl")]
    val = [restore_messages(row) for row in p.json_rows(base / "val.jsonl")]
    p.assert_split(dict(train=train, val=val))
    seeds, protected, available = old.make_seeds(train, val, packs)
    avoid_rows = train[:]
    for source in avoid_sources:
        avoid_rows.extend(p.json_rows(source))
    by_docs = {}
    for row in avoid_rows:
        key = tuple(sorted(d["document_id"] for d in row["documents"]))
        by_docs.setdefault(key, set()).add(row["question"])
    for index, row in enumerate(seeds):
        row["id"] = f"inputfail22-{index:04d}"
        row["requested_kinds"] = schedule(index)
        key = tuple(sorted(d["document_id"] for d in row["documents"]))
        row["avoid_questions"] = sorted(by_docs[key])
    out.mkdir()
    (out / "questions").mkdir()
    with (out / "questions/requests.jsonl").open("x") as stream:
        stream.writelines(p.compact(row) + "\n" for row in seeds)
    plan = dict(
        version=VERSION, base=str(base), packs=packs, available_packs=available,
        planned_new_questions=packs * 3,
        requested_kind_counts=dict(Counter(kind for row in seeds for kind in row["requested_kinds"])),
        protected_families=protected,
        inputs_sha256={str(path): p.file_sha(path) for path in
                       [base / "train.jsonl", base / "val.jsonl", *avoid_sources]},
        code_sha256={name: p.file_sha(ROOT / name) for name in FILES},
        qrequests_sha256=p.file_sha(out / "questions/requests.jsonl"),
        settings={stage: vars(settings(stage)) for stage in ("questions", "annotation")},
        diagnosis="generic error types from final1117 input-only development review",
        fixed_eval_text_sent_to_teacher=False, old_labels_sent_to_teacher=False,
        questions_not_new_story_corpus=True, independent_human_gold=False,
        ready_for_training=False, auto_training_enabled=False, production_modified=False,
        running_training_dataset_changed=False, no_gpu_usage=True,
        maximum_combined_api_calls=600, maximum_combined_accounted_tokens=6_000_000,
    )
    p.atomic_json(out / "plan.json", plan)
    return plan


def verify(out):
    plan = json.loads((out / "plan.json").read_text())
    assert plan["version"] == VERSION
    assert plan["qrequests_sha256"] == p.file_sha(out / "questions/requests.jsonl")
    for name, digest in plan["code_sha256"].items():
        assert p.file_sha(ROOT / name) == digest, name
    for name, digest in plan["inputs_sha256"].items():
        assert p.file_sha(name) == digest, name
    for stage in ("questions", "annotation"):
        assert plan["settings"][stage] == vars(settings(stage))
    return plan


def write_question_receipts(runner):
    temporary = runner.out / "api_receipts.jsonl.tmp"
    temporary.write_text("".join(p.compact(r) + "\n" for r in runner.ledger.receipts()))
    os.replace(temporary, runner.out / "api_receipts.jsonl")


def stage(out, name, limit=0):
    verify(out)
    folder = out / name
    rows = list(p.json_rows(folder / "requests.jsonl")) if name == "questions" else p.load_requests(folder)
    with p.exclusive_run(folder):
        runner = Runner(folder, rows, settings(name))
        try:
            status = runner.run(limit)
            return status
        finally:
            if name == "questions":
                write_question_receipts(runner)
            else:
                runner.export()
            runner.ledger.close()


def build_requests(out):
    from tokenizers import Tokenizer
    plan = verify(out)
    assert not (out / "annotation").exists(), "requests already prepared"
    base = Path(plan["base"])
    # Validation is read only for exact split protection/deduplication, not teacher messages.
    existing = [row["question"] for name in plan["inputs_sha256"] for row in p.json_rows(name)]
    used = {p.normalized_question(q) for q in existing}
    tokenizer = Tokenizer.from_file(str(ROOT / "official/tokenizer.json"))
    tasks, rejected = [], Counter()
    ledger = p.Ledger(out / "questions/ledger.sqlite3")
    try:
        for seed in p.json_rows(out / "questions/requests.jsonl"):
            job = ledger.job(seed["id"])
            if not job or job[1] != "accepted":
                continue
            receipt = job[2]["generation"]
            expected = p.sha(p.compact(dict(
                phase="question_generation", messages=generation_messages(seed),
                settings=settings("questions").cache_settings())))
            assert receipt["cache_key"] == expected
            for index, question in enumerate(receipt["parsed"]["questions"]):
                q = question["question"].strip()
                normalized = p.normalized_question(q)
                if normalized in used:
                    rejected["duplicate_question"] += 1
                    continue
                round_value = question_round(int(seed["id"].rsplit("-", 1)[1]), index)
                messages, prompt = p.prompt_for(q, round_value, seed["documents"])
                tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
                if tokens + 8192 + 16 > 32768:
                    rejected["whole_sample_over_context"] += 1
                    continue
                row = dict(
                    id=p.sha(p.compact(dict(version=VERSION, question=q,
                                           source=seed["id"], round=round_value)))[:24],
                    question=q, round=round_value, source_round=round_value, split="train",
                    documents=seed["documents"], source_families=seed["source_families"],
                    source_row_id=seed["id"], evidence_count=20, truncated=False,
                    teacher_messages=messages, prompt=prompt, prompt_tokens=tokens,
                    teacher_prompt_sha256=p.sha(p.compact(messages)), old_labels_used=False,
                    targeted_kind=question["kind"], question_generation_response_id=receipt["response_id"],
                )
                assert not set(row["source_families"]) & set(plan["protected_families"])
                p.check_row(row)
                tasks.append(row)
                used.add(normalized)
    finally:
        ledger.close()
    assert tasks, "no accepted generated questions"
    p.assert_split(dict(train=tasks, val=[restore_messages(row) for row in p.json_rows(base / "val.jsonl")]))
    directory = out / "annotation"
    directory.mkdir()
    with (directory / "requests.jsonl").open("x") as stream:
        stream.writelines(p.compact(row) + "\n" for row in tasks)
    manifest = dict(
        version=p.VERSION, targeted_version=VERSION, planned_train=len(tasks), planned_val=0,
        requests_sha256=p.file_sha(directory / "requests.jsonl"),
        targeted_counts=dict(Counter(row["targeted_kind"] for row in tasks)),
        rejected=dict(rejected), evidence_count_per_example=20, truncated_documents=0,
        ready_for_training=False, assistant_semantic_review_pending=True,
        auto_training_enabled=False, not_human_gold=True,
        production_modified=False, running_training_dataset_changed=False,
        old_labels_used=False, train=0, val=0, complete=False,
    )
    p.atomic_json(directory / "manifest.json", manifest)
    return manifest


def workflow(out):
    verify(out)
    with (out / "workflow.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(pid=os.getpid(), started_unix=time.time(), no_gpu_usage=True,
                      ready_for_training=False, auto_training_enabled=False,
                      running_training_dataset_changed=False, production_modified=False)
        def save(name, **fields):
            status.update(stage=name, updated_unix=time.time(), **fields)
            p.atomic_json(out / "workflow.json", status)
        try:
            # A resume reuses completed receipts, never regenerates accepted questions.
            save("question_preflight")
            first = stage(out, "questions", limit=1)
            if not first["accepted"]:
                raise RuntimeError("question preflight did not yield a valid result")
            save("question_generation")
            for _ in range(2):
                result = stage(out, "questions")
                if not result["remaining_tasks"] or result["stop_reason"]:
                    break
            if result["remaining_tasks"]:
                raise RuntimeError("question generation pending; inspect receipts before resume")
            if not (out / "annotation").exists():
                save("prepare_annotation")
                build_requests(out)
            save("annotation_preflight")
            first = stage(out, "annotation", limit=1)
            if not first["accepted"] and not first["quarantined"]:
                raise RuntimeError("annotation preflight did not yield a valid review")
            save("annotation_and_all_label_review")
            for _ in range(2):
                result = stage(out, "annotation")
                if not result["remaining_tasks"] or result["stop_reason"]:
                    break
            save("api_finished_awaiting_assistant_review" if not result["remaining_tasks"] else "pending_inspection",
                 accepted=result["accepted"], quarantined=result["quarantined"],
                 remaining=result["remaining_tasks"])
        except Exception as exc:
            save("pending_inspection", error=str(exc))
            raise
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "launch", "workflow", "status"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--packs", type=int, default=60)
    parser.add_argument("--avoid-source", type=Path, action="append", default=[])
    args = parser.parse_args()
    out = args.out.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "prepare":
        result = prepare(args.base.resolve(), out, args.packs,
                         [path.resolve() for path in args.avoid_source])
    elif args.command == "verify":
        result = verify(out)
    elif args.command == "status":
        result = {name: json.loads((out / name).read_text()) for name in
                  ("workflow.json", "questions/status.json", "annotation/status.json")
                  if (out / name).exists()}
    else:
        if not os.environ.get("BIGMODEL_API_KEY"):
            os.environ["BIGMODEL_API_KEY"] = old.previous.read_credential(Path("${REMOTE_HOME}/.bigmodel_env"))
        if args.command == "workflow":
            result = workflow(out)
        else:
            verify(out)
            assert not (out / "launch.json").exists(), "already launched; inspect before resuming"
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "workflow", "--out", str(out)]
            with (out / "workflow.log").open("x") as log:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
            result = dict(pid=proc.pid, out=str(out), no_gpu_usage=True,
                          ready_for_training=False, credential_persisted=False)
            p.atomic_json(out / "launch.json", result)
    print(p.compact(result), flush=True)


if __name__ == "__main__":
    main()
