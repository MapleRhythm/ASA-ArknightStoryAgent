"""Read-only integrity audit plus explicitly scoped assistant content judgments.

No API calls, worker imports with side effects, source repairs, merges or training.
Semantic judgments are supplied separately, never inferred by this audit.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

import annotation_protocol as p
import expand_input_failures_20261022 as e

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "input_failure_review_snapshot_20261022"
DECISIONS = ROOT / "INPUT_FAILURE_DATA_REVIEW_20261022.json"
POSITIONS = (1, 2, 3, 22, 43, 44, 45, 68, 111, 112, 113, 114,
             129, 142, 158, 164, 172, 173, 174, 178)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ledger_jobs(path):
    # Never instantiate Ledger for a read: its constructor recovers inflight calls.
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        return {key: (fingerprint, state, json.loads(value)) for key, fingerprint, state, value in
                db.execute("SELECT task_id,fingerprint,state,result FROM jobs")}


def audit(snapshot=SNAPSHOT, decisions_path=DECISIONS):
    plan = read(snapshot / "plan.json")
    manifest = read(snapshot / "annotation/manifest.json")
    requests_path = snapshot / "annotation/requests.jsonl"
    requests, exported = rows(requests_path), rows(snapshot / "annotation/train.jsonl")
    seeds = {row["id"]: row for row in rows(snapshot / "questions/requests.jsonl")}
    qjobs = ledger_jobs(snapshot / "questions/ledger.sqlite3")
    jobs = ledger_jobs(snapshot / "annotation/ledger.sqlite3")
    receipts = rows(snapshot / "annotation/api_receipts.jsonl")
    decisions = read(decisions_path)
    assert digest(requests_path) == manifest["requests_sha256"] == decisions["requests_sha256"]
    assert digest(snapshot / "annotation/train.jsonl") == manifest["data_sha256"]["train"] == decisions["export_sha256"]
    assert digest(snapshot / "questions/requests.jsonl") == plan["qrequests_sha256"]
    assert len(requests) == len(jobs) == 180
    assert len(seeds) == len(qjobs) == 60
    assert len({row["id"] for row in requests}) == len(requests)
    assert len({p.normalized_question(row["question"]) for row in requests}) == len(requests)
    assert all(job[1] == "accepted" for job in qjobs.values())
    for name, expected in plan["code_sha256"].items():
        assert digest(ROOT / name) == expected, f"sealed local code changed: {name}"
    val = rows(snapshot / "base_val.jsonl")
    val_sha = next(value for name, value in plan["inputs_sha256"].items()
                   if name.endswith("/val.jsonl"))
    assert digest(snapshot / "base_val.jsonl") == val_sha
    p.assert_split(dict(train=requests, val=val))
    for path in (snapshot / "workflow.json", snapshot / "plan.json",
                 snapshot / "annotation/manifest.json"):
        data = read(path)
        assert data["ready_for_training"] is False
        assert data["auto_training_enabled"] is False
        assert data["production_modified"] is False
    assert manifest["assistant_semantic_review_pending"] is True
    assert len(receipts) == read(snapshot / "annotation/status.json")["api_calls"] == 370
    by_receipt = {r["call_id"]: r for r in receipts}
    by_export = {r["id"]: r for r in exported}
    assert len(by_export) == len(exported) == 171
    statuses, actions, kind_actions = Counter(), Counter(), defaultdict(Counter)
    unresolved, fact_counts = [], Counter()
    for position, row in enumerate(requests, 1):
        p.check_row(row)
        assert row["split"] == "train"
        assert row["prompt_tokens"] + 8192 + 16 <= 32768
        assert set(row["source_families"]) == {p.family(d["document_id"]) for d in row["documents"]}
        assert not set(row["source_families"]) & set(plan["protected_families"])
        seed = seeds[row["source_row_id"]]
        assert row["documents"] == seed["documents"]
        generation = qjobs[seed["id"]][2]["generation"]
        question_index = next(i for i, q in enumerate(generation["parsed"]["questions"])
                              if q["question"].strip() == row["question"])
        assert generation["parsed"]["questions"][question_index]["kind"] == row["targeted_kind"]
        assert row["round"] == e.question_round(int(seed["id"].rsplit("-", 1)[1]), question_index)
        assert generation["response_id"] == row["question_generation_response_id"]
        base = p.sha(p.compact(dict(
            prompt=row["teacher_prompt_sha256"],
            settings=e.settings("annotation").cache_settings(), review_fraction=1, review_seed=20261022)))
        expected_fp = p.sha(p.compact(dict(
            base=base, version=e.VERSION, script=digest(ROOT / "expand_input_failures_20261022.py"))))
        fingerprint, state, result = jobs[row["id"]]
        assert fingerprint == expected_fp
        statuses[state] += 1
        if state == "pending":
            assert row["id"] not in by_export
            unresolved.append(dict(position=position, id=row["id"],
                                   error=result["error"], retry_not_run=True))
            continue
        label, review = result["label"], result["review"]
        p.check_target(label["parsed"], row)
        p.check_review(review["parsed"])
        for phase, receipt in (("annotate", label), ("review", review)):
            assert by_receipt[receipt["call_id"]] == receipt
            assert receipt["accepted"] is True and receipt["finish_reason"] == "stop"
            assert receipt["response_id"] and receipt["returned_model"] == "glm-5.3"
            assert receipt["endpoint"] == p.CODING_PLAN_ENDPOINT
            assert receipt["teacher_prompt_sha256"] == row["teacher_prompt_sha256"]
            assert receipt["task_id"] == row["id"] and receipt["phase"] == phase
            assert json.loads(receipt["content"]) == receipt["parsed"]
            messages = (e.annotation_messages(row) if phase == "annotate" else
                        e.review_messages(row, label["parsed"]))
            assert receipt["cache_key"] == p.sha(p.compact(dict(
                phase=phase, messages=messages,
                settings=e.settings("annotation").cache_settings())))
        verdict = review["parsed"]
        passed = verdict["label_supported"] and verdict["decision_appropriate"] and not verdict["issues"]
        assert (state == "accepted") == bool(passed)
        if state == "accepted":
            output = by_export[row["id"]]
            for key in ("question", "documents", "prompt", "prompt_tokens", "round", "targeted_kind"):
                assert output[key] == row[key]
            assert json.loads(output["target"]) == label["parsed"]
            assert output["semantic_review"] == verdict
            assert output["annotation_response_id"] == label["response_id"]
            action = label["parsed"]["next_action"]
            actions[action] += 1
            kind_actions[row["targeted_kind"]][action] += 1
            if action == "answer_directly":
                fact_counts[len(label["parsed"]["supported_facts"])] += 1
        else:
            assert row["id"] not in by_export
    assert dict(statuses) == dict(read(snapshot / "annotation/status.json")["job_states"])
    assert len({c["position"] for c in decisions["cases"]}) == len(decisions["cases"])
    assert tuple(sorted(c["position"] for c in decisions["cases"])) == POSITIONS
    reviewed_counts, sampled_states, type_counts = Counter(), Counter(), Counter()
    holds, sampled_ids = [], set()
    for case in decisions["cases"]:
        row = requests[case["position"] - 1]
        assert case["id"] == row["id"]
        assert case["targeted_kind"] == row["targeted_kind"]
        assert case["pipeline_state"] == jobs[row["id"]][1]
        assert case["read_scope"] in ("all20_complete_passages", "relevant_complete_passages")
        read_eids = {d["evidence_id"] for d in row["documents"]} if case["read_eids"] == "all20" else set(case["read_eids"])
        assert set(case["source_eids"]) <= read_eids <= {f"E{i}" for i in range(1, 21)}
        if case["read_scope"] == "all20_complete_passages":
            assert len(read_eids) == 20
        assert case["content_judgment"] in ("pass", "fail", "needs_review")
        assert bool(case["error_tags"]) == (case["content_judgment"] == "fail")
        assert case["note"] and case["subquestions_checked"]
        assert not case["content_judgment"] == "fail" or case["repair_direction"]
        if case["content_judgment"] == "fail":
            holds.append(dict(id=row["id"], position=case["position"],
                              original_state=case["pipeline_state"], reasons=case["error_tags"]))
        reviewed_counts[case["content_judgment"]] += 1
        sampled_states[f"{case['pipeline_state']}:{case['content_judgment']}"] += 1
        type_counts[case["type_fit"]] += 1
        sampled_ids.add(row["id"])
    return dict(
        integrity_passed=True,
        structural_rows_checked=len(requests),
        glm_pipeline_states=dict(statuses), exported_actions=dict(actions),
        exported_kind_actions={k: dict(v) for k, v in kind_actions.items()},
        exported_fact_counts=dict(fact_counts),
        exported_over_three_facts=sum(n for k, n in fact_counts.items() if k > 3),
        no_sample_truncation=True, unique_questions=True,
        protected_split_verified_against_actual_val=True, base_val_rows=len(val),
        prompt_token_counts_recomputed=False, recorded_context_limits_pass=True,
        api_receipt_and_cache_key_binding_verified=True,
        manual_reviewed=len(decisions["cases"]), manual_judgments=dict(reviewed_counts),
        sample_state_judgments=dict(sampled_states), type_fit=dict(type_counts),
        all180_semantically_reviewed=False,
        exported_not_semantically_reviewed=len(by_export.keys() - sampled_ids),
        sample_method=decisions["sample_method"], random_sample=False,
        sample_pass_rate_is_not_population_accuracy=True, independent_human_gold=False,
        proposed_holds=holds, unresolved=unresolved,
        source_sha256={name: digest(snapshot / name) for name in (
            "plan.json", "questions/requests.jsonl", "questions/api_receipts.jsonl",
            "annotation/requests.jsonl", "annotation/train.jsonl",
            "annotation/api_receipts.jsonl", "annotation/ledger.sqlite3", "base_val.jsonl")},
        decisions_sha256=digest(decisions_path),
        source_dataset_modified=False, remote_dataset_modified=False, new_api_calls=0,
        merged=False, training_started=False, ready_for_training=False,
        note="Only this separate review/hold proposal is new; original export still has 171 rows. "
             "Integrity checks do not grade semantics; unreviewed labels are not approved.")


def report(result):
    cases = read(DECISIONS)["cases"]
    lines = [
        "# 新增输入层补数：独立于GLM自复核的助手审核",
        "",
        "日期：2026-10-22（上海时间）。只读检查远端快照，无新API调用、无改标签、无合并、无训练。",
        "",
        "## 全批次完整性",
        "",
        "- 180道题、每题完整20段证据；171条GLM通过、5条隔离、4条输出未完成。",
        "- 全180条检查证据哈希、原生提示、证据组/问题绑定、分组与实际验证集隔离；"
        "有标签的176条另检查原始API回执、缓存键、引用ID与协议；171条导出与原标签逐字段一致。",
        "- 编码与格式通过不等于内容正确。上下文长度检查复用已记录token数，未重新分词。",
        f"- 171条导出动作分布：{result['exported_actions']}。",
        f"- 其中{result['exported_over_three_facts']}条直接回答超过3个事实，"
        "与“通常1–3条”的简洁目标存在偏离，但不据此自动判错。",
        "",
        "## 本轮原文语义复核",
        "",
        "- 审核20条：15条GLM通过样本，加全部5条GLM隔离样本。",
        "- GLM通过样本中13条内容可接受、2条需返修；全部5条既有隔离项的问题均得到原文支持。",
        "- 另有2条内容可接受但设计类型不成立：第45题完整证据可解释玻璃破裂；"
        "第114题可由上下文识别插话者，未形成预期的一半未知。",
        "- 这是按风险、类型和较短完整证据组选择的样本，不是随机样本；"
        "不能将13/15推成整批标注正确率。原始171条仍未改动，另有156条导出标签尚未经本轮语义复核。",
        "- 缺证结论需读全部20段；本轮对组0000、0014、0037、0042、0057读完20段。"
        "其余正向指认或已足以证伪的标签，只检查列出的完整相关段落，范围逐条列明。",
        "",
        "## GLM通过但本轮拦下的两条",
        "",
        "- 第3题：E15只说斯卡蒂解决了其他争夺者、否则东西会落入其手；"
        "没有说明她拿到钥匙的具体过程。“钥匙由此到她手上”把阻止他人获得补成了取得方式。"
        "应保留E7中帮格拉尼处理赏金猎人的理由，并把拿钥匙过程列为未知。",
        "- 第158题：E2中A称B看错、交给他白贝壳，不能直接当成旁白证实B先前误认；"
        "B看到A手中有红色也不能证明B辨色不可靠。"
        "当前标签把角色话语和一种解释升级为客观定论，应降低断言强度；"
        "本轮也没有反向武断认定一定发生换贝壳。",
        "",
        "## 逐条记录",
        "",
        "| 序号 | GLM状态 | 本轮内容 | 类型适配 | 原文意见 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        lines.append(f"| {case['position']} | {case['pipeline_state']} | {case['content_judgment']} | "
                     f"{case['type_fit']} | {case['note']} |")
    lines.extend([
        "",
        "## 后续建议（本轮未实施）",
        "",
        "1. 第3、158题送返修；5条既有隔离不直接放回。将45、114重归类或重新出真正缺证的问题。",
        "2. 4条pending保持待处理，不伪造补全；其中可能存在已成功标注但复核未完成的缓存。",
        "3. 继续审核剩余156条，优先查部分可答中的假未知、过强因果和扩写主体错配，"
        "以及缺证类是否真的包含未能回答的核心子问。",
        "4. 仍保持ready_for_training=false。审查意见只是独立提案；"
        "没有从171条导出中实际删除样本，也没有自动把余下169条视为合格。",
        "",
        "机器核验：`python3 review_input_failure_data_20261022.py`；"
        "细项：`INPUT_FAILURE_DATA_REVIEW_20261022.json`；"
        "汇总：`INPUT_FAILURE_DATA_AUDIT_20261022.json`。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.write:
        p.atomic_json(ROOT / "INPUT_FAILURE_DATA_AUDIT_20261022.json", result)
        (ROOT / "INPUT_FAILURE_DATA_REPORT_20261022.md").write_text(report(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
