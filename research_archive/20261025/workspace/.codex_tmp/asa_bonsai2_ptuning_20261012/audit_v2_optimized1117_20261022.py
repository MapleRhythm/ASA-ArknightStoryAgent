"""Integrity/count audit of separate manual judgments, NEVER semantic auto-grading."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import mean

from audit_input_final1117_20261019 import aggregate as audit_baseline

ROOT = Path(__file__).resolve().parent
REVIEW = "V2_OPTIMIZED1117_CONTENT_REVIEW_20261022.json"
CATEGORIES = ("complete", "partial", "wrong", "no_answer")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def transition(old, new):
    a, b = old == "complete", new == "complete"
    return "both_complete" if a and b else "gained_complete" if b else "lost_complete" if a else "neither_complete"


def audit(root=ROOT):
    review = read(root / REVIEW)
    snapshot = root / review["snapshot"]
    baseline_dir = root / review["baseline_run"]
    files = {
        root / review["tasks"]: "tasks_sha256",
        snapshot / "generated_native46.jsonl": "output_sha256",
        snapshot / "paired_final46_for_content_review.jsonl": "paired_sha256",
        snapshot / "run.json": "run_sha256",
        snapshot / "generation_receipt.json": "generation_receipt_sha256",
        root / review["baseline_review"]: "baseline_review_sha256",
        baseline_dir / "generated_native46.jsonl": "baseline_output_sha256",
    }
    for path, field in files.items():
        assert sha(path) == review[field], f"immutable file changed: {path}"
    baseline_audit = audit_baseline(root)
    assert baseline_audit["integrity_passed"]
    base_review = read(root / review["baseline_review"])
    run, receipt = read(snapshot / "run.json"), read(snapshot / "generation_receipt.json")
    checkpoint = read(snapshot / "latest_checkpoint.json")
    baseline_run = read(baseline_dir / "trained/run.json")
    assert read(snapshot / "workflow.json")["stage"] == "complete"
    assert run["training_complete"] and run["optimizer_steps"] == checkpoint["step"] == 1117
    assert run["train_examples"] == 1117 and run["val_examples"] == 46
    assert run["backbone_bytewise_unchanged"] and run["backbone_frozen"]
    assert run["identity"]["data_sha256"] == baseline_run["data_sha256"]
    assert run["input_baseline_run_sha256"] == sha(baseline_dir / "trained/run.json")
    assert receipt["completed"] and receipt["count"] == 46
    assert receipt["output_sha256"] == review["output_sha256"]
    assert receipt["no_input_truncation"] and not receipt["answer_validation_enabled"]
    assert review["semantic_adjudication_complete"] and review["format_and_citation_binding_excluded"]
    tasks = rows(root / review["tasks"])
    new = rows(snapshot / "generated_native46.jsonl")
    old = rows(baseline_dir / "generated_native46.jsonl")
    pairs = rows(snapshot / "paired_final46_for_content_review.jsonl")
    cases, old_cases = review["cases"], base_review["cases"]
    assert len(tasks) == len(new) == len(old) == len(pairs) == len(cases) == len(old_cases) == 46
    assert len({r["id"] for r in new}) == len({r["id"] for r in old}) == 46
    paired = {key: [] for key in ("both_complete", "gained_complete", "lost_complete", "neither_complete")}
    groups, bindings = {}, []
    for pos, (task, before, after, pair, case, old_case) in enumerate(
            zip(tasks, old, new, pairs, cases, old_cases), 1):
        assert task["position"] == pair["position"] == case["position"] == old_case["position"] == pos
        assert task["id"] == before["id"] == after["id"] == pair["id"]
        assert task["question"] == before["question"] == after["question"] == pair["question"]
        assert task["bonsai_prompt_sha256"] == text_sha(task["bonsai_prompt"])
        assert task["bonsai_prompt_sha256"] == before["prompt_sha256"] == after["prompt_sha256"] == pair["prompt_sha256"]
        assert pair["input_only_raw"] == before["raw_text"]
        assert pair["optimized_v2_raw"] == after["raw_text"]
        assert len(task["documents"]) == before["evidence_count"] == after["evidence_count"] == 20
        for output in (before, after):
            assert output["prefix_tokens"] == 16
            assert not output["truncated"] and not output["output_limit_reached"]
            try:
                parsed = json.loads(output["raw_text"])
            except ValueError:
                parsed = None
            assert parsed == output["parsed"]
            assert output["valid_json"] == (parsed is not None)
        for doc in task["documents"]:
            assert text_sha(doc["text"]) == doc["sha256"]
            assert doc["text"] in task["bonsai_prompt"]
        eids = [d["evidence_id"] for d in task["documents"]]
        checked = eids if case["source_eids"] == "all20" else case["source_eids"]
        assert checked and set(checked) <= set(eids)
        assert case["category"] in CATEGORIES and type(case["core_complete"]) is bool
        assert case["category"] != "complete" or case["core_complete"]
        assert isinstance(case["note"], str) and len(case["note"]) >= 20
        paired[transition(old_case["category"], case["category"])].append(pos)
        groups.setdefault(task["question_group"], []).append(case["category"] == "complete")
        bindings.append(dict(
            position=pos, id=task["id"], question=task["question"],
            prompt_sha256=after["prompt_sha256"],
            output_raw_sha256=text_sha(after["raw_text"]),
            baseline_raw_sha256=text_sha(before["raw_text"]),
            category=case["category"], core_complete=case["core_complete"],
            baseline_category=old_case["category"], baseline_core_complete=old_case["core_complete"],
            source_eids=checked, all20_checked_for_missing_link=case["source_eids"] == "all20",
            note=case["note"]))
    assert len(groups) == 43
    counts = Counter(c["category"] for c in cases)
    baseline_counts = Counter(c["category"] for c in old_cases)
    borderline = [c["position"] for c in cases if c.get("borderline_alternative") == "complete"]

    def verbosity(outputs):
        # Literal field occurrence counts remain inspectable even for malformed JSON.
        fields = [len(re.findall(r'"fact"\s*:', r["raw_text"])) for r in outputs]
        return dict(
            mean_raw_characters=mean(len(r["raw_text"]) for r in outputs),
            mean_generated_token_count=mean(len(r["token_ids"]) for r in outputs),
            literal_fact_field_counts=dict(Counter(fields)),
            rows_with_eight_or_more_literal_fact_fields=sum(n >= 8 for n in fields),
            descriptive_metrics_not_semantic_grading=True)

    return dict(
        integrity_passed=True, semantic_grading_method=review["review_type"],
        review_sha256=sha(root / REVIEW), output_sha256=review["output_sha256"],
        evaluated=46, question_groups=43, train_examples=1117, optimizer_steps=1117,
        optimized_v2=dict(
            categories={key: counts[key] for key in CATEGORIES},
            complete_rate=counts["complete"] / 46,
            core_complete=sum(c["core_complete"] for c in cases),
            core_coverage_rate=sum(c["core_complete"] for c in cases) / 46,
            core_coverage_is_not_whole_answer_accuracy=True,
            mean_teacher_forced_nll=mean(x["nll"] for x in run["final_validation"]),
            valid_json=sum(r["valid_json"] for r in new), verbosity=verbosity(new)),
        input_only_baseline=dict(
            categories={key: baseline_counts[key] for key in CATEGORIES},
            complete_rate=baseline_counts["complete"] / 46,
            core_complete=sum(c["core_complete"] for c in old_cases),
            core_coverage_rate=sum(c["core_complete"] for c in old_cases) / 46,
            mean_teacher_forced_nll=baseline_audit["final_teacher_forced_mean_nll"],
            valid_json=sum(r["valid_json"] for r in old), verbosity=verbosity(old),
            historical_judgments_reused_not_new_independent_review=True),
        complete_rate_change_percentage_points=(counts["complete"] - baseline_counts["complete"]) * 100 / 46,
        paired_complete_transitions=paired,
        groups43_equal_weight_complete_rate=mean(mean(v) for v in groups.values()),
        borderline_sensitivity_not_confidence_interval=dict(
            positions=borderline, complete_if_all_four_relaxed=counts["complete"] + len(borderline)),
        invalid_json_positions=[pos for pos, r in enumerate(new, 1) if not r["valid_json"]],
        content_complete_despite_invalid_json=[pos for pos, (r, c) in enumerate(zip(new, cases), 1)
                                              if not r["valid_json"] and c["category"] == "complete"],
        format_used_for_content_grading=False, no_input_truncation=True,
        output_limit_reached=0, answer_validation_enabled=False,
        same_training_data=True, optimizer_bundle_not_single_factor_ablation=True,
        layerwise_method=run["method"], trainable_parameters=run["trainable_parameters"],
        recorded_adapter_sha256=checkpoint["adapter_sha256"],
        remote_weight_rehash_performed_by_this_local_audit=False,
        nll_is_not_answer_accuracy=True, blind_test=False, end_to_end=False,
        independent_human_gold=False, production_modified=False, cases=bindings)


def report(result):
    n, b = result["optimized_v2"], result["input_only_baseline"]
    lines = [
        "# 优化版逐层 P-tuning v2：46题内容复核",
        "",
        "日期：2026-10-22。逐题对照原始证据，由助手判断内容；不是独立人工金标。JSON格式与引文绑定不扣内容分。",
        "",
        "| 同题版本（均1117条训练） | 完整正确 | 部分正确 | 错误 | 完整正确率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for title, value in (("输入层版（历史评分经完整性校验后复用）", b), ("优化版逐层v2（本次新核对）", n)):
        c = value["categories"]
        lines.append(f"| {title} | {c['complete']} | {c['partial']} | {c['wrong']} | {value['complete_rate']:.2%} |")
    lines += [
        "",
        f"整题完整正确净变化 {result['complete_rate_change_percentage_points']:+.2f} 个百分点。"
        "这不是16/46 JSON有效率的改写：本次评分读取全部可读原始回答，格式无效也可判内容正确。",
        "",
        "## 读懂关键点与整篇可靠性是两回事",
        "",
        f"核心必要点覆盖：输入层 {b['core_complete']}/46（{b['core_coverage_rate']:.2%}），"
        f"v2 {n['core_complete']}/46（{n['core_coverage_rate']:.2%}）。"
        "这个宽松指标容许答案同时有实质错误，不能称为正确率，也不能据此认定达到60%上线要求。",
        "",
        "v2补上了第1题直接动机、第31题死亡/降生生命观、第37题“菲林”认可含义，"
        "第24题不再误叫女子郑愁。但回答常继续追加其他段落，带入人物、受话人、阶段或地点错误：",
        "",
        "- 第13题：督察/朋友答对，又把陈将来接班的身份给了星熊。",
        "- 第15题：尘土和漏报女妖答对，又把对阿米娅的继承人批评给了曼弗雷德。",
        "- 第33题：远程接入答对，却混入另一处通讯塔广播安全区的任务。",
        "- 第35题：战斗态度对比答对，却把谁不愿让谁独自面对危险反过来。",
        "- 第41题：反抗机制答对，又把对博士的恨与报复给了普瑞赛斯。",
        "- 第45题：PCS不能确认答对，却追加造物主断言和错误合谋对象。",
        "",
        "纯重复本身没有扣内容分；上述扣分针对重复或追加部分中的实质事实错误。"
        "第28题“推倒”相较原文“轻推”的动作强化沿既有轻微概括容忍，没有单独因此判错。"
        "第17题末句代词可合理指戴菲恩，最终不按最不利歧义扣分，计完整正确。",
        "",
        f"v2平均原始回答 {n['verbosity']['mean_raw_characters']:.0f} 字符，"
        f"输入层 {b['verbosity']['mean_raw_characters']:.0f}；"
        f"v2有 {n['verbosity']['rows_with_eight_or_more_literal_fact_fields']} 题出现至少8个fact字段，"
        f"输入层为 {b['verbosity']['rows_with_eight_or_more_literal_fact_fields']} 题。"
        "这是过度扩写的行为证据，不证明逐层注入或某个优化器参数是单独原因。",
        "",
        "## 同题变化",
        "",
    ]
    labels = dict(gained_complete="变为完整正确", lost_complete="失去完整正确",
                  both_complete="两版均完整正确", neither_complete="两版均非完整正确")
    for key, label in labels.items():
        lines.append(f"- {label}：{result['paired_complete_transitions'][key]}。")
    sensitivity = result["borderline_sensitivity_not_confidence_interval"]
    lines += [
        "",
        f"边界题 {sensitivity['positions']} 即使全部放宽，v2也是 "
        f"{sensitivity['complete_if_all_four_relaxed']}/46"
        f"（{sensitivity['complete_if_all_four_relaxed']/46:.2%}），不是主结果，也不是置信区间。",
        "",
        "## 运行与评分边界",
        "",
        f"- JSON有效：输入层 {b['valid_json']}/46；v2 {n['valid_json']}/46，另行报告。",
        f"- v2格式无效但内容完整正确的题号：{result['content_complete_despite_invalid_json']}。",
        "- 全部20段证据保留；两版均无输入截断、无输出触顶记录，未启用回答校验。",
        f"- 最终验证NLL：输入层 {b['mean_teacher_forced_nll']:.6f}，v2 {n['mean_teacher_forced_nll']:.6f}；不是问答正确率。",
        "- 同1117条数据、同46题、同提示与训练步数；这是调整多项优化器设置的组合实验，不是单因素因果验证。",
        "- 历史开发题46道、43问题组，多次使用；非盲测、非独立金标、非含检索全链路，不作为泛化或上线结论。",
        "- 代码只核验哈希、任务/证据身份、统计与配对，不自动判断剧情语义；没有修改发布模型。",
        "",
        "## 可追溯文件",
        "",
        f"- 逐题判断：`{REVIEW}`。",
        "- 每题原始输出/提示哈希及证据：`V2_OPTIMIZED1117_CONTENT_AUDIT_20261022.json`。",
        "- 校验程序：`audit_v2_optimized1117_20261022.py`。",
        f"- 复核文件SHA256：`{result['review_sha256']}`。",
        f"- v2原始输出SHA256：`{result['output_sha256']}`。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write derived local audit/report artifacts only")
    args = parser.parse_args()
    result = audit()
    if args.write:
        for name, content in (
            ("V2_OPTIMIZED1117_CONTENT_AUDIT_20261022.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n"),
            ("V2_OPTIMIZED1117_CONTENT_REPORT_20261022.md", report(result)),
        ):
            path = ROOT / name
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(content)
            os.replace(temp, path)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
