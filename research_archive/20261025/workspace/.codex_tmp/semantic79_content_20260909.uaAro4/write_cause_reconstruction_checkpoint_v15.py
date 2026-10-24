"""Persist a bounded checkpoint without promoting an unfinished broad audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / "docs/CANONICAL_CAUSE_SCOPE_AND_ANSWER_RECONSTRUCTION_20260908_irPugk_v15.md"
    assert not destination.exists()
    scope = load(HERE / "slot_cause_scope40_postflight_v3.json")
    preflight = load(HERE / "reconstruction_coverage39_preflight_v3.json")
    assert scope["protocol_verified"] and scope["all40_requests_and_verbatim_spans_verified"]
    assert preflight["all_source_states"] == 79 and preflight["direct_states"] == 39
    assert preflight["changed_answers"] == 34 and preflight["unchanged_answers"] == 5
    assert preflight["unique_provider_inputs"] == 72
    source = HERE / "canonical_fixed79_validation_v1.json"
    assert sha(source) == preflight["source_sha256"]
    structural = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True, "training_ready": False,
        "source_sha256": sha(source), "source_states": 79,
        "direct_states": 39, "changed_parsed_to_shown": 34, "unchanged": 5, "rows": [],
        "warning": "Changed text does not imply worsened meaning; literal modal-marker loss is only a diagnostic, not semantic certainty analysis.",
    }
    for row in load(source)["results"]:
        if row["status"] != "ok" or row["grounded"]["next_action"] != "answer_directly":
            continue
        parsed, shown = row["parsed"]["answer"], row["grounded"]["answer"]
        structural["rows"].append({
            "id": row["id"], "question": row["question"], "parsed_answer": parsed,
            "shown_answer": shown, "changed": parsed != shown,
            "grounding_warnings": row["grounded"]["grounding_warnings"],
            "lost_literal_modality_markers": [m for m in ("可能", "或许", "也许", "似乎", "未必", "不一定") if m in parsed and m not in shown],
        })
    assert len(structural["rows"]) == 39
    assert sum(r["changed"] for r in structural["rows"]) == 34
    assert not any(r["lost_literal_modality_markers"] for r in structural["rows"])
    with (HERE / "answer_reconstruction_structural39_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(structural, stream, ensure_ascii=False, indent=2)
    broad_output = HERE / "reconstruction_coverage39_v3.json"
    status = (
        "所有请求已收齐，但本报告尚未认证其逐条语义结论。"
        if broad_output.exists() else
        "请求仍在进行；尚无最终对照结论，失败将保留。"
    )
    text = f"""# ASA：原因层级审核与答案展示改写诊断（v15）

记录时间（UTC）：{datetime.now(timezone.utc).isoformat()}。延续 20260908 实验系列命名。

规范源仍为 241，训练镜像为 248。本轮新增的是隔离审核实验和报告，不是线上生成/校验代码改动。没有训练、切默认模型、重启服务或覆盖旧数据标签。提高真实回答正确率的目标仍在推进。

## 一、修复审核定义，避免诱导无依据的深层原因

旧审核提示会把“执行上级的具体命令”误作没有回答下属为何行动，要求补充上级更深的动机。
原“维多利亚军官不让士兵讨论换旗”题的完整 E1–E6 已重读：E1 明确提及开斯特公爵下令不许讨论。不能默认把问题扩大成必须解释公爵的深层动机。

v3 只在 v2 的原有原因规则后添加定义：

> 原因可以是促使被问主体行动的具体上级命令、规则约束或客观条件；在已指出这类具体原因时，不再强求解释其背后另一个主体的更深动机，除非问题明确问到。仅复述被问主体的同一行为，不构成原因。

未删去泛称、套话、遗漏信息、逐字摘句等约束；严格校验器不变。新的文件是 `probe_answer_slots_v3.py`，旧 v1/v2 与既有结果没有被改写。

### 独立合成题对照

20 道非剧情合成题在请求前冻结，含具体命令、规则、条件、同一主体的循环复述、明确追问深层目的、身份缺失以及两个多问部分回答。
每题 v2/v3 各一次，共 40 次；10 对 v2 先、10 对 v3 先，不重试。
实际发送只有问题和候选答案，没有预期标签、判分解释、剧情证据或先前审核意见。

| 条件 | 严格协议成功 | 与预登记合成覆盖标准一致 |
|---|---:|---:|
| v2 | 20/20 | 19/20 |
| v3 | 20/20 | 20/20 |

唯一差异为 `scope08`：“快递员为什么暂停派送？”→“调度员通知快递员暂停派送。”
v2 判 none、认为只是复述；v3 接受具体指令作为下属行动的直接原因。
反向控制“调度员为什么发出通知？”若答案只重复通知，两个版本都判 none；两个多问控制也均保持 partial，没有把所有命令一律算作完整。

40 条结构化判断已读完；请求哈希、真实返回模型、引用子串、顺序和冻结文件均验证。
这是**助手预编写的合成覆盖契约**，不是独立人工剧情 gold；只差 1 题，不能据此宣称审核器泛化准确率为 100%，更不能把它作为线上拦截规则。

工件：`slot_cause_scope_synthetic20_v1.json`、`slot_cause_scope40_v3.json`、`slot_cause_scope40_postflight_v3.json`。
合成结果 SHA256：`{scope["report_sha256"]}`。

## 二、展示改写并不是少量个例

在冻结的 fixed79 规范重放中，39 个校验后直接回答有 **34 个**与保存的 parsed/pre-validation answer 不同，5 个相同。
这里比较的是解析后的答案字段，不把可能经过 JSON 恢复的字段冒充未经处理的原始生成字符串。

这一统计不说明 34 个都变差：改写可能去掉无证据附加，也可能丢掉具体细节。不能直接关闭保护来增加回答量。
对原 39 条扫描“可能/或许/也许/似乎/未必/不一定”的字面消失得到 0；这只是窄范围诊断。
v14 新 adapter12 夜刀输出观察到的确定性变化属于另一批生成，不能移植成旧 fixed79 的统计。

完整结构比较保存为 `answer_reconstruction_structural39_v1.json`，不修改旧答案。

### 更广的盲对照已冻结并执行

范围包含全部 39 个 direct 状态，比较各自 parsed 与实际 shown；34 个变化对与5个不变对全部保留。
按完全相同的问题+答案合并请求，得到 **72 个唯一审核输入**。不变对共享同一结果，以免对同一个输入重复采样制造差异。
仅发送问题和当前候选答案，不给审核员提示这是校验前还是展示后，也不提供事实证据、旧标签、grounding 警告或想要的结论。
最多2路并发，一次请求，所有失败保留。

当前状态：{status}

工件：`reconstruction_coverage39_preflight_v3.json`、`reconstruction_coverage39_v3.progress.jsonl`，最终输出为 `reconstruction_coverage39_v3.json`。

该对照只能定位“有没有回答被问信息”的变化，不能证明新答案事实正确。
对原 79 中其余40个非直接回答，本子分析不作判断，也不从全基准分母删除。
如发现覆盖下降，仍须逐源文分辨：删掉了错误扩展，还是删掉了正确且必要的信息，随后才能设计安全改动。

## 三、保持现有保护与指标边界

241 规范回归总数仍为 276；本轮额外4个原因层级测试和5个盲对照装配测试属于实验，不混入该总数。
248 镜像应用后的120项子集已通过，三个入口的隔离 CPU `--help`/import 检查也成功；不等于整套训练环境已认证。
默认模型、quote 长度规则、训练权重与原始标签未改变。当前仍无新增可认证的端到端正确率或 RAG 召回率。
"""
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"report": str(destination), "sha256": sha(destination), "broad_audit_status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
