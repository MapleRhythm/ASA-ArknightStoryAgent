"""Write a new evidence-backed checkpoint; never overwrite an older report."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_counts(path):
    return [
        {k: int(s.attrib.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
        for s in ET.parse(path).getroot().findall("testsuite")
    ]


applied = load(HERE / "applied_verified.json")
replay = load(HERE / "stored_judgment_replay_v2.json")
kto = load(RUN / "quote80_kto_candidate_audit_v1.json")
assert applied["canonical_tests"] == {"tests": 173, "failures": 0, "errors": 0, "skipped": 0}
assert test_counts(HERE / "baseline_tests_v2.xml") == [
    {"tests": 50, "failures": 45, "errors": 0, "skipped": 0}
]
assert test_counts(HERE / "merge_baseline_tests_v1.xml") == [
    {"tests": 8, "failures": 7, "errors": 0, "skipped": 0}
]
for relative, info in applied["installed_files"].items():
    assert sha(ROOT / relative) == info["candidate_sha256"]
for info in kto["splits"].values():
    assert sha(Path(info["path"])) == info["sha256"]
example = next(
    row for row in load(Path(kto["splits"]["train"]["path"]))
    if row["id"] == "537edb855b699a30-verifier-chosen"
)
example_prompt = example["conversations"][0]["value"]
example_target = json.loads(example["conversations"][-1]["value"])
assert "output_schema: grounded_action_v1" in example_prompt
assert example_target["next_action"] == "answer_directly" and "answer" in example_target
assert "supported_facts" not in example_target and "final_answer" not in example_target
before, after = replay["replay"]["before"], replay["replay"]["candidate"]
assert after["positive_direct_contract_issues"] == {}
assert after["label_conflict"] is None
assert after["global_split"]["overlapping_question_keys"] == []
now = datetime.now(timezone.utc)
source_rows = "\n".join(
    f"| `{name}` | `{info['candidate_sha256']}` |"
    for name, info in applied["installed_files"].items()
)
text = f"""# ASA 训练目标协议与分片切分修复检查点 v7

- 时间：{now.isoformat()}（UTC）；{now.astimezone(timezone(timedelta(hours=8))).isoformat()}（北京时间）。
- 规范源：241，`{ROOT}`；248 是训练镜像，本阶段未启动训练或切换权重。
- HEAD：`{applied['head']}`，没有提交、push 或服务重启。
- 证据目录：`{HERE}`。

## 本阶段结论

当前数据构建脚本与分片合并脚本中存在可复现的标签/协议完整性问题，已在规范仓库修复。新增 58 个测试加现有 release 115 个测试，共 **173/173 通过**。旧构建脚本在 50 个同组测试中 45 失败、5 通过；旧合并脚本在 8 个测试中 7 失败、1 通过。

这是一项训练输入质量修复，**不是回答正确率提升的实测证明**。没有运行新的生成模型评测，没有重建或覆盖真实训练集，也没有训练、晋级任何 adapter。现有 79 条历史证据状态仍不能当成 79 道独立的端到端题目。

## 1. 已保存 KTO 数据的协议冲突

候选权重：`{kto['selected_adapter']}`，父权重是此前评测过但未晋级的 quote-SFT。记录的 KTO 已完成 370/370 步、2 epoch；本阶段没有对该 KTO 权重做推理评测。

| 已保存的 filtered split | 记录 | 不同问题 | 正例 | 直接回答正例 | 其中含 supported_facts/quote |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 737 | 192 | 414 | 118 | 0 |
| val | 54 | 30 | 31 | 9 | 0 |

train/val 有 27 个规范化精确问题重叠。train 正例中另有 1 条非法 JSON。负例有引用或非法内容可能是负例设计，不应混算成正例污染。

实例 `537edb855b699a30-verifier-chosen`（“斯第奇·画布为什么要离开际崖城去地上？”）的输入要求 `grounded_action_v1`，直接回答须有 supported_facts、evidence_refs、quote、final_answer；目标却使用旧式 answer、missing_slots、clarification_question 等字段。这与同一条训练输入自己的指令冲突，不只是展示字段偏好。

追溯边界：当前修改前脚本已经会寻找 supported_facts 候选，不是无条件输出旧式 answer。Git 中 2026-05-31 的 `8bf1ce7` 版本仍有旧式构建逻辑，2026-06-07 的 `9ec7d83` 已改为部分 grounded 格式。历史数据的形态与旧逻辑一致，但缺少经过认证的历史执行源码哈希，**不能据此断言这批数据当时用了哪个版本，更不能归罪于 2026-09-06 某次模型运行**。

## 2. 规范源码已经修复的部分

1. 引用、事实 ID、推断引用原样保留；不再通过裁到 80 字、压缩空白、删除多余引用或事实来伪造合规目标。超限由验证报告拒绝，不自动截断。
2. `use_for_training` 必须严格为布尔 true；不再覆盖审核器的否决。
3. 直接回答正例只从明确审核无错、无先验补写、evidence_sufficient=true 的候选中选取。必须是严格 JSON 对象，不能把代码围栏、重复键等修成正例。
4. 对 supported_facts 和 inferred_facts 都检查结构、各自对应的可见证据编号、逐字 quote 与预算。不能拿其他块或隐藏全文补引用。该检查不等于语义蕴含检查。
5. 负例保留审核器实际看见的原始响应，包括被判错的非法 JSON；不再把响应改写后沿用原标签。
6. 构建和合并共用 `soda_dataset_integrity.py`：按规范化问题整体切分；缺问题时按完整输入分组；在写出前阻止同输入同输出的正负冲突。JSON 键顺序/缩进差异不能绕过冲突检查。
7. 合并阶段先汇总所有分片再统一切分；同 ID 不同内容、缺失 ID、非法 val_ratio 都明确失败，不再静默取第一条或夹紧比例。
8. 显式保留未审核 conclusion 的选项也不能绕过直接回答正例的机械协议检查；报告标记 `training_ready=false`，不等同于已完成语义标签审批。

本阶段改动范围：2 个既有脚本、1 个新共享模块、2 个新测试文件。此前已有 dirty 内容的逐文件哈希经检查未变；源审计文件和该 KTO 的 train/val 哈希未变。248、模型指针、服务和真实训练数据未被改动。

## 3. 真实已保存审核结果的只读回放

对 3 个输入分片及保存的审核结果执行当前源码前后对照，所有训练输出仅在内存中存在。没有调用 API、GPU、检索或训练。与上面的 737/54 filtered 历史数据不是相同分母。

| 指标 | 修改前当前源码回放 | 修复源码回放 |
| --- | ---: | ---: |
| 输出记录 | {before['records']} | {after['records']} |
| 直接回答正例 | 160 | 68 |
| 正例直接回答的机械协议问题 | 有 | 0 |
| 同输入同输出正负冲突 | 检出，至少 1 组 | 0 |
| 全量切分后 train/val 重叠问题 | {len(before['global_split']['overlapping_question_keys'])} | 0 |
| 全量切分 train/val 记录 | 868 / 75 | 795 / 71 |

159 个问题横跨多个输入分片，因此只修“单个分片的切分”不够，合并必须重新全局分组。该合并路径已有合成多分片测试覆盖。

修复后 68 条直接回答正例只代表通过机械检查；仍可能含主客体颠倒、答非所问或不被引文语义支持的断言。不能把它们直接拿去训练，也不能把 160→68 当成答案正确率。其余保留记录包含 390 条未在本阶段重新审核的 hypothesis 正负样本。负例恢复原样后有 16 条不具备可解析 action，这是保留原始错误输出，不是新生成的正例错误。

## 4. 尚未解决，不能扩大声称

- 后续需要逐条语义复核保留下来的直接回答候选、建设问题级隔离的训练/验证/测试集合，再做受控训练与同条件评测；本阶段不批准训练。
- 历史样本含诊断词混入 hypothesis/entities/keywords 的污染。当前 release 已会清理 missing_slots；已污染 hypothesis 是否继续被保留，仍需专门运行探针验证，不能直接把历史污染当当前运行回归。
- 当前运行时同一事实组内部仍存在“借用其他事实引用词项”的待查风险；上一阶段只隔离了 supported/inferred 两组引用池。
- 另外发现 `clean_soda_api_verifier_dataset.py` 仍有独立的按 prompt 切分和旧格式定点答案重写。本阶段没有调用或修改它；不得在新构建后直接跑这个旧清洗路径并声称完整性仍受保障。
- 机械 quote 匹配不验证主客体/因果/动机，自动 judge 标签也不能直接视为 gold。
- 原有 79 状态、strict9、提示对照与 quote-SFT 实验的结论仍以 v4–v6 为准。本阶段没有新的端到端回答正确率，也没有证据支持切换默认 LoRA。

## 5. 复核凭据

- `quote80_kto_candidate_audit_v1.json`（父目录）：已保存 KTO 数据与谱系审计。
- `original.py`、`merge_original.py`：修改前规范源码备份。
- `baseline_tests_v2.xml`、`merge_baseline_tests_v1.xml`：旧实现真实失败。
- `candidate_combined_tests_v3.xml`：58 个候选测试通过。
- `canonical_combined_tests_v1.xml`：规范源码新增测试 + release 测试，173 个通过。
- `stored_judgment_replay_v2.json`：全量只读回放与 6 个源输入文件哈希。
- `historical_protocol_conflict_example_v1.json`：真实协议冲突样本的完整输入和目标；仅复制用于审计，没有修改原数据。
- `preapply_manifest.json`、`applied_verified.json`：应用前后源码和已有 dirty 内容校验。

| 规范文件 | 本阶段安装 SHA-256 |
| --- | --- |
{source_rows}

报告状态：持续改进中的检查点，不是项目完成声明。
"""
path = ROOT / "docs/CANONICAL_VERIFIER_CONTRACT_REPAIR_20260908_irPugk_v7.md"
assert not path.exists()
with (HERE / "historical_protocol_conflict_example_v1.json").open("x", encoding="utf-8") as output:
    json.dump({
        "id": example["id"], "source": kto["splits"]["train"]["path"],
        "source_sha256": kto["splits"]["train"]["sha256"],
        "prompt": example_prompt, "target": example_target,
    }, output, ensure_ascii=False, indent=2)
with path.open("x", encoding="utf-8") as output:
    output.write(text)
print(json.dumps({"report": str(path), "sha256": sha(path)}, ensure_ascii=False))
