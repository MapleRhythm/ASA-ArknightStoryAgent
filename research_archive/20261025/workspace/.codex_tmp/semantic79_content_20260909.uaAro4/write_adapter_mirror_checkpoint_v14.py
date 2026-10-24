"""Publish completed evidence; do not claim the accuracy goal is complete."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
ADAPTER = RUN / "adapter_pair_v1.p0RvMj"
SYNC = RUN / "training_mirror_sync_v1.CWHPXM"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / "docs/CANONICAL_ADAPTER_ABLATION_AND_TRAINING_MIRROR_SYNC_20260908_irPugk_v14.md"
    assert not destination.exists()
    post = load(ADAPTER / "adapter_pairs_postflight_v1.json")
    review = load(ADAPTER / "adapter_pairs_source_review_v1.json")
    applied = load(SYNC / "mirror_applied_verified_v1.json")
    deps = load(SYNC / "isolated_dependencies_receipt_v1.json")
    bundle = load(SYNC / "canonical_bundle_receipt_v1.json")
    manifest = load(SYNC / "canonical_bundle_manifest_v1.json")
    assert post["actual_adapter_calls_and_returned_echo_verified"] and post["pairs"] == 6
    assert len(review["reviewed_rows"]) == 12 and review["training_ready"] is False
    assert applied["replaced"] == 3 and applied["tests"]["tests"] == 120
    assert applied["manifest_sha256"] == sha(SYNC / "canonical_bundle_manifest_v1.json") == bundle["manifest_sha256"]
    assert sha(SYNC / "mirror_postapply_tests120_v1.xml") == applied["tests"]["sha256"]
    for name in ("canonical_mirror_tests120_v1.xml", "candidate_tests120_v1.xml", "mirror_postapply_tests120_v1.xml"):
        suites = ET.parse(SYNC / name).getroot().findall("testsuite")
        assert sum(int(s.attrib["tests"]) for s in suites) == 120
        assert all(all(int(s.attrib.get(k, 0)) == 0 for k in ("errors", "failures", "skipped")) for s in suites)
    assert len(manifest["files"]) == applied["files"] == 212
    assert all(sha(ROOT / rel) == info["sha256"] for rel, info in manifest["files"].items())
    slots = load(HERE / "answer_slot_schema_pairs14_v2.json")
    assert slots["summary"] == {
        "retry_v1": {"states": 7, "statuses": {"error": 5, "ok": 2}, "coverage": {"audit_error": 5, "complete": 2}},
        "schema_clarified_v2": {"states": 7, "statuses": {"error": 2, "ok": 5}, "coverage": {"audit_error": 2, "none": 2, "complete": 2, "partial": 1}},
    }
    text = f"""# ASA：LoRA 权重配对消融与训练镜像修复同步（v14）

记录时间（UTC）：{datetime.now(timezone.utc).isoformat()}。文件名延续 20260908 实验系列，不表示所有步骤都发生于同一当地日期。

241 为唯一规范源，248 为训练镜像，本地仅转运。本阶段没有切默认模型、训练、重启服务、改旧数据或覆盖历史标签；提高真实回答正确率的总目标仍未完成。

## 一、已完成的真实生成对照

目录：`{ADAPTER}`。

6 个选定开发问题 × LoRA 开/关，共 12 次生成。问题分别为美食节、行星碎屑、娜斯提、玛露西尔、夜刀、里昂。
每对原始 prompt、完整可见证据、tokenizer、seed、采样参数完全一致；3 对关先、3 对开先。不提供旧答案、参考答案或审核纠错。
关适配器是在同一 LoRA-enabled vLLM engine 内传 `lora_request=None`，并非另一套独立 base-only 部署。

| 条件 | 状态数 | 规范化后严格 JSON | 原始直接回答 | 校验后直接回答 | 校验后继续检索 |
|---|---:|---:|---:|---:|---:|
| LoRA 关 | 6 | 6 | 6 | 5 | 1 |
| LoRA 开 | 6 | 5 | 5 | 3 | 3 |

这些是机械校验/动作统计，不是正确率。所有生成均 stop，无输入截断或长度上限结束。
逐请求的 adapter_active、实际/返回 LoRA ID、seed、prompt SHA 均已核对；重新编码 token ID SHA 与预检一致。
base/adapter 全部已记录资产的后验哈希、运行时 Python 源码及脚本哈希不变。
共享 metadata 中的默认 seed/runtime mode 不能替代逐请求 receipt；receipt 已证明实际开关成功。

### 逐源文结论

- 美食节：关组把 E4 的反问改写为陈述并放入 quote，故拒绝；开组展示节日、准备、食物分享，核心有支持，但丰富的摊位/解谜/甜点细节在展示前被 fact 拼接改写丢掉。只评分真实展示答案。
- 行星碎屑：关组新增足够推力这一启动机制，较旧的单纯效果复述有信息增量，但没有明确连接运动与伤害，不先认定完整回答“为什么”；开组第二个 quote 的 JSON 引号坏了，经恢复后缺 refs，被拒绝。这不是长度截断。
- 娜斯提：两边都有互相帮助的核心；关组延伸到“拒绝被动接受命运”，需保留解释边界；开组把过去互相嫌弃/互相帮助的同一模式说成与理想模式“形成对比”，未获所引关系支持。原正对照在新输出上也可能退化。
- 玛露西尔：关组以“为未来作美好打算”解释“不急着赶路”，因果无来源支持；E1 实际还说莱欧斯更着急、齐尔查克缩短行程。开组继续检索，没有答案，不能计正确回答。
- 夜刀：两边仍只说“另一个危机”，未给出被问的危机名称。开组 supported_fact 去掉“极有可能”，fact 拼接展示又放大了确定性。完整 E3/E6 可见矿石病相关线索。
- 里昂：关组简洁核心有完整 E3 支持，但所选短 quote 未直接摘出最关键的固执/透支句；开组改写 quote 起始词，另将莱昂图索的暴力反思错归里昂，并把“马上要完蛋”压成已经完蛋。正确核心和错误附加应分开。

逐条独立助手复核保存在 `adapter_pairs_source_review_v1.json`，不是 human gold，不回写训练资格。
未把旧 Transformers strict9 base 实验与本组分母合并。
结论：**不能因 5/6 大于 3/6 就关闭默认 LoRA；机械放行增加同时带来了坏因果答案。**

## 二、248 可恢复同步已落地

248 目录：`${REMOTE_HOME}/ASA-ArknightStoryAgent/.canonical_sync_20260908.lxnhWO`。

这处镜像检查时没有 `.git`、`release` 或 `tests`，但有 `src/goldenglow`。只陈述已检查的路径，不声称 248 全机没有仓库。
builder、cleaner、merger 的旧字节与 241 修复前备份一致，无观察到的训练机专有修改。同步对旧哈希和新路径执行 guard，拒绝未知现存文件或符号链接。

替换 3 个脚本：

1. `scripts/build_soda_api_verifier_dataset.py`
2. `scripts/clean_soda_api_verifier_dataset.py`
3. `scripts/merge_soda_api_verifier_shards.py`

新增 209 个文件：共享 integrity helper、4 个根测试文件，以及完整 ASA Python 包的 204 个 `.py` 依赖。
只加 package 是为避免 builder 引入诊断词清理后缺间接依赖；没有搬模型、配置、索引、业务数据，也不覆盖原 `src/goldenglow`。
旧脚本可从 `{applied["backup"]}` 恢复；逐文件 journal、原/新 SHA 清单和回退说明均已保存。

| 检查位置 | CPU 合成测试 |
|---|---:|
| 241 规范源，打包前 | 120/120 |
| 248 隔离 candidate，覆盖前 | 120/120 |
| 248 主镜像，应用后 | 120/120 |

这是此前规范 276 项中的子集，不是又增加 360 个独立测试。241 规范总数仍为 276。
包 SHA256：`{bundle["sha256"]}`；manifest SHA256：`{bundle["manifest_sha256"]}`。
248 应用回执 SHA256：`{sha(SYNC / "mirror_applied_verified_v1.json")}`。

测试依赖只安装到 staging 的 `cpu_test_deps`，通过 PYTHONPATH 使用，未修改系统 Python 或现有训练环境。
采用 pytest 9.1.1/tqdm 4.67.3，所有实际下载 wheel 逐一与官方 PyPI 元数据核对后才离线安装；依赖 manifest 已保存。
**交接摘要中的这两个 wheel SHA 均有误，本轮重新核对后纠正，错误值没有被用于安装。**
pip 还报告宿主可见的 torch 元数据存在未满足依赖；没有安装/卸载 torch 或据此判断现有 GPU 任务健康。此处成功只认证隔离 CPU 测试，不认证整套训练环境。
248 A100 的已有进程未停止、未抢占；本轮未启动 GPU 训练。

## 三、审核可靠性继续修复中

原 39 个直接回答的槽检查仍保留 32 成功、7 失败；不修补旧 JSON 后冒充当时成功。
原提示同时要求每个 slot 有 answer_spans，又说 unfilled 不得有 answer_spans，存在可确认歧义。
7 个失败原文中：3 个未填槽漏字段、3 个摘句不是逐字子串、1 个有额外字段。不能将全部 7 个失败归因于该歧义。
新 v2 只将此句澄清为“字段必须保留，未填槽值为 []”，校验器和其余语义标准不变，12 个实验测试通过，不计入规范 276。
已完成预登记的 7 对、14 次请求：旧提示普通重试为 2 成功/5 失败，澄清提示为 5 成功/2 失败。
其中 3 条未填槽漏字段在 v2 都消失；剩余 2 条仍是摘句非原样。旧标签不改，原 39 状态的失败记录和每组 7 条分母均保留。
v2 的自动覆盖标签为 2 complete / 1 partial / 2 none / 2 audit_error，不是事实正确率；“军官为何执行禁令”仍需防止审核员越界要求更深层动机。
即使修复 schema 成功，槽审核也只能证明回答覆盖，不能证明人物归属、因果和时态正确，更不能直接接入生产 gate。

## 四、当前总指标口径不变

历史 fixed79 是 58 个去重问题的 79 个证据状态，不是 79 个独立端到端题。
规范重放为 39 direct / 31 retrieve / 9 abstain。旧自动内容审核为 19 complete / 11 partial / 8 unsupported / 40 no_answer / 1 audit_error，但已有确认的假阳性，不能称 19/79 为真实正确率。
本轮 adapter12 与槽审核均不是新 RAG 召回测试。暂无新的可认证全链路回答正确率；不以 evidence 可见性或 quote 匹配率替代。

后续优先项：验证语义审核的失败边界；检查 fact 拼接时的确定性变化和有用内容丢失；在固定的更广覆盖评测上比较候选，而非继续把这 6 道开发题调到高分。
"""
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"report": str(destination), "sha256": sha(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
