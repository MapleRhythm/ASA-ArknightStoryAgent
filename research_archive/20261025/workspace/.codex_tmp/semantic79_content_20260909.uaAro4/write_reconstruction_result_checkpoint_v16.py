"""Close the completed subexperiment, not the overall accuracy-improvement goal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SOURCE = ROOT / "release/ASA-ArknightStoryAgent/src"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    destination = ROOT / "docs/CANONICAL_RECONSTRUCTION_COVERAGE_RESULTS_20260908_irPugk_v16.md"
    assert not destination.exists()
    result = load(HERE / "reconstruction_coverage39_postflight_v3.json")
    assert result["protocol_verified"] and result["all39_answer_pairs_read"]
    preflight = load(RUN / "adapter_pair_v1.p0RvMj/adapter_pairs_preflight_v1.json")
    assert {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")} == preflight["source_hashes"]
    matrix = load(RUN / "adapter_pair_v1.p0RvMj/adapter_pairs12_v1.matrix.json")
    assert sha(ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json") == matrix["canonical_config_sha256"]
    regression = RUN / "builder_followup_integrity_v1.4mrTlm/canonical_combined_tests_v2.xml"
    tree = ET.parse(regression).getroot()
    suites = tree.findall("testsuite")
    tests = tree.findall(".//testcase")
    assert sum(int(s.attrib["tests"]) for s in suites) == len({(t.attrib.get("classname"), t.attrib["name"]) for t in tests}) == 276
    assert all(all(int(s.attrib.get(k, 0)) == 0 for k in ("errors", "failures", "skipped")) for s in suites)
    text = f"""# ASA：答案展示改写覆盖对照结果（v16）

记录时间（UTC）：{datetime.now(timezone.utc).isoformat()}。延续 20260908 实验系列。

v15 启动的审核已完成，本报告补充最终结果，不覆盖 v15 的当时运行状态记录。
241 规范运行时 Python 源码与默认配置哈希仍与 adapter12 预检一致。没有关闭校验、切模型、训练或改写旧数据标签；总目标仍在推进。

## 一、39 个直接回答的完整盲对照

原79个状态中选取全部39个最终direct状态，比较保存的 parsed/pre-validation answer 与真实 shown answer。
34对文本变化、5对相同。相同问题+答案只发一次，合计72个唯一API输入，全部收齐：**67成功、5失败**。

| 自动槽覆盖标签 | 校验前答案，39状态 | 实际展示答案，39状态 |
|---|---:|---:|
| complete | 34 | 31 |
| partial | 1 | 1 |
| none | 3 | 3 |
| audit_error | 1 | 4 |

**不能把34→31直接称为丢失3个正确答案。** 其中混有审核自身失败。

两边审核都成功的34对为：

| 自动标签变化 | 对数 |
|---|---:|
| complete → complete | 30 |
| none → none | 2 |
| partial → partial | 1 |
| complete → none | 1 |

其余5对至少一侧审核失败，保留在39状态分母中，不按错误或正确答案强行归类。
5个唯一输入失败分别为2个无效JSON、2个非逐字摘句、1个顶层字段误写。
其中一个响应包含类似JavaScript条件表达式的文本；只按JSON解析失败保存，未执行、未手改后补记成功。

请求内容、请求哈希、保存的原始响应、成功与失败的验证结果、共享输入关系及旧源文件哈希均已复核。
接口返回的 model 字段与 glm-5.3 一致，**这不是对服务商实际后端权重的独立认证**。

## 二、源文复核：不能统一恢复原答案

### 行星碎屑：确实丢了有来源的启动机制信息

这是唯一两侧审核成功且从 complete 变为 none 的状态。
parsed答案含“通过充能可以朝干员部署方向推动”；展示版只剩效果、轨道与重力约束。
完整E1–E8已读，E1明确给出相邻攻击充能、一定次数后朝部署方向推动，E3有巨大石块与静止阻路的信息。
因此可以确认：**有源文支撑的充能推动信息在展示改写中丢失了。**
但自动complete标签仍不是human gold，也不自动认证原答案的所有因果解释充分。
正确改进方向应是让需要的机制信息进入可追溯事实，并核验最终答案覆盖；不能直接取消引用保护。

### 梅希亚：看似确定性变化，读完原文并非负面改动

parsed版将对付莱茵的计划说成“发现”，把走私说成“怀疑”；展示版将“怀疑”放回计划，将走私写为发现。
完整E1–E6包含先前“还未求证”的计划猜测，以及后来的走私/违禁品对话和高登的相关承认。
因此不能仅看“怀疑”从一句移到另一句就判定展示层制造错误；这一处展示更贴合证据中的事件进展。
这也说明按整篇答案数“可能/怀疑”等词不等于逐事实的确定性校验。

### 虞澄：新增人物角色需区分两个阶段

完整E1–E6支持过去委托解真暗中寻访，以及后来需要麟青砚用公开手段使真相重见天日。
不能因展示版多写麟青砚的角色就简单判定错人。另一方面，两版对“不直接告知”的冒险/保密理由均较压缩，自动complete并不能替代这一语义复核。

其余39对答案已逐条读过；没有把全部30个complete→complete状态重新认证为事实正确。
乔迪亲属关系、莫斯提马身份、其他人物归属等既有源文问题，不会因本次“覆盖完整”而消失。

## 三、据此调整下一步重点

1. 不再仅凭“34/39被改写”就把展示改写当作主要瓶颈；文本变化率不是信息损失率。
2. 保留引用与来源保护，优先验证事实选择、人物归属、原因链以及遗漏的机制信息。
3. 对明确损失的信息研究可追溯修复，再做对照，而不是恢复未经核验的整段原答案。
4. 继续研究模型/训练质量，但不把开发题成绩或覆盖审核结果当作新端到端正确率。

241规范回归工件再次校验为276个唯一测试、0失败/0错误/0跳过；本轮未重跑训练或修改运行时源码。
248同步成果仍见v14：三个旧脚本已备份替换，依赖补齐，规范/候选/应用后各120项子集通过。

关键工件目录：`{HERE}`。

- `reconstruction_coverage39_v3.json`
- `reconstruction_coverage39_postflight_v3.json`
- `answer_reconstruction_structural39_v1.json`
- `slot_cause_scope40_postflight_v3.json`

最终审核输出 SHA256：`{result["report_sha256"]}`。
逐源文意见为独立助手复核，不是人工gold；旧79结果、旧判断和所有失败记录原样保留。
目前仍没有新增可认证的全链路回答正确率或RAG召回率。
"""
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"report": str(destination), "sha256": sha(destination), "runtime_and_config_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
