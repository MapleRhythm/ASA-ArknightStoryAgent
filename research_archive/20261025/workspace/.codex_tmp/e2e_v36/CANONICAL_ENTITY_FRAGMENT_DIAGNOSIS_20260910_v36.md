# ASA：规划实体碎片只读诊断 v36

241 为唯一规范源；北京时间系列日期，机器回执使用 UTC。
本轮没有改生产抽取、没有删词、没有重跑检索，也不是新准确率结果。

## 可复现原因

`common/patterns.py` 先把连续中文按 2–8 字切窗；
`planning/query_tokens.py` 再对每个窗口拆功能词；
`planning/entity_extraction.py` 使用较宽的候选过滤；
`payload/hypothesis_payload.py` 把这些启发式实体追加到模型实体列表。

煌题的输入会产生“出煌”“挂面做法细”“节不足后”“仍然决定继续教煌”，
而单字“煌”本身未被这条启发式路径识别。
雅拉题把“克丽斯腾”切成“克丽斯”“腾视”等片段。

以明确标注的合成 payload `["小大厨", "煌", "挂面"]` 做隔离重现，
normalization 确实会追加上述四个碎片。合成 payload 不是找回的原始模型输出，
不是人工标签，也不能声称它证明旧模型最初给出了干净结果。

前三个源文件与 HEAD 字节相同。实体启发式 merge 在 HEAD 已经存在，
最近清洗补丁只是在它之前清理 diagnostic terms；不能把此缺陷归咎于该补丁，
也不能据此整段撤回已有清洗修复。

## 原始 planner 数据边界

另行检查 fixed58 的原结果文件和 trace 写入代码：
trace 中的 hypothesis 来自 `asdict(hypothesis)`，而生成入口返回的是
normalization 或启发式 fallback 后的对象。
顶层及各轮 trace 没有保存 raw planner 字段。

因此不能把保存列表减去启发式列表，就冒充恢复出的原始模型 payload。
后续 normalization 对照必须使用明确的合成输入，或另做新生成并保存 raw。
此结论限于所查 JSONL 和代码，不宣称遍历过服务器上的所有旧日志。

## 修复边界

`extract_content_tokens` 同时被 grounding 使用；全局替换分词会联动引用校验，
不属于一个隔离的规划修复。本轮不作该改动。
下一步先检验“有效模型 payload 的实体是否需要再次无条件追加启发式片段”，
保留 fallback、关键词、别名、引用及检索规则作为对照边界。
未经验证的候选不能上线，开发题结构检查不能充当端到端正确率。

规范目录：`outputs/experiments/entity_fragment_diagnosis_20260910.v36/`。
工件：`diagnosis_v1.json`、`trace_provenance_v1.json`，以及各自只读脚本。
输入结果和被检查源码均做 SHA256 绑定；诊断前后源文件保持不变。
