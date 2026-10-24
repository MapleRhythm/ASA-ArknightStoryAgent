# ASA：planner 枚举串位的字段级恢复 v39

241 为唯一规范源；北京时间系列日期，机器回执为 UTC。
只对已保存的 v37 原始 payload 做 CPU 对照，没有新模型调用、检索或回答生成。
生产代码、默认模型、引用验证和训练数据均未改动。

## 恢复条件

仅处理 initial planner 把合法 query_type 枚举写入 intent、因而被 canonical
normalizer 明确拒绝的情况。原始 payload 保留；在副本里使用现有
`detect_intent(question)` 的 fallback intent，再运行完整原校验。

任意未知 intent、额外 schema 字段、缺失/无效实体或关键词仍然拒绝。
不扩大 intent 枚举，不恢复 follow-up 的非法 current_intent。
原错误、原枚举、替代值及来源放在独立诊断回执，不能作为检索词发送。

## 全量回放与两个候选

54 个原本有效的 payload 逐字段不变。4 个 `mystery/reveal` 枚举串位
可以恢复，原错误均保留：

| 问题 | 相比整体 fallback 恢复的题面完整名称 |
|---|---|
| 007 女妖与骸骨荒原 | 骸骨荒原 |
| 010 老鲤与梁洵 | 梁洵、宗师 |
| 046 偷取《武典》 | 槐天裴 |
| 049 保存者 | 无新增名称，原 fallback 已保留 |

基本字段恢复同时改变了四题的 query_type 和 expected_answer_type。
例如 046 会从原 fallback 的 reasoning 变为模型写的 fact，可能丢掉题目中的
动机请求。因此不能将这个版本当作仅恢复姓名的安全改动。

另一个更保守的候选保持整体 fallback 的 intent、query_type、
expected_answer_type、question、dialogue_context 原样，只补充经过原规范化的
检索线索。它保留全部旧实体与关键词；合并超出原容量则完全不恢复，不截断旧词。
4 题均未超限，完整题面名称得到保留，所有路由字段及旧线索保持。

协议测试 17 项在本地和 241 各自通过，计为 17 个用例，不累加为 34。
58 条回放与原输入/源码 SHA256 绑定，`checkpoint_v39.json` 已核验所有不变量。

## 当前决定

不把 4/4 schema 恢复说成 4/4 回答正确，更不说成全局正确率提高。
模型线索仍未经过剧情原文认证，旧的碎片启发式也仍存在。

基本字段恢复版本不进入下一步；仅保留“路由不变、旧线索不丢”的候选，
用于下一轮隔离检索验证。在确认检索与完整引用流程影响之前，不改生产行为。

规范目录：`outputs/experiments/planner_intent_field_fallback_20260910.v39/`。
工件：`preflight.json`、`replay_v1.json`、`checkpoint_v39.json`、
候选函数、协议测试及回放脚本。
