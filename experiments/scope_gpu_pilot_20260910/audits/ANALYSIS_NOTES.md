# 分析口径与勘误

1. 原始动作 `answer_directly` 不自动等于有依据地回答。输出可能在 supported_facts 内写“无法确定”。
   动作与语义谨慎性需分别统计，不能把这种输出算作已确认幻觉。
2. JSON 语法错误、协议字段缺失、normalizer 是否接纳是不同指标。
   例如首轮 base 的 follow-up 字符串不符合旧严格评分 schema，但本地 normalizer 可以修复。
3. natural_controls 中 32 条标注 unknown；只统计行为，不算对错。
4. symbolic 合成支持可以分布在多个原子事实中。首版分析的“同一 fact 必须引用完整路径”偏向某一种写法，
   不适合作为问答正确率或必需条件；当前 analyzer 改为答案 token＋所有引用并集含一条合法路径。
   它仍只是表面代理，不验证每条 claim 的语义绑定、方向或额外错误。
   `audits/base_only_interim/` 如由早期版本生成，只作为中间尝试，最终以主 `audits/summary.json` 为准。
5. 对 minimal hypothesis 做本地 normalizer 回放时，未提供的内部 intent 用 `fact` 构造，
   此默认在逐条记录中明确，未修改真实模型输入；因此它不是远端生产 serving 结果。
6. 提示词补充对照是在阅读部分基础输出后设计的；属于探索实验，且多条规则一起变化、token 增加。
   不能分别归因给某个句子，也不是同计算预算或盲测泛化证据。
