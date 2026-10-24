# 保留归因作用域的替代证明：下一阶段 pilot 规范

状态：设计，尚未开始新推理/训练。主结论见 [研究报告](RESEARCH_REPORT.md)。
当前 84 条规则探针和上一轮 strict9 衍生数据全部是开发资产，不是待训练 gold 或新盲测集。

## 1. 要回答的三个实验问题

| 假设 | 配对对照 | 如果不成立 |
|---|---|---|
| H-data：合法替代支持标注能减少错误弃答 | 单一 gold 删除规则 vs 全候选重新审定；同训练目标 | 优先审计标签，不能硬造替代证明 |
| H-scope：作用域保持/破坏难例带来超越实体覆盖的自然泛化 | 同数据量的普通关系难例 vs 作用域难例；文本/结构基线 | 收缩为一般数据清洗或已有归因方法适配 |
| H-pair：跨状态目标有独立贡献 | 相同状态、辅助任务、样本权重、token/step 的 SFT vs 加配对项 | 不宣称新的后训练方法 |

先预先锁定主指标、允许的语义匹配规则、切分和停止条件；开发中可以修订，但必须版本化并重新冻结测试集。
负结果与正结果都保留，不能看完测试后只留下有利类型。

## 2. 每个问题族需要的离线标注

建议最小记录，字段名称是实验内部设计，不是生产接口变更：

```json
{
  "family_id": "new_story_event_001",
  "split": "unassigned_until_source_cluster_split",
  "question": "...",
  "accepted_question_premises": [],
  "required_answer_atoms": [],
  "optional_answer_atoms": [],
  "source_policy": "explicitly_reviewed",
  "visible_sources": [
    {"stable_id": "...", "text_sha256": "...", "spans": []}
  ],
  "claim_annotations": [
    {
      "claim_id": "...",
      "predicate": "...",
      "arguments": [],
      "scope_path": [],
      "event_time": null,
      "source_spans": [],
      "support_status": "annotation_unknown"
    }
  ],
  "accepted_support_sets": [],
  "support_search_scope": "finite_frozen_candidate_pool",
  "candidate_pool_review_complete": false,
  "review_status": "not_independently_reviewed",
  "exclude_from_training": true
}
```

`scope_path` 至少区分来源、言说/信念/意图、否定附着位置；时间未知不得补造。
alias 桥接有自己的原文出处，不能仅因姓名共现就合并。
证明集必须覆盖同一版必要答案；缺可选细节不影响可答，但回答主动添加时仍需证据。

至少逐题审阅以下问题：

1. 删除已标注段后，另一段是否仍可结合题设形成答案？
2. 所谓独立证据是否只是重叠 chunk 或转抄同一话语？
3. 某个主张的反例是否实际上允许回答“否”，而不是信息不足？
4. 整句转述的外层发言者是否被错当观点持有人？
5. “未提到”与“明确否定”、角色间分歧与同层冲突是否分开？
6. 当前有限候选池的判断是否被错误推广成全语料真值？

不确定样本隔离；模型间一致不能自动替代原文审阅。
正式测试需独立于方法开发的复核者；若只能由 Codex 审阅，报告必须注明证据等级，不标“人审 gold”。

## 3. 数据切分与干预控制

先按来源故事、事件、原始问题和原文近重复关系形成分组，再划分开发/训练/测试，最后生成变体。
若这些约束连成较大的来源组件，组件整体分配；不能为了凑比例拆开泄漏组。

每族目标是获得若干**已经审定**的状态，而不是强制所有问题都有两套证明：

- 合法 A/B/AB；
- 完全断裂和可回答但答案改变，分开；
- 静态补证；
- E-ID 重编号、证据顺序；
- 对归因、方向、时间、否定的单因素或预注册交叉对照；
- round-only 与 hypothesis-only 控制单列。

自然证据不改写原文来假装真实来源。人工生成的反事实文本单列 synthetic/counterfactual，
不塞回检索索引，不把生成段当原剧情。
同类支持状态中也要有否定词，非支持状态中也要有答案名字，避免表面分类捷径。

长度匹配按实际 tokenizer 的 prompt token 计算；字符长度只能做初筛。
对无法严格等长的自然配对，保留差值，并报告分层结果；不得偷偷截断限定语以满足预算。

## 4. 推理输入与内容绑定

reader 仅看到现有 non-thinking 协议需要的 question、允许的历史、hypothesis 和可见原文。
不得传入 gold atoms、proof graph、support labels、pair type、expected action 或正确答案片段。
训练时的监督目标可以使用这些离线标注，推理输入不可以。

每条保存：

- checkpoint/adapter/tokenizer/chat-template 标识和哈希；
- 实际 `enable_thinking=False`、完整 system/user prompt 和各自哈希；
- source 文本、引用映射、query 与索引版本；
- prompt/output token 数、上下文上限、finish reason；
- raw output、解析输出、normalizer 输出、grounding 后输出；
- wall time、调用数、随机种子及 sampling 参数。

不支持 non-thinking、触及输入上限或输出截断时单独记录失败，不删样本后重算分母。
先做 2–4 条技术 smoke，不用 smoke 选最有利 checkpoint。
结果放 `runs/<新且唯一的run_id>/`；不能读取旧预测伪装成新实验。

## 5. 训练预算与对照公平性

先复用相同 4B 基座进行隔离 adapter 试验，不把历史 adapter 当无污染 gold 标准。
固定 retriever、渲染器、引用上限、可见证据预算、推理轮次和服务门禁。

训练对照至少匹配：

- 训练问题族、所有状态及其采样权重；
- Exx 和辅助状态任务的比例；
- 训练 token、优化 step、batch/累积策略；
- 推理调用数、输出预算；
- 开发集调参次数和随机种子。

配对训练会增加同时处理的状态数，须记录实际显存、前后向次数和 GPU 时间。
给普通 SFT 同等训练预算，不能仅保证 epoch 相同。
先试少量 step 测峰值，再确定 LoRA 配置；本规范不承诺单卡容量或运行时长。

禁止直接训练当前隔离样本；未经来源/标签验收不得把 `exclude_from_training` 改为 false。

## 6. 指标：四种问题分开回答

1. **单状态完整正确**：必要答案完整，主动添加的事实有支持，引用合法；分开记录原输出与门禁后结果。
2. **证明替换稳健性**：A、B 均充分时两者都正确的族比例；不使用字符串一致率，也不把“两边都弃答”算稳健。
3. **断裂/答案改变响应**：分别统计错误抢答、过度弃答，以及有充分否定答案时是否正确回答。
4. **恢复**：
   - static repair：补回证据后的固定输入效果；
   - actual retrieval repair：真实 issued queries 返回并显示必要支持后，完整作答的比例。
   两者不可混称。

补充报告 claim 的来源/角色、否定、时间错误，引用集合支持和输入/输出成本。
无支持率不能单独作为主指标；同时报告 coverage、完整正确率与弃答率。

按独立问题族先聚合再比较，提供配对差值、退化数和分组置信区间。
变体级统计仅作机制分析；不能用四组改名把有效样本量扩大四倍。
对重复共享正文的自然探针，不作独立样本显著性推断。

## 7. 真实检索阶段的额外要求

仅在静态阅读和标签通过验收后开展。模型必须真的执行一轮新的检索：

```text
已见证据
  → 原始 follow-up
  → normalize / merge / 全部分支
  → 带真实历史的 scheduler
  → 实际 issued queries
  → 检索结果与可见 evidence packing
  → 新 reader 输出
```

分别统计“查询找到支持”“支持被渲染展示”“reader 正确使用支持”，避免把 packing 失败算成 rewriter 失败。
记录哪些跳只用已有缓存、哪些调用真实发生。新文档率仅作诊断，不等于有效修复奖励。
不从 frozen gold 查询获取补证再声称模型自己找回。

## 8. 晋级和停止

先做新增独立自然问题的 pilot；只有同数据强对照留下清楚差异，才扩大样本和训练预算。
当前无实测增益，不能预填 expected improvement、训练完成时间或论文接收概率。

如果收益只来自数据修正、清理 hypothesis、改排序或多给 token，就如实归类为数据/工程结果。
如果差异只存在 oracle 结构、模板内或旧 strict9，暂不主张可部署泛化。
即便得到正结果，仍需完成近期文献查全、跨领域验证和独立评测，才讨论论文贡献。
