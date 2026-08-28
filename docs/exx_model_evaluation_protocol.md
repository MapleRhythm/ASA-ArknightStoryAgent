# Exx 模型与端到端评估协议

## 目标

主结论必须回答三个不同问题，不能用一个总分混在一起：

1. 新 adapter 在相同完整证据下是否比旧 adapter 更会选择动作、回答并绑定 Exx。
2. 真实检索链路是否把足以回答的问题证据送进 prompt。
3. 完整系统是否在 20 秒预算内提高正确回答率，同时降低幻觉和过度弃答。

## 数据隔离

- `eval50` 与 `hard10` 只作为历史回归集。当前 partial 训练集与其分别有 36/50、7/10 个精确问题重叠，不能报告为泛化结果。
- 主评测由 `scripts/build_exx_eval_split.py` 生成，并从实际训练 JSON 中按规范化问题文本排除所有重叠项。
- `model_only_labelled.json` 用于固定证据的模型层评测；`e2e_questions.txt` 用于真实链路评测。
- 任何 API judge 都只能看到问题、系统实际可见证据和模型输出，不能看到训练标签或隐藏 gold evidence。

## 对照组

至少运行以下两组，除 adapter 外所有变量严格一致：

- `baseline`: 训练起点 adapter。
- `exx_sft`: 新训练完成的 adapter。

推荐附加一个诊断组：

- `teacher_action_oracle`: 固定同一 prompt 证据，只用教师标签作为动作/事实上限，不进入产品对比。

解码固定为单样本、temperature 0、无 web、MiniRAG 关闭。检索、reranker、prompt 证据顺序与截断策略冻结。运行目录必须保存 config、adapter SHA256 和数据 SHA256。

## 第一层：固定完整证据的模型层评测

输入使用未进入训练的 labelled Exx prompt。对原始模型输出和运行时解析后输出分别计分。

自动指标：

- `strict_json_rate`
- `schema_valid_rate`
- `legacy_field_rate`: 出现 `quote/final_answer/inferred_facts/evidence_refs/answer`
- `valid_action_rate`
- `valid_e_id_rate`: 所有 E 编号均在当前 prompt
- `fact_count_valid_rate`: answer 为 1–8 facts
- `action_accuracy` 与三类 macro-F1
- `answerable_recall`: gold answer_directly 中模型 answer_directly 的比例
- `over_abstain_rate`: gold answer_directly 却 abstain
- `over_retrieve_rate`: gold answer_directly 却 retrieve_more
- `premature_answer_rate`: gold 非 answer_directly 却 answer_directly
- `exact_action_and_schema_rate`
- `parse_recovery_rate` 与 `runtime_rejection_rate`

语义指标使用盲审：

- `claim_support_precision`: 生成的原子事实中被其 Exx 正文直接支持的比例
- `question_coverage`: 问题核心信息需求被支持事实覆盖的比例
- `fully_supported_answer_rate`
- `unsupported_claim_rate`
- `usable_answer_rate`: answer_directly、完整回答、全部事实有支持

决策原则：模型输出正确但运行时拒绝，归因于 parser/validator；模型原始输出本身错误才归因于 adapter。

### 重复问题族、引用绑定与冻结集报告

- 行级指标之外，必须按规范化问题文本分组并报告 question-family macro action accuracy 与 schema+action accuracy，避免同一问题的多个 evidence state 被重复加权。
- 必须报告唯一问题族数、重复问题族数和 gold action 冲突的问题族数；冲突族不能被当作无噪声标签解释。
- 引用指标不能只比较整条回答的 E-ID 并集。应对每条预测 fact 与 teacher fact 做一对一最优匹配，并联合计算 fact 文本相似度与该 fact 自身 E-ID 的 Jaccard，报告 claim-local citation alignment。
- claim-local citation alignment 仍是对 teacher 标签的一致性指标，不等价于“引文语义蕴含该主张”；后者必须由候选正文盲审、NLI/verifier 或人工复核完成。
- 报告完全或规范化后重复 fact 的行级比例；重复 fact 导致生成到 token 上限或 JSON 未闭合时，必须单独列为生成退化。

当前用于 RLVR 开发的 `val79` 只有 58 个唯一问题族、20 个重复问题族，并存在 1 个 gold action 冲突族。它是冻结开发集，不可单独作为泛化或发布结论；模型晋级仍需去重、无冲突、与训练 question-family 隔离的主评测集。

## 第二层：检索与 prompt 可见证据评测

对去泄漏问题跑相同检索配置，生成一次 retrieval snapshot，随后 baseline 和新 adapter 共用该 snapshot，避免模型续检索差异污染第一轮比较。

有 gold evidence 的题统计：

- dense、BM25、融合池、reranker、最终 prompt 各自 Recall@1/3/5/10
- MRR、全 gold 单元覆盖率、任一 gold 单元命中率
- `retrieval_answerable_rate`: prompt 是否已经足够回答
- 来源 lane 的边际召回贡献
- reranker 正向提升率和误杀率

链路归因：

- 候选池无 gold：召回问题。
- 候选池有 gold、reranker top-k 无：排序问题。
- prompt 选择前有 gold、prompt 不可见：证据选择或上下文预算问题。
- prompt 已足够、模型 retrieve/abstain：动作模型问题。
- 模型给出支持事实但运行时拒绝：parser/validator 问题。
- 运行时接受但事实不受 Exx 支持：grounding/verifier 问题。

## 第三层：完整系统评测

每题保存：

- 原始模型输出、规范化输出、parse status、grounding issues
- 每轮 queries、候选与最终 prompt 证据
- 最终 action、facts、Exx 和用户可见答案
- 分阶段耗时：query planning、dense、BM25、fusion、rerank、prompt selection、generation、validation、total

主指标：

- `correct_answer_rate`: 正确且完整的用户可见回答 / 全部可答题
- `usable_answer_rate`
- `abstention_rate`
- `over_abstain_rate`
- `hallucination_rate`: 任一关键事实无其所引 Exx 的直接支持
- `error_rate`
- total latency p50/p90/p95/p99、`under_20s_rate`
- 可答题和不可答题分别报告，不用整体弃答率掩盖校准问题

API judge 采用成对盲审，顺序随机，至少对全部分歧案例重复两遍；不一致案例进入人工复核。最终同时报告 bootstrap 95% CI 和配对胜/平/负，样本较小时不只报告均值。

## 晋级门槛

partial/full adapter 进入下一阶段 KTO 或 GRPO 前建议满足：

- strict JSON >= 99%
- E 编号合法率 = 100%
- legacy field rate = 0
- unsupported claim rate 不高于 baseline，且目标 <= 5%
- over-abstain 相对 baseline 至少下降 20%，premature-answer 不显著上升
- usable-answer rate 有正向提升，配对 bootstrap 95% CI 不跨 0（样本不足时标为趋势）
- 端到端 p95 <= 20 秒；若当前硬件/单进程初始化计入方式不满足，需同时报告 warm p95 和 cold-start

未过门槛时先依据链路归因修对应层，不直接上 RL。
