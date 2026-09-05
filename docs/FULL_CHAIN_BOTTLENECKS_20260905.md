# ASA 全链路瓶颈与实验路线（2026-09-05）

## 当前结论

当前主要问题不是单一的 4B 生成能力，而是“检索到的证据集合”和“答案中的事实集合”没有用同一个覆盖目标优化。系统目前有四种不同的排序对象：

1. BM25/Faiss：单候选与问题的相关度；
2. BGE reranker：问题与单候选（以及候选链）的匹配度；
3. prompt selection：按已有分数选 top-k；
4. 生成器：在有限上下文中选择要写出的 facts。

它们没有共享“回答问题所需的信息需求”或“候选集合的联合覆盖”指标。即使每个阶段局部排序正确，也可能在最终 prompt 中丢掉第二段因果、主体或时间证据。

## 已有证据

- clean-SFT 与 gap-mix 的校准评估：格式提升明显，但 action 与 claim-local binding 没有提升。见 `docs/CALIBRATED_BINDING_EVAL_20260905.md`。
- binding verifier v2 在“主张→证据”难例上的 AUC 从 v1 的 0.805 提升到 0.904；这是绑定判别器结果，不是问题→候选检索 Recall。见 `docs/BINDING_VERIFIER_AUGMENTED_V2_FINDINGS_20260905.md`。
- B1/B2 验证表明重复原查询的新增覆盖为 0，而针对缺失证据构造的查询有明显增益；这说明二轮检索应由“未覆盖证据”驱动，而不是泛化改写。见 `docs/CLEAN_SFT_RETRY_AND_RETRIEVAL_FINDINGS_20260904.md`。
- GLM set-audit v2 在相同的 8 个问题族上发现：clean-SFT 为 4 complete / 1 contradicted / 1 partial / 2 none；gap-mix 为 5 complete / 0 contradicted / 1 partial / 2 none。gap-mix 修复了一个反证绑定，但其同题输出夹带了无关、虽被证据支持的事实，说明“证据支持”不能代替“问题相关性”。
- GLM set-audit v3 增加问题相关性轴后，clean-SFT 为 4 complete / 1 contradicted / 1 partial / 2 none、0 条 irrelevant；gap-mix 为 3 complete / 0 contradicted / 3 partial / 2 none、2 条 irrelevant。v3 证明 gap-mix 的绑定修复伴随回答范围膨胀，不能把 v2 的 5/8 complete 当作最终正确率。
- 当前 GPU runtime 的 `prompt_evidence_top_k=10`，而 `select_prompt_evidence` 默认只按 `prompt_evidence_score` 排序；它不计算问题信息需求的增量覆盖。MMR 仍关闭。见 `src/asa_arknight_story_agent/inference/evidence/prompt_ordering.py` 与 `configs/runtime_gpu_reranker_qwen35_4b.json`。
- 当前生产候选池配置仍为 dense 120 + sparse 120 → fusion 80 → rerank 32 → prompt 10。任何在 32 之后才需要的证据都无法恢复，且 prompt 选择还有一次额外截断。
- `binding_verifier v2` 的训练输入是 claim/evidence pair；生产 reranker 的输入是 question/document。两者必须分别报告，不能把 verifier 指标当作检索指标。

## 需要优先验证的三个断点

### A. 候选池 Recall

对每条问题记录 gold/校准 evidence 是否出现在：

- sparse top-120；
- dense top-120；
- fusion top-80；
- rerank top-32；
- final prompt top-10。

这五个数字能直接区分“召回不到”和“后续排序/截断丢失”。评估必须按 question-family 隔离，不能只看 79 条冻结集。

### B. 集合覆盖而非单文档分数

先不改 reranker 模型，增加一个离线候选选择 ablation：

- top-score；
- MMR；
- query-token coverage 的贪心集合选择；
- evidence-chain coverage（仅使用检索器已生成的 chain metadata）。

比较 final-prompt Recall、GLM blind set-support、生成耗时。只有 final-prompt Recall 提升且 p95 不超过预算，才考虑上线。

### C. 生成器是否在“证据已经到位”时仍产生错绑

对同一批输入固定 evidence，比较 baseline/clean-SFT/gap-mix：

- 每 fact 的 GLM support；
- `citation_complete`；
- set support；
- premature answer 与 over-abstain。

如果 evidence 到位仍错绑，才是生成/绑定训练问题；若固定证据后明显变好，则应优先修检索与 prompt selection。

## 协议修正

`audit_exx_set_grounding_v3` 是离线测量协议，不进入生产或 RLVR。它：

- 允许一个 fact 使用多个 E-ID 联合支持；
- 只允许检查该 fact 自己列出的 E-ID；
- 允许用其他 fact 解析代词，但不允许借用其他 fact 的证据作为隐含支持；
- 独立输出 `context_sufficiency`、`set_support` 和 `action_appropriateness`；
- 校验关系索引、关系证据必须来自两端引用并集；
- 逐条写入 progress JSONL，可断点恢复并有单写者锁；
- 对重复 fact 保留语义审计，但另报告结构不合格，避免把格式错误混入语义分母。
- 对每个 fact 增加 `question_relevance`（direct/supporting/irrelevant），避免把无关但真实的剧情事实算作完整回答。

## 暂不切生产的原因

1. gap-mix 在校准 binding 指标上反而下降，不能以格式收益替代正确率；
2. verifier v2 尚未证明 question→document Recall 提升；
3. prompt selection 仍是独立 top-score 截断，模型提升可能被最后一步抵消；
4. 冻结验证集只有 58 个 question family，且 gold 仍允许替代证据漏计；
5. 线上目标是单题 GPU 20 秒内完成，任何集合审计/GLM 只能离线。

## 下一轮可执行顺序

1. 完成 clean/gap 各 8 条不同 question-family 的 GLM set audit；
2. 统计 set-support、context-sufficiency、fact support 和动作混淆；
3. 产出 candidate-pool/final-prompt Recall 脚本与无重复 family 评测集；
4. 在同一候选池做 top-score/MMR/coverage 三路 ablation；
5. 只有当 Recall 提升且延迟合格，再训练 question→document reranker；binding verifier 保留为离线 verifier/reward；
6. 最后才在校准标签上做小规模 RLVR smoke，并与 clean-SFT、原模型三方盲测。

## 直接修复方向

生成训练不能只给“被证据支持的事实”正例，还要给“被证据支持但与问题无关的事实”负例。建议在 answer_directly 的每个 fact 上离线标注 `direct/supporting/irrelevant`，训练目标同时优化：

`fact_utility = evidence_support × question_relevance × set_coverage`

其中 `set_coverage` 只奖励回答问题尚未覆盖的信息需求，不奖励无关事实数量。RLVR 可以使用本地 verifier 的 support 分数，但 relevance 与覆盖应由离线 GLM/人工审计蒸馏后再使用；不能让 verifier 单独决定“回答是否正确”。
