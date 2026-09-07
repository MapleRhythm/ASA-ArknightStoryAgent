# ASA 回答正确率提升研究方案

日期：2026-09-07
适用分支：`fix/grounding-pool-chain-priority`

## 1. 结论先行

当前 ASA 尚未达到生产级回答正确率。必须先把“回答正确率”与
`schema/action` 指标分开，并使用无问题族泄漏的端到端评测来指导改动。

当前可复核基线如下：

| 评测层 | 当前结果 | 可解释结论 |
| --- | ---: | --- |
| 固定完整证据、relevance-aware | 31/68 = 45.6% complete | 只代表 `answer_directly` 子集 |
| 固定完整证据、relevance-aware 折算全部 79 题 | 31/79 = 39.2% | 不能当作泛化准确率 |
| 固定证据、relevance-aware 未重合问题族 | 13/44 = 29.5% | 更接近真实泛化基线 |
| 固定完整证据、clean-SFT | 25/62 = 40.3% | 结构有所改善，语义仍不足 |
| relevance-aware 关键未支持主张 | 15/68 = 22.1% | 仍未通过研究门槛 |
| 最新 Qwen0.6B sparse-heavy 端到端 smoke | 1/8 明确正确、1/8 部分正确 | 样本太小，但说明真实链路明显弱于固定证据审计 |

因此，后续不应再用约 73%–82% 的 action accuracy、schema valid 或
verifier agreement 代替事实正确率。当前生产默认值保持不变：

- 最多 2 轮检索；
- GPU prompt evidence 默认 10 条；
- set-coverage、第三轮、parallel retrieval、compact graph 均只作旁路实验；
- 不把 GLM 直接接入线上；
- 不把 5588 条 `glm_verdict=not_run` hard-negative 当作干净负例；
- 暂不启动正式 RLVR。

## 2. 评测口径和错误归因

### 2.1 必须建立的主评测集

主评测集由 `scripts/build_exx_eval_split.py` 生成，必须满足：

1. 与实际 SFT 训练问题精确去重；
2. 用规范化文本排除近重复问题（当前默认阈值 0.88，需记录阈值）；
3. 按 question-family 做隔离和宏平均；
4. 保留一份固定证据的 model-only 集和一份真实检索的 e2e 问题集；
5. 保存输入、配置、模型/adapter SHA256，禁止把训练重合题混入主要结论。

历史 79 题继续作为回归和诊断集，不作为泛化发布集。已有 34 个问题族与
relevance-aware 训练数据重合，必须单独报告重合/未重合结果。

### 2.2 三层指标

第一层：固定证据的模型层。

- strict JSON、schema valid、合法 action、合法 E-ID；
- `answer_directly`、`retrieve_more`、`abstain` 的 confusion；
- premature answer、over-abstain、over-retrieve；
- duplicate fact、truncation、legacy field；
- 每个 fact 的 semantic support、claim-local citation、question relevance；
- `complete / answer_directly`、`complete / all rows`、关键 unsupported/contradicted。

第二层：检索和 prompt 可见性。

- sparse/dense/fusion/rerank/final prompt 各阶段 Recall@1/3/5/10；
- 任一 gold evidence 命中率、全 gold 单元覆盖率、MRR；
- `context_sufficiency`；
- prompt selection 丢失率；
- round 间 top-k Jaccard、novel-doc rate、query 重复率。

第三层：完整系统。

- 用户可见的完整正确回答率；
- unsupported/contradicted、irrelevant fact、premature answer、over-abstain；
- 可答题与不可答题分开报告；
- warm p50/p90/p95/p99、`under_20s_rate`；
- 保存每轮 query、候选、最终 prompt、输出、解析和 grounding 结果。

### 2.3 固定归因规则

| 现象 | 归因 |
| --- | --- |
| candidate pool 没有必要证据 | 召回/索引问题 |
| candidate pool 有，rerank 后消失 | 排序问题 |
| rerank 有，final prompt 不可见 | prompt selection/预算问题 |
| prompt 已充分，模型仍 retrieve/abstain | 动作模型/充分性判断问题 |
| 模型引用不受其 E-ID 支持 | grounding/生成绑定问题 |
| 模型原始 JSON 正确，但运行时拒绝 | parser/validator 问题 |

## 3. 已确认的根因

### 3.1 证据覆盖是第一瓶颈

当前 GPU 链路为 dense 120 + sparse 120 → fusion 80 → rerank 32 →
prompt 10。79 题实际证据顺序审计显示：

- top-10 能覆盖全部 gold E-ID 的题目为 59/62；
- top-16 为 62/62；
- 最大 gold 出现位置为 E12。

这说明至少一部分错误发生在候选已找到、但最终 prompt 截断过早，而不是
reranker 完全没有召回。

### 3.2 充分性是集合属性，当前没有显式缺口状态

单候选分数池化特征对 chain completeness 的 AUROC 约为 0.45–0.55，
不能通过一个阈值判断“是否已经足够回答”。在 50 个
`retrieve_more → answer_directly` 分歧中，仅约 2% 可视为模型完全正确，
约 34% 是缺关键面仍抢答，约 64% 明确 unsupported 或引用幻觉。

因此需要把问题拆成实体、动作、原因、关系、时间、结果等 slot，并显式记录
`supported / missing / uncertain`，而不是让生成器同时猜缺口、决定动作、
生成事实和绑定 E-ID。

### 3.3 多轮检索主要在重复

既有 trace 审计：

- round1→2 top-5 Jaccard 均值 0.729；
- round2→3 均值 0.936，29/35 完全相同；
- follow-up question 文本在 46/46、35/35 个过渡中不变；
- cumulative Recall@10 仅 0.68→0.74→0.76；
- round3 fresh Recall@10 降至 0.52。

重复原查询的新增 coverage gain 为 0；oracle 缺口 query 的平均 gain 为
0.923、novelty 为 0.821。后者是上限，不是可直接部署的 planner。

### 3.4 生成格式已改善，但仍有重复和绑定错误

SFT/KTO 已显著改善 schema 和重复输出，但仍存在：

- stop-malformed（通常只缺外层一个 `}`）；
- max-token 截断；
- 重复 fact/keyword 循环；
- partial evidence 被写成完整 fact；
- 主语、身份、因果和多子句绑定错误。

GOLD_RECAL 的 1929 个 fact 中，只有 1518（78.69%）为 clean support，
288（14.93%）unsupported，50 道题因事实被删除而发生动作翻转。原 gold
不能未经 set-level 复核就直接用于 SFT 正例。

## 4. 实验路线（按 ROI 排序）

以下实验均先在 family-held-out 数据上进行，并使用同一随机种子、同一候选
快照和同一解码参数。每个实验必须保存逐题结果，不能只保存均值。

### E0：评测与链路归因面板（先做，离线）

**目的**

把候选池、rerank、prompt、动作、生成和 grounding 的错误分开，防止以后
再把 Recall、action accuracy 或 verifier AUC 误报成回答正确率。

**实现范围**

- 新增只读脚本
  `scripts/evaluate_retrieval_trace.py`，统一读取 JSON/JSONL stage snapshot；
- 每个 snapshot 必须显式提供 `gold_evidence_ids`、`stages.dense`、
  `stages.sparse`、`stages.fusion`、`stages.rerank`、`stages.prompt`；
- 脚本同时输出 unit Recall@k、任一命中率、全 gold 覆盖率和逐题记录；
- 每题记录 gold evidence 是否出现在 sparse、dense、fusion、rerank、prompt；
- 若提供 `rounds`/`retrieval_trace`，记录 `context_sufficiency`、missing slots、
  round overlap、novel-doc rate 和 query 重复率；
- 结果按 question-family、query type、证据链长度分层；
- 对全程 miss 做 span-level corpus coverage 检查，不能只查拼接后的
  `gold_excerpt` 是否整段命中。

**通过条件**

- 同一输入能稳定重算；
- 缺失 stage 时明确报错，不从其他 stage 猜测或补数据；
- 每个错误能落入唯一主要归因；
- 回归测试覆盖重复问题族、缺失 E-ID、非法 schema 和空证据。

**上线边界**

只新增离线报告和测试，不改变任何默认配置。E0 未完成前不启动训练或 RLVR。

### E1：集合级 prompt evidence coverage（最高优先级）

**假设**

当前 top-score 选择会让高分重复片段挤掉互补证据；贪心 set coverage 或
top-16 能提高 final-prompt Recall，从而提高回答完整率。

**对照和变体**

在同一 rerank 候选池上比较：

1. baseline：top-score，top-10；
2. MMR，top-10；
3. query-token coverage，top-10；
4. query-token coverage，top-16；
5. oracle upper bound：只用于核对 coverage 上限，不用于生成。

保持 reranker、证据文本、顺序随机种子和生成模型完全一致。

**指标**

- final prompt 全 gold 单元覆盖率、任一 gold 命中率；
- `context_sufficiency`；
- complete rate、question coverage；
- claim support precision、关键 unsupported/contradicted、irrelevant；
- prompt 字符数、生成 token 数、warm p95。

**成功门槛**

- 在未重合 question-family 上，final-prompt 全 gold 覆盖至少提高 3 个百分点，
  或达到 98%，且不能在另一份独立集下降超过 1 个百分点；
- complete rate 至少提高 5 个百分点，配对 bootstrap 95% CI 不跨 0
  （样本不足时只能标记为趋势）；
- unsupported/contradicted 不比 baseline 增加超过 2 个百分点；
- warm p95 不超过 20 秒，且 `under_20s_rate` 不下降超过 5 个百分点。

**停止条件**

coverage 提升不能传导到完整回答、或者无关/未支持事实显著增加时停止。

**上线边界**

现有 `runtime_gpu_reranker_qwen35_4b_evidence_id_coverage_ablation.json`
只作旁路 ablation。E1 通过前不替换 GPU/CPU/API 默认 top-k 或选择器。

### E2：slot ledger + 外置 sufficiency gate（第二优先级）

**假设**

“是否完整”必须显式建模为证据集合状态。专用 gate 比让 4B 在一次生成中
同时完成 sufficiency、动作、事实和 E-ID 绑定更可靠。

**诊断数据构造**

对每个问题制作四种配对条件：

| 条件 | 证据构造 | 期望行为 |
| --- | --- | --- |
| Full | 完整 gold chain + 固定干扰 | answer_directly |
| One-edge-missing | 只删除一个必需链环 | retrieve_more |
| None | 删除全部 gold chain，保留等量干扰 | retrieve_more/abstain |
| Decoy-full | 高相关但不组成完整链的片段 | retrieve_more |

每种条件比较 raw evidence prompt 与 slot-ledger prompt。slot ledger 先由
离线 harness 注入，目的是判断“组织信息”是否能修复动作边界，不直接假设
模型已经学会自动拆 slot。

**指标**

- sufficiency/missing-slot precision、recall、F1、Brier/ECE；
- One-edge-missing 的 retrieve_more recall；
- Full 条件 complete rate 和 claim-local support；
- premature answer、over-abstain、错误 E-ID；
- slot ledger 的额外 token 与延迟。

**成功门槛**

- One-edge-missing 的 retrieve_more recall 至少达到 60%，并较 raw prompt
  提高 25 个百分点；
- premature answer 相对 baseline 下降至少 30%；
- Full 条件 complete rate 不下降超过 2 个百分点；
- over-abstain 增加不超过 5 个百分点；
- gate 的低置信度样本必须显式标为 `uncertain`，不能强行 answer。

**停止条件**

Full 仍明显失败、或 slot ledger 不能改善缺口识别且引用仍大面积错误时，
停止继续堆 planner，转向生成器能力/绑定训练。

**上线边界**

先做离线 gate 诊断。只有在 E1 和 E2 都通过、并有独立盲测证据后，才考虑
把 gate 接入运行时；GLM 仅作离线标注，不进入线上依赖。

### E3：gap-driven 多轮检索（第三优先级）

**假设**

第二轮应针对未覆盖 slot 检索，而不是复述原问题。查询需要保留原 query
作为 anchor，并对已见文档/source 加 novelty penalty。

**变体**

1. 当前模型 follow-up；
2. 重复原 query（负对照）；
3. oracle missing-slot query（上限对照）；
4. slot-conditioned query：每个 missing slot 一个 query；
5. slot-conditioned + anchor + seen-doc/source penalty + query-key 去重。

**策略约束**

- follow-up question 不得无条件复制上一轮；
- `missing_slots` 经过 `clean_missing_slots_for_retrieval`；
- 每个缺口最多生成有限 query，去重后再检索；
- 无新文档、无 coverage gain 或 query 高度重复时提前停止；
- 第三轮只作为 recovery，不作为默认路径。

**指标**

- round-only 和 cumulative Recall@10/@20；
- top-5 Jaccard、novel-doc/source rate、query duplicate rate；
- 每个 missing slot 的命中率；
- complete rate、unsupported/contradicted、premature answer；
- 每增加一轮的检索和总延迟。

**成功门槛**

- round2 fresh Recall@10 相对当前至少提升 10 个百分点；
- round1→2 top-5 Jaccard 降至 0.60 以下，且 novel-doc rate 不低于 0.35；
- cumulative Recall 不低于 baseline；
- e2e complete rate 提高至少 5 个百分点；
- 无新证据时 90% 以上样本能提前停止，且 unsupported 不增加超过 2 个百分点。

**停止条件**

overlap 下降但 fresh recall 不升、或 query 注入猜测性实体导致 unsupported
上升时停止。oracle 也命不中时，先回查索引切片/语料覆盖，不继续调 planner。

**上线边界**

E3 全部为离线 ablation。第三轮保持默认关闭；即使 E3 通过，也要在独立
e2e 集和 20 秒预算内复测后才能考虑默认启用第二轮策略。

### E4：最小充分事实 + 结构化解码（第四优先级）

**假设**

格式失败主要来自自由展开、重复和截断；结构化解码可以降低解析损失，但
不能把格式改进误认为事实正确率改进。

**变体**

1. 当前自由 JSON 解码；
2. grammar/constrained JSON；
3. grammar + answer 只允许 1–3 条原子事实；
4. grammar + 受限 follow-up 数组 + 重复循环检测；
5. 仅对 `finish=stop`、无尾部垃圾、只缺一个外层 `}` 的输出做白名单安全修复，
   修复后重新执行 schema/E-ID/grounding。

**指标**

- strict JSON、schema valid、合法 E-ID；
- duplicate fact、truncation、legacy field；
- action accuracy、premature/over-abstain；
- complete rate、claim support、关键 unsupported；
- parser rejection 与修复后重新校验通过率；
- 额外解码延迟。

**成功门槛**

- strict JSON ≥ 99%；
- invalid E-ID = 0；
- duplicate fact ≤ 10%；
- 截断率 ≤ 1%；
- semantic unsupported 不高于 baseline，complete rate 不下降。

**停止条件**

格式提高但 unsupported、irrelevant 或 premature answer 上升时，不能上线；
仅增加 `max_tokens` 不算通过方案。

**上线边界**

结构化解码能力按 backend 分别验证。安全修复必须是白名单、可审计、修复后
全量校验；任意截断文本不得直接放行。

### E5：训练数据重建与 RLVR 先决条件（第五优先级）

**数据重建**

- SFT 正例只保留 `clean_support` 且 set-level complete 的样本；
- `partial` 事实删除后必须重新做 set-level audit，不能直接当完整正例；
- 从完整 gold chain 删除一个必需链环，生成反事实
  `retrieve_more + missing_slot` 样本；
- 增加身份/主语归属、因果、否定、多子问、无关但真实事实 hard negative；
- 只使用已 GLM 确认的 1005 条 hard negative；
- 5588 条 `not_run` 仅作 provisional 候选，不能作为干净负例；
- 按 question-family 做 train/val/test 隔离。

**训练顺序**

1. 数据清洗和 set-level 复核；
2. fixed-evidence SFT；
3. 结构化解码和 slot/gap 诊断；
4. 小规模 KTO 或 RLVR pilot；
5. 独立 blind e2e 复测；
6. 只有稳定正收益才扩大训练。

**RLVR 允许启动的前置条件**

全部条件必须满足：

1. E0 面板完成，且训练/验证/测试问题族无泄漏；
2. gold recalibration 的原始 judgement、变更和 set-level 复核可追溯；
3. 本地 verifier 只作为离线审计或 pilot reward，不能把 91.2% agreement、
   AUC 0.937 当作事实正确率；
4. verifier 在 fact/relation 正例难例上有独立验证，尤其处理主体、身份、
   因果和多子句完整性；
5. E1/E2 的 evidence coverage 和 sufficiency gate 至少完成诊断；
6. baseline、clean-SFT、候选 RLVR 三方在 family-held-out 集上可配对比较；
7. pilot 至少同时报告 complete、support、unsupported、premature、
   over-abstain 和延迟，不能只报告 reward/action。

**研究晋级门槛**

- fixed evidence 的 complete rate 相对 clean-SFT 提高至少 5 个百分点；
- 未重合 family-held-out 的 complete rate 有正向配对差异；
- claim support ≥ 0.80；
- 关键 unsupported/contradicted ≤ 20%，且不高于 baseline；
- duplicate fact ≤ 10%、invalid E-ID = 0；
- premature answer 不显著上升，over-abstain 不以明显增加为代价；
- warm p95 ≤ 20 秒。

**生产门槛**

生产还需更严格地满足：

- strict JSON ≥ 99%、schema/E-ID 通过率稳定；
- claim-level unsupported 进入现有生产协议的严格门槛（目标 ≤ 5%）；
- 用户可见回答的完整正确率在独立 e2e 集上有稳定正收益；
- 所有分歧样本完成双盲复核；
- warm/cold 延迟和降级行为有单独报告；
- 回滚路径和版本化配置已准备好。

如果 RLVR 只提高 action accuracy、schema 或 reward，而 complete/support
不升，应立即停止扩大训练，回到 E1–E3 的链路归因。

## 5. 预期收益、风险和资源排序

| 方案 | 预期收益 | 主要风险 | 初始成本 | 默认配置建议 |
| --- | --- | --- | --- | --- |
| E0 评测面板 | 高可解释性，避免误判 | 不能直接提高质量 | 低 | 立即做，离线 |
| E1 set coverage | 修复 prompt 截断和互补证据丢失 | token/延迟增加；词法 coverage 可能误导 | 低-中 | 先 ablation |
| E2 slot/gate | 修复抢答和缺口不可见 | gate 误拒答、标注成本 | 中 | 先诊断 |
| E3 gap-driven rounds | 提高新鲜召回，减少重复 | query 猜测、索引边界、延迟 | 中 | 第三轮关闭 |
| E4 constrained decode | 降低 JSON/截断失败 | backend 差异；格式掩盖语义问题 | 低-中 | 通过语义门槛后再上 |
| E5 SFT/RLVR | 可能改善生成和动作边界 | 放大噪声标签或 verifier 偏差 | 高 | 最后做 |

推荐执行顺序：

`E0 → E1 → E2 → E3 → E4 → E5`

其中 E1 和 E2 可以并行准备数据，但正式训练必须等待 E0 的数据隔离和
归因报告完成。

## 6. 当前不应做的事情

以下行为在现有证据下都不应作为默认质量改动：

1. 仅因为 sparse-heavy 的 Recall@10/20 较高就替换生产融合权重；
2. 仅因为 compact graph 能把首轮检索从约 18.73 秒降到约 9.76 秒就上线；
3. 默认打开 top-16 + set coverage；
4. 默认打开第三轮、parallel retrieval 或 compact graph；
5. 把 GLM/4B agreement 91.2% 直接解释为回答事实准确率；
6. 把 5588 条 `not_run` hard negative 当作干净训练负例；
7. 把有训练问题族重合的 79 题上的 31/68 当作泛化提升；
8. 删除 unsupported fact 后不重新审计，就把剩余半答案作为完整 SFT 正例；
9. 只看 action/schema/reward，不看 complete、support、irrelevant 和
   premature answer；
10. 在尚无独立盲测和语义门槛时启动正式大规模 RLVR。

## 7. 建议的下一次执行清单

1. 固化 family-held-out 的 model-only 与 e2e manifest，并写入 SHA256；
2. 为每题生成 sparse/dense/fusion/rerank/prompt 五阶段 evidence snapshot；
3. 跑 E1 的 top-score/MMR/coverage/top-16 四路选择器 ablation；
4. 构造 Full/One-edge-missing/None/Decoy-full 配对集，跑 E2 raw vs slot ledger；
5. 对 E3 保存每轮 query、top-k、novelty 和 coverage gain；
6. 先在固定证据集测试 constrained decode 和安全括号修复；
7. 汇总 complete/support/unsupported/action/latency 的配对报告；
8. 仅当所有前置门槛满足后，才启动小规模 RLVR pilot；
9. 本轮研究结果写入独立版本目录，不覆盖既有生产配置或历史审计产物。

E0/E1 的离线执行命令如下。manifest 中的 `gold_evidence_ids` 必须是真实
corpus document ID，不能使用 prompt-local 的 `E1/E2` 编号代替：

```bash
python3 scripts/build_retrieval_stage_snapshots.py \
  --manifest <family-held-out-manifest.jsonl> \
  --runtime-config configs/runtime_gpu_reranker_qwen35_4b.json \
  --output outputs/retrieval_stage_snapshots/<name>.jsonl

python3 scripts/evaluate_retrieval_trace.py \
  --snapshots outputs/retrieval_stage_snapshots/<name>.jsonl \
  --output outputs/retrieval_stage_snapshots/<name>.metrics.json

python3 scripts/evaluate_prompt_selection_ablation.py \
  --snapshots outputs/retrieval_stage_snapshots/<name>.jsonl \
  --output outputs/retrieval_stage_snapshots/<name>.selection.json
```

## 8. 交付判定

本方案阶段的完成标志不是“训练出了一个新 adapter”，而是：

- 能指出每个错误发生在召回、排序、prompt、动作、生成还是 grounding；
- 至少一个离线实验在无泄漏集上同时改善 complete/support，且不牺牲延迟；
- 所有负面改动都有可复现的对照和回滚依据；
- 生产默认值仍只包含经过独立盲测验证的改动。
