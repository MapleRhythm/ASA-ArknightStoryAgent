# ASA strict9 retrieval smoke findings

日期：2026-09-07
分支：`fix/grounding-pool-chain-priority`
代码基线：`aa2545f438ae9a2e7d2044a0aea017d34bf655b9`

## 1. 结论

本轮完成了 9 个 family-held-out 正例任务的严格 corpus-ID manifest、五阶段
retrieval snapshot 和同 rerank 池 prompt-selection ablation。实验没有调用
答案生成器，也没有修改生产配置。

主要结论：

1. 这 9 题的 32 条 rerank 原池均包含全部 GLM 校准引用，原池
   all-gold coverage 为 9/9。
2. 当前 top-score、MMR、query-token coverage 在 top-10 和 top-16 上结果
   完全相同：ID Recall 为 83.3%，all-gold coverage 为 5/9。
3. top-16 和 set coverage 在该 smoke 上没有收益，不能据此替换默认
   top-score@10。
4. 旧报告的 4 个 `prompt_drop` 不能全部解释为 selector 失败：
   3 题是公共文本去重删除了 gold 文档 ID，只有 1 题是去重后仍存在的
   gold ID 未被 selector 选中。
5. 3 个 dedupe-drop 题也不能直接解释为证据内容丢失。4 个被删 gold chunk
   中，3 个被完整包含在保留候选的 `evidence_chain_text` 中；剩余一对是
   95.9% token 重叠的相邻 chunk。
6. 当前 all-gold 指标还把 GLM 输出中的全部辅助或冗余引用都当作回答必需
   证据。原始标签复核表明，该假设不成立。
7. 因此本实验暴露的首要问题是评估表示不理解 fact citation group、
   evidence chain member 和重叠切块，而不是已经证明生产去重器或
   prompt budget 有缺陷。

该实验只有 9 题、9 个问题族，只能作为链路 smoke，不能更新回答正确率。
当前回答正确率仍沿用既有人工/GLM 语义复核：

| 口径 | 结果 |
| --- | ---: |
| 固定完整证据、relevance-aware complete / answer 子集 | 31/68 = 45.6% |
| 折算全部 79 题 | 31/79 = 39.2% |
| 未重合问题族 | 13/44 = 29.5% |
| 固定完整证据、clean-SFT | 25/62 = 40.3% |
| relevance-aware 关键 unsupported | 15/68 = 22.1% |
| 最新端到端 8 题 | 1/8 明确正确，1/8 部分正确 |

系统仍未达到生产级。

## 2. 数据和可复现性

严格 manifest 来源：

- 626 条 Exx task；
- family-held-out KTO val 的 25 行、12 个正例 task；
- 1005 条 GLM gold recalibration。

过滤结果：

- 写入 9 题、9 个问题族；
- 排除 1 个 recalibrated `retrieve_more`；
- 同一“灯塔计划”问题的两套证据绑定整组排除；
- query type：causality 4、mystery 2、relation 1、reveal 2；
- gold chain length：1 条×1、2 条×2、3 条×5、4 条×1。

输入 SHA256：

| 输入 | SHA256 |
| --- | --- |
| tasks | `febc02e8f13eeac9c77bbca9def619c450f4e7049c7b4ec90e4e6a6eb91fddd6` |
| preference val | `2128ea9076631a9518232e1915cd1ea2e77d300be1b1873181f4fbc1a31fdf96` |
| gold recalibration | `3a2da70987c1413895f3708cb79cc3d95a59d4d690414c3f1ff6689399f0e541` |

产物 SHA256：

| 产物 | SHA256 |
| --- | --- |
| strict manifest v1 | `67c825229e83d9755ae882aa4a1b698e1331e5f48b22053a959de8a1143e0f8e` |
| fact-group manifest v2 | `1c75c4b13481376a52d4aa407a5b25f599a820413f10a5884f337b1ae9e7854d` |
| runtime config | `ecbc088d91616b698a843342ca88fe79ff25da710c6d6616bf0bc752141405f2` |
| snapshots | `68b9e0cd12c254a0142c0870066706cf897cd98564cc42d92b5dc754704a0ff3` |
| retrieval metrics | `079d2bdc9c7d77a76e19b81af87c3693e44e3500e2f7809c7dee9d0905b7016f` |
| selection ablation v1 | `1f752a453cab07a4603cfb5d4bfc2ba6f46629d8742a8f235fb8c966e30cc406` |
| selection ablation v2 | `95f19e56aaae11a816b47b08f1436c3ab6b6398bcf7a658072052d11138be3c8` |
| snapshots v3 | `5fc3420576571d2a05b24263c8ced33af24b77ccaa1709001ebcc7ba1a857777` |
| retrieval metrics v3 | `8269231b55155ace7a02cb3c7c46e7b496d47c8b3511ed0ac5e094d68bad344d` |
| selection ablation v3 | `d1d582581eb5a46deb7f765bc536a7cbd2d3fe762a9aaa2344b7a35c6881bcc9` |

fact-group manifest v2 保留 34 条 `supported_facts`，其中 2 条事实有多引用。
将 v1 和 v2 投影到题目、family、query type、gold IDs、hypothesis 和 source
后逐行比较无差异，所以现有冻结 snapshot 仍可用于本轮诊断。

v3 snapshot 与 v1 snapshot 的每题 query 以及 dense、sparse、fusion、
rerank、prompt 文档 ID 顺序逐项一致；v3 只新增 fact group 和 chain member
审计元数据，没有改变检索结果。

## 3. 运行配置

实验在 241 的独立目录执行，未使用该机器根仓库的脏工作树：

`/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/experiments/answer_quality_20260907_aa2545f`

关键配置：

- dense 120；
- sparse 120；
- fusion 80；
- rerank 32；
- prompt 10；
- MiniRAG 开启；
- neighbor expansion 开启；
- GPU 0；
- set coverage、MMR、pyramid、pinning、CRAG 默认关闭；
- 不调用答案生成器。

snapshot 9/9 成功，0 failure。

## 4. 五阶段结果

| 阶段 | Recall@10 | 任一命中@10 | 全 gold@10 |
| --- | ---: | ---: | ---: |
| dense | 46.3% | 66.7% | 22.2% |
| sparse | 50.9% | 77.8% | 33.3% |
| fusion | 51.9% | 66.7% | 33.3% |
| rerank | 83.3% | 100% | 55.6% |
| prompt | 83.3% | 100% | 55.6% |

补充：

- rerank Recall@20 = 96.3%；
- rerank all-gold@20 = 88.9%；
- 完整 32 条 rerank 原池的 ID Recall 和 all-gold coverage 都是 100%。

本报告中的 stage `@k` 是对各阶段排序前 k 条的指标。完整 rerank 原池诊断
另行计算，不能把 rerank@10 与 32 条原池混为一谈。

## 5. Prompt-selection ablation

同一冻结 rerank 池：

| 策略 | ID Recall | 任一命中 | 全 gold |
| --- | ---: | ---: | ---: |
| top-score@10 | 83.3% | 100% | 55.6% |
| MMR@10 | 83.3% | 100% | 55.6% |
| coverage@10 | 83.3% | 100% | 55.6% |
| top-score@16 | 83.3% | 100% | 55.6% |
| MMR@16 | 83.3% | 100% | 55.6% |
| coverage@16 | 83.3% | 100% | 55.6% |
| oracle after common dedupe | 87.0% | 100% | 66.7% |

旧 ablation 的 oracle 也先执行公共去重，所以它不是 32 条 rerank 原池的
真正 oracle。v2 报告将公共池单独列出：

| 公共池 | ID Recall | 全 gold | 平均候选数 |
| --- | ---: | ---: | ---: |
| rerank 原池 | 100% | 100% | 32.0 |
| 公共去重后 | 87.0% | 66.7% | 25.3 |

v3 使用真实 `evidence_chain_member_ids` 后：

| 策略/池 | chain-visible Recall | chain-visible 全 gold |
| --- | ---: | ---: |
| 公共去重后 | 96.3% | 88.9% |
| top-score@10 | 92.6% | 77.8% |
| MMR@10 | 92.6% | 77.8% |
| coverage@10 | 92.6% | 77.8% |
| top-score/MMR/coverage@16 | 92.6% | 77.8% |
| oracle@16 after common dedupe | 96.3% | 88.9% |

选择归因：

- `dedupe_drop`：3/9；
- `selector_drop`：1/9；
- `selection_complete`：5/9。

## 6. Dedupe-drop 内容复核

| 问题 | 删除的 gold | 保留候选 | 内容状态 |
| --- | --- | --- | --- |
| 莫斯提马“普通信使” | `act5d0_st03#chunk-0010` | `act5d0_st03#chunk-0011` | 保留候选的 chain 完整包含删除 chunk，token recall 100% |
| 迷迭香与阿米娅 | `main_07-11_beg#chunk-0015` | `main_07-11_beg#chunk-0014` | 相邻重叠 chunk，删除文本 token recall 95.9% |
| 骸骨荒原的女妖 | `act47side_08_end#chunk-0023` | `act47side_06_beg#chunk-0019` | 保留候选的 chain 完整包含删除 chunk，token recall 100% |
| 骸骨荒原的女妖 | `act47side_08_beg#chunk-0028` | `act47side_06_beg#chunk-0019` | 保留候选的 chain 完整包含删除 chunk，token recall 100% |

原因是 `dedupe_prompt_evidence_candidates` 使用最终 prompt 文本做相似度判断；
有 `evidence_chain_text` 时优先比较 chain。多个不同原子文档可能携带同一个
chain，因此文档 ID 被去重不等于 chain member 内容被删除。

这不证明当前去重一定正确。它只证明现有 9 题报告不足以支持“去重导致回答
缺证据”的结论。若没有 chain member corpus IDs，纯文本近似也不能成为正式
coverage 指标。

## 7. 唯一 selector-drop 复核

问题：

> 从老天师对沉默的樵夫和诡异织物的态度，推断她为什么说“不想再亲手杀一个徒弟”？

未选中的 gold 是萌百背景文档，在 rerank 原池排名 24；top-16 不可能选中。
生产 prompt 已包含两个原始剧情 gold chunk：

- 樵夫称老天师为祖师、神志到极限、老天师说不想“再”亲手杀徒弟；
- 老天师自述杀过许多诡异织物，并直接消灭当前织物。

萌百文档支持的是“镇守北境多年、徒子徒孙很多、杀过许多怪物”的额外背景
事实。它不是当前 gold 中核心因果句的唯一来源。因此：

- ID-level `selector_drop` 是事实；
- “最终 prompt 不足以回答问题”尚未被证明；
- 为了把该文档塞进 prompt 而取消来源惩罚或默认提高 top-k，没有证据支持。

## 8. 对标签和评估器的修正

旧 manifest 将所有 fact 的所有 citation 拍平成 `gold_evidence_ids`，然后用
all-gold coverage 要求每个文档都出现。原始校准标签表明：

- 同一个 fact 可能同时引用两个高度重叠 chunk；
- 某些引用只支撑额外背景 fact，不一定是用户问题的最小充分答案；
- chain 中已经可见的 member 内容可能没有作为外层 prompt evidence ID 出现。

本轮工具修正：

1. `build_recalibrated_retrieval_manifest.py` v2 保留
   `gold_supported_facts`，记录每条 fact 对应的 corpus IDs 和原始 E-ID；
2. `build_retrieval_stage_snapshots.py` v2 透传并校验 fact citation group；
3. `evaluate_prompt_selection_ablation.py` v2 分开报告 rerank 原池、公共去重、
   selector，并解释 dedupe-drop 的保留文档和内容重合度。
4. v3 工具在 chain model scoring 时保存真实
   `evidence_chain_member_ids`，snapshot 和 selection ablation 可精确报告
   outer-ID coverage 与 chain-visible coverage。
5. strict9 v3 snapshot 已完成 9/9、0 failure，并验证所有阶段文档顺序与 v1
   完全一致。

这些都是离线评估改动，不修改生产默认。

尚未完成但必须补充：

1. 为 supported fact 标注 `answer_essential / supporting / redundant`，不能由
   citation 数自动猜测；
2. 在更大的无泄漏集上计算 fact coverage，并用实际生成验证它是否传导到
   complete/support。

## 9. 提升方案排序

下一步优先级：

1. 扩大 strict family-held-out manifest；9 题不足以比较策略。
2. 在 chain-member-aware 基础上加入 answer-essential fact 标注，继续修正
   fact-level 评估假阴性。
3. 对每题人工/双模型标注最小充分事实，再构造
   Full/One-edge-missing/None/Decoy-full。
4. 在冻结 snapshot 上运行答案生成器，报告用户可见 complete、support、
   unsupported、irrelevant 和动作，而不是只报 retrieval Recall。
5. 只有在内容/事实 coverage 确有缺口后，再比较 clean-text dedupe、identity-only
   dedupe、top-k、MMR 和 coverage。
6. 只有离线 coverage 改善能传导到独立 e2e complete/support，才考虑生产改动。

当前明确不应做：

- 默认开启 top-16 或 set coverage；
- 因 3 个 ID-level dedupe drop 直接取消语义去重；
- 为召回一个辅助萌百引用取消现有来源惩罚；
- 把 strict9 retrieval smoke 报成回答正确率；
- 在评估表示尚未校正前启动正式 RLVR。
