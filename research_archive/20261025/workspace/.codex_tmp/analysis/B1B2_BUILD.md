# B1+B2 构建：gap 驱动的多轮检索奖励 + RL

日期：2026-09-04

## 目标
把 follow-up 策略从"复述查询"训成"针对缺失链环检索"。
- B1 溯因对比：奖励召回**新** gold 段（novelty），罚与前轮文档重叠。
- B2 覆盖过程奖励：充分性判官给每轮 retrieve_more 一个 **gold 链覆盖增量**的密集过程奖励。

## codex 确认的机制证据（支撑本方案）
- follow-up hypothesis 的 question 文本每轮完全不变（46/46, 35/35）。
- round2→3 top-5 文档 29/35 完全相同；关键词只追加泛化同义词，不指向缺失槽位。
- round-only recall@10：0.68→0.68→0.52（后轮退化）。
- 命中 40 个里 round1=37。
→ 结论：奖励必须显式激励"新证据邻域 + 缺失链环覆盖"，否则策略停在同一语义邻域。

## 架构决策
- 检索 + RL 全部在 248（A100 + 0.6B/4B 索引镜像 + hybrid retriever 都在本地），无跨机 RPC。
- 复用：GRPO 脚手架（train_exx_grpo_rlvr.py 已跑通单轮）、E段覆盖判据（sufficiency_probe）、GLM 分级/gold 链。

## 奖励核心（retrieval_reward.py，已上传 248）
score_followup(followup_query, gold_segments, prior_pool):
  new_pool = retrieve(followup_query)
  coverage_gain = |gold covered by prior∪new| - |gold covered by prior|   # B2
  novelty = 1 - |new∩prior|/|new|                                          # B1
  reward = 1.0*coverage_gain + 0.3*novelty

## 构建阶段
1. [进行中] 奖励引擎 + 离线验证：证明 reward 能区分"复述 follow-up(冗余)"vs"oracle 缺口 follow-up"。这是上 GPU 前的去风险闸门。
2. gap 条件 follow-up SFT 暖启：用 gold 链合成"缺口查询"目标（删一段→缺口已知，兼 C3 反事实），给策略好的初始化。
3. GRPO 多轮：策略生成 follow-up，reward=上面的 coverage+novelty，加已见文档 novelty penalty。
4. 评测：多轮 cumulative recall@k、round-only recall 退化是否消失、top-k overlap 是否下降。

## 关键设计原则
- 奖励可验证、难 hack：coverage 用 gold 链的 E段子序列覆盖（客观），novelty 用文档 id 重叠（客观）。
- 防 reward hacking：novelty 单独高但 coverage=0（乱查新文档骗 novelty）要限权重；主奖励是 coverage。

## 阶段1 结果：奖励验证通过（2026-09-04）
hard59 前30题，26 题在 round1 后有 gap：
- 复述 follow-up（当前模型行为）：coverage_gain=0，novelty=0，0/26 有改善 ← 确认冗余无用
- oracle 缺口 follow-up（查询=缺失段文本）：coverage_gain=0.923，novelty=0.821，**16/26 召回到新 gold**

结论：
1. 奖励有强区分度（0 vs 0.923），RL 有明确可优化梯度。
2. 62%(16/26) 的 gap 可由更好的 follow-up 找回 → RL 的可达上限真实存在。
3. 剩余 10/26 即使 oracle 也找不回 → 真·召回天花板/corpus 问题，不归 RL 管（对齐 A2 叙事命题索引那条线）。
去风险闸门通过，可以投 GPU 做 RL。
