# 证据绑定验证器 + RLVR 奖励设计（连接 agentic RL）

日期：2026-09-04

## 一个工件,三处用途
核心是一个 **binding verifier**：输入 (claim, chunk) → P(chunk 支持 claim) ∈ [0,1]。
- 训练数据：hard-neg v2（codex 在建）。正例=(claim, gold chunk);负例=(claim, GLM验证过的"相关但不支持"chunk)。
- 模型：微调现有 cross-encoder reranker(bge-reranker-v2-m3)为 claim-chunk 蕴含打分器(pointwise 二分类),架构现成。

三处用途(同一验证器):
1. **服务端引用过滤**(零重训即时收益):生成器出 fact→E-ID 后,逐条跑验证器,低分的 fact 丢弃/标记 → 立刻抬 claim-citation alignment。
2. **对比 SFT**:把"fact 绑到 hard-neg"作为负例进 SFT,教生成器只引用真支持的证据。
3. **grounding-RLVR 奖励**:reward = 模型各 fact→E-ID 绑定的验证器均分 − 引用 hard-neg 的罚分。**这就是让 grounding-RLVR 值得做的可验证奖励**(v3 已把格式/action 打满 98.7/82.3,格式奖励饱和;绑定奖励才是下一个有信号的)。

## 为什么这是 agentic RL 的关键一环(回答用户)
- hard-neg 数据集 → SFT(监督,能力注入)
- 同一数据 → binding verifier → **RLVR 的奖励函数**
- 用户直觉对:证据绑定验证器 = grounding-RLVR 的奖励核心。现在建它是双用途(先 SFT 后 RLVR)。

## 与其它线的关系
- 绑定质量(claim-citation)、retrieve_more 判断率(28.7%)、充分性探针(AUROC 0.5)是**同一上游能力**:分不清"相关 vs 支持"。binding verifier 直接学这个区分 → 也能反哺 retrieve_more(充分性=所有必需 claim 都能被支持)。
- 因此 binding verifier 和 B1B2 的 coverage judge 是互补的两个可验证奖励:一个判"这条证据支不支持这条主张"(绑定/生成侧),一个判"证据集合覆不覆盖 gold 链"(检索/多轮侧)。

## 去风险闸门(建完 v2 数据后先做,不直接上 RL)
仿 B1B2 的做法:用 GLM 作验证器,对 val79 现有预测做服务端引用过滤,量化 claim-citation alignment 能从 0.209 抬到多少 → 证明奖励有信号,再训小验证器 + 上 RLVR。

## 执行顺序
1. [codex 进行中] hard-neg v2 数据 + GLM 验证质量报告 + 疑似漏标正例数(校准 29.85% 里有多少是标注多解假阴性)。
2. 去风险:GLM 作验证器,val79 引用过滤,量化 alignment 提升上限。
3. 训 binding verifier(微调 reranker,pointwise)。
4. 三处接入:服务端过滤(即时)→ 对比 SFT → grounding-RLVR 奖励。

## v2 hard-neg 数据结果（codex，2026-09-04）
- 1706 fact-row，6718 dense top-4 候选；hard-neg 与 claim cos 均值 0.560（v1 trigram 0.0076，74×）。
- GLM 验证前 300 行/1130 候选：1005 unsupported（干净负例），125 supported→移入 suspected_missed_positives。
- **关键校准**：125 个漏标正例分布在 98/300 行(32.67%) → exact_evidence_set=29.85% 系统性低估 4B 绑定能力（约 1/3 题引用了另一段同样支持的证据被判错）。
- 建议评测口径改：从"引用集==gold 集"改为"每条引用是否 GLM/verifier-支持"。
- 训练用法：1005 confirmed-unsupported 优先，5588 not_run 作 provisional，125 missed-positive 单独做多解性分析（不当负例）。

## 待定实验（进行中）
probe_4b_binding.py：4B zero-shot 在 (claim,chunk)→支持/不支持 上的 acc/prec/rec/spec + 批量吞吐 ms/pair。
判据：zero-shot acc>75% → 能力在，走 D1 分解+hard-neg 训练；很差 → 真容量瓶颈，验证器用更大模型/GLM 蒸馏。
