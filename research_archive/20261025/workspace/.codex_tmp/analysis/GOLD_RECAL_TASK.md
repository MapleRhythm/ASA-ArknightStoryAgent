# 任务：用 GLM 校准 Exx SFT 的 gold 证据绑定标签（治本的根节点）

你是 gpt-5.6-sol 子 agent。上一步已证明：gold 正例中约 48.4% 被 GLM 独立判为"不支持"（在"完整主张直接由该证据支持"的严标准下）。若直接拿这批带噪 gold 做 SFT/RLVR，是在教模型学错的绑定。本任务：用 GLM 对全部 gold (claim, cited-chunk) 绑定重判，产出干净标签集，并量化清洗前后分布。

## 关键前提（务必内化，别测出假结论）
- 证据绑定 = supported_facts 里每条 {fact, evidence_ids}。一条 fact 可引用 1-2 个 E-ID。
- 判据严格度必须与上一步 AGREEMENT 一致（同一个 GLM 提示语/标准），否则前后不可比。判"该证据是否直接、完整支持该主张"，支持/不支持 + 一句理由。
- 一条 fact 引用多个 E-ID 时：要判"这些被引证据合起来是否支持该主张"（多段合并判），也要记录每段单独是否支持——因为纠正时可能只需踢掉不支持的那个 E-ID，保留支持的。
- 不要只重判"疑似有问题"的，要**全量重判**，才能得到无偏的清洗后标签分布。

## 数据
- 源：`${REMOTE_STORE}/exx_grounding_v1/data/exx_grounding_v3_sft_minimal559_glm53_20260903/train.json`（910 行，其中 grounded_action_generation=783；answer_directly 行才有 supported_facts）。
- 每 answer_directly 行的 prompt 里有 `evidence:` [E1..En] 段，输出 supported_facts 有 fact + evidence_ids(gold)。可复用上一步的 evidence 解析逻辑（见 `${REMOTE_STORE}/exx_grounding_v1/tools/build_binding_hardneg_v2.py` 或 build_binding_hardneg.py 里的 parse_evidence）。
- val/test 同结构，也要一并校准（用于评测口径修正）。

## 你要做的
### 1) 全量 GLM 重判每条 gold fact-binding
- 对每条 (fact, 被引 chunk 合并文本) 调 GLM（glm-5.3, reasoning_effort=medium, max_tokens=4096, timeout 120, 并发≤3, 清空代理 env, source ~/.bigmodel_env, 429/1301 退避跳过并记录）。
- 若 fact 引多个 E-ID：额外对每个 E-ID 单独判一次（判"单独这段是否支持"），用于细粒度纠正。
- 输出三类标签：
  - `clean_support`：GLM 判合并证据支持 → 保留。
  - `partial`：合并判支持但某个单独 E-ID 不支持 → 建议踢掉不支持的 E-ID（记录该踢哪个）。
  - `unsupported`：合并证据都不支持该 fact → 该 fact 需降级（见下）。

### 2) 产出干净标签集（多个版本，供对比）
- `gold_recalibrated_v1_strict.jsonl`：只保留 clean_support + partial(修正后的 E-ID)；unsupported 的 fact 从该题的 supported_facts 里移除。若移除后该题 supported_facts 为空 → 该题动作应从 answer_directly 改为 abstain/retrieve_more（记录并统计有多少题会翻转）。
- 保留原始行 id/question_key 对齐，方便重训。
- 同时输出一个"变更清单"jsonl：每条被修改/移除的 fact，原 E-ID、GLM 判决、理由。

### 3) 量化报告（写 stdout + .report.json）
- 清洗前后：fact 总数、clean/partial/unsupported 各占比。
- 有多少题因移除 unsupported fact 而 supported_facts 变空 → 动作需翻转（这是重要数字，说明有多少"看似回答其实证据不足"）。
- 按 query_type 分层 unsupported 率（对齐上一步：fact/relation 应偏高）。
- partial（多 E-ID 里有冗余引用）的比例——量化"引用了多余证据"的普遍程度。
- 抽 10 个 unsupported 例子附 GLM 理由，人工可读校验。

## 判据与产物
- 若 unsupported 率 ≈ 上一步的 48.4% 附近 → 交叉验证一致，清洗可信。
- 若差异大 → 说明本任务的 GLM 提示语/严格度和上一步不一致，先对齐再重跑。
- 结论写 `${ASA_ROOT}/.codex_tmp/analysis/GOLD_RECAL_FINDINGS.md`；数据产物存 248 `${REMOTE_STORE}/exx_grounding_v1/data/gold_recalibrated_20260904/`。

先跑 30 fact 的 smoke 打通 + 核对严格度与上一步一致（smoke 的 unsupported 率应落在 40-55%），再全量。全量 GLM 调用量约 1700+ fact，注意用并发 + 断点续跑（输出追加、按 fact 唯一 id 去重）。
