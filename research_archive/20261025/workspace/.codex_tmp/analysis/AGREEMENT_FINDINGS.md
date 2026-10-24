# 4B 本地验证器 vs GLM-5.3 独立判决

评估日期：2026-09-04（Asia/Shanghai）

## 结论先行

在同一批 500 个 claim-chunk pair 上，base Qwen3.5-4B zero-shot 与独立 GLM-5.3 的一致率为 **91.2%**，Cohen's κ 为 **0.781**，4B 支持分数对 GLM 二值判决的 **AUC=0.937**。以 GLM 为 ground truth 的混淆矩阵为 TP=117、FP=29、TN=339、FN=15，即 precision=80.1%、recall=88.6%、specificity=92.1%。

因此，本地 4B **已经达到作为 RLVR 奖励的量化门槛**（agreement≥85%、AUC≥0.9，κ 处于可用水平），RLVR 训练阶段不需要让 GLM 在环。推荐先用 4B 的连续支持分数做奖励；若要用于生产级高精度证据绑定，建议再用新鲜 GLM 标签对 4B 做 LoRA/监督微调，重点修正主语归属、身份、因果和多子句“完整支持”判断。微调是稳健性建议，不是达到本次门槛的前置条件。

## 数据与协议

- 输入：`${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v2_20260904.jsonl`
- 原始数据有 1,706 个 fact-row；同时存在一条正例和一条 `glm_verdict=unsupported` hard-negative 的可配对行有 **295** 条。
- 全量抽取 250 个 fact-row，每行 1 个 positive chunk + 1 个 hard-negative，共 **500 pair**，正负各 250。smoke 为 20 行/40 pair。
- 为保证两个裁判看到的输入完全相同，chunk 统一截断到 600 字符；全量 184/500 pair 因原文更长而被截断。query_type 分布（全量 pair）：causality 276、fact 86、reasoning 96、relation 34、reveal 8；没有可配对的 mystery 行。
- 抽样使用固定 seed `20260904`，选择和 pair manifest 可复现。

GLM：

- 每个 pair 单独新调一次 `glm-5.3`，`reasoning_effort=medium`、`max_tokens=4096`、timeout=120 s、并发 3。
- 调用前显式清除 `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/all_proxy`，并使用无代理 HTTPS opener；key 来自 `source ~/.bigmodel_env`。
- 全量 500/500 成功，**0 skip（1301 内容过滤）、0 error**。

4B：

- 模型：`${REMOTE_STORE}/ASA-ArknightStoryAgent/model/qwen3.5-4b`，bf16，`max_model_len=2048`，`gpu_memory_utilization=0.85`，`enforce_eager=True`，`CUDA_VISIBLE_DEVICES=0`，`VLLM_USE_FLASHINFER_SAMPLER=0`。
- vLLM 脚本包含 `if __name__ == "__main__":` guard，避免 spawn 递归崩溃。
- 判决为 zero-shot 生成的“支持/不支持”；连续分数为首 token 的
  `sigmoid(logP(支持) - logP(不支持))`。支持 token id=97273，不支持 token id=115881；500/500 都成功取得两者 logprob，因此 AUC 不是二值标签的伪分数。

## Smoke（40 pair）

| 指标 | 数值 |
|---|---:|
| usable pair | 40 |
| agreement / accuracy | 90.0% |
| Cohen's κ | 0.772 |
| AUC | 0.970 |
| precision / recall / specificity | 84.6% / 84.6% / 92.6% |
| confusion (TP, FP, TN, FN) | (11, 2, 25, 2) |

GLM 将 smoke gold-positive 判支持 13/20（65.0%），判不支持 7/20；hard-negative 判不支持 20/20（100%）。

## 全量（500 pair）

| 指标 | 数值 |
|---|---:|
| usable pair | 500 |
| agreement / accuracy | **91.2%** |
| Cohen's κ | **0.781** |
| AUC | **0.937** |
| precision / recall / specificity | **80.1% / 88.6% / 92.1%** |
| confusion (TP, FP, TN, FN) | **(117, 29, 339, 15)** |
| GLM 支持率 / 4B 支持率 | 26.4% / 29.2% |

Smoke 到全量的 agreement 为 90.0%→91.2%，AUC 为 0.970→0.937，方向一致，没有出现只在 smoke 上成立的假高一致。

## 分层结果

### 按 query_type（正负侧合并）

| query_type | n | agreement | κ | AUC |
|---|---:|---:|---:|---:|
| causality | 276 | 93.1% | 0.828 | 0.949 |
| fact | 86 | 86.0% | 0.646 | 0.905 |
| reasoning | 96 | 91.7% | 0.802 | 0.962 |
| relation | 34 | 88.2% | 0.642 | 0.813 |
| reveal | 8 | 87.5% | 0.750 | 0.933 |

`relation` 的 AUC=0.813 低于总体门槛，但 n=34；`reveal` 只有 8 个 pair，不足以做稳定结论。总体主要短板不是 causality，而是 fact/relation 中的实体身份、归属和关系完整性。

### 按正例/负例侧

| 侧别 | n | agreement | κ | AUC | GLM 支持率 |
|---|---:|---:|---:|---:|---:|
| positive | 250 | **85.6%** | 0.711 | 0.900 | 51.6% |
| negative | 250 | **96.8%** | 0.321 | 0.652 | 1.2% |

负例侧的 κ 被极端类别不平衡压低：GLM 在 250 个 hard-negative 中只支持 3 个；agreement 和 specificity 才是这一层更直观的指标。负例侧 AUC=0.652 也受仅 3 个正标签影响，不应替代总体 AUC。

### query_type × 侧别 agreement

| query_type | positive n / agreement | negative n / agreement |
|---|---:|---:|
| causality | 138 / 89.9% | 138 / 96.4% |
| fact | 43 / **76.7%** | 43 / 95.3% |
| reasoning | 48 / 83.3% | 48 / 100.0% |
| relation | 17 / 82.4% | 17 / 94.1% |
| reveal | 4 / 75.0% | 4 / 100.0% |

44 个全量不一致中，36 个来自 positive 侧（81.8%），8 个来自 negative 侧。正例侧的系统性偏离是“4B 把部分不完整主张判成支持”，以及少量长上下文中的漏判；负例侧整体很稳。

## GLM 与 gold 先验的对照

这项对照说明为什么不能把数据集原有 positive 标签直接当独立 ground truth：

| gold 先验侧 | GLM 支持 | GLM 不支持 |
|---|---:|---:|
| positive（n=250） | 129（51.6%） | **121（48.4%）** |
| hard-negative（n=250） | 3（1.2%） | **247（98.8%）** |

GLM-vs-gold 先验总体一致率只有 **75.2%**。hard-negative 的确认率 98.8%，说明负例构造很干净；但将近一半的 gold-positive 在 GLM 的严格“完整主张直接支持”标准下不成立，主要是主张包含证据未明确给出的主体、身份、动机、因果或第二个并列子断言。

## 错误模式

抽查不一致样例后，GLM 拒绝而 4B 支持的典型模式包括：

- 证据出现事件，但没有明确说话者/主体是谁（例如把未标名发言归给某角色）。
- 证据支持一半主张，却缺失另一个并列子断言（例如“做了 X，并且出于 Y”只明确了 X）。
- 证据与角色/团体相关，但没有直接证明“属于某家族/团体”或“某人提供/指挥”。

少量反向错例是长文本中的明确支持被 4B 漏判。由此，后续补标/微调应优先覆盖 `fact-positive`、`relation-positive` 以及“身份归属 + 多子句完整性”样本，而不是继续堆普通 hard-negative。

## RLVR 使用建议

1. **立即可用（RLVR pilot）**：zero-shot 4B，奖励使用连续 logprob margin/支持分数，而不是只用字符串 0/1。总体 AUC=0.937，适合排序和软奖励；可在开发集上再选择阈值或设置低置信度 abstain band。
2. **推荐的稳定形态（production）**：用本次 GLM 独立标签及后续补标的主体/关系难例，对 4B 做 LoRA/SFT verifier 微调。目标是减少正例侧的过支持和长上下文漏判，提高 fact/relation 层，而不是为了跨过本次门槛。
3. **暂不需要 reranker**：当前任务是二分类 pairwise evidence binding，4B 已有可用连续分数；只有在需要更复杂的多候选排序、阈值校准或长文档重排时，才值得引入单独 reranker。
4. **GLM 是否需要在环**：不需要作为 RLVR 每步奖励的在线依赖；保留 GLM 用于离线抽样审计、难例补标和周期性漂移监控。

## 远端原始产物

目录：`${REMOTE_STORE}/exx_grounding_v1/eval/verifier_glm_agreement_20260904/`

- `pairs_smoke.jsonl` / `pairs_full.jsonl` 及各自 `.meta.json`
- `glm_smoke.jsonl` / `glm_full.jsonl`
- `fourb_smoke.jsonl` / `fourb_full.jsonl`
- `metrics_smoke.json` / `metrics_full.json`
- 对应 `*.log` 与 `metrics_*.txt`

全量三方 JSONL 行数均为 500，smoke 三方 JSONL 行数均为 40；metrics 的 `excluded` 为空。
