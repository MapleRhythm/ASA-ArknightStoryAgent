# Exx RLVR structural smoke evaluation — 2026-09-03

本次实验在 241 的 GPU0 上完成，旧模型、旧数据和旧评测结果均未覆盖。

## 评测产物

- v1 结构 profile adapter：
  `/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/models/exx_grounding_v1_grpo_rlvr_glm_precision_structural_smoke64_s16_20260902/`
- v1 79 条评测：
  `/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/eval/fixed_evidence_rlvr_glm_precision_structural_smoke64_s16_val79_20260903/`
- v2 结构 profile adapter：
  `/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/models/exx_grounding_v1_grpo_rlvr_glm_precision_structural_v2_smoke64_s16_20260903/`
- v2 79 条评测：
  `/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/eval/fixed_evidence_rlvr_glm_precision_structural_v2_smoke64_s16_val79_20260903/`

## 指标对比

| 模型 | strict JSON | schema valid | action | schema+action | 重复事实 | 非法 E-ID | evidence Jaccard | fact similarity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 RLVR-512 | 91.14% | 79.75% | 81.01% | 69.62% | — | 3.80% | 45.47% | 30.93% |
| 旧 RLVR-384 | 92.41% | **88.61%** | 75.95% | **72.15%** | — | — | 48.83% | **32.40%** |
| GLM smoke | 93.67% | 81.01% | 81.01% | 68.35% | — | 3.80% | **49.29%** | 31.38% |
| structural v1 | **94.94%** | 60.76% | **83.54%** | 50.63% | 24.05% | 3.80% | 49.11% | 30.97% |
| structural v2 | 92.41% | 65.82% | 82.28% | 55.70% | **20.25%** | **2.53%** | 46.70% | 30.01% |

结论：v2 相对 v1 确实减少了重复事实（-3.80 个百分点）和非法 E-ID（-1.27 个百分点），但没有同时提升 schema+action 和证据覆盖；相对 GLM smoke，schema+action 仍低 12.66 个百分点，证据覆盖低 2.59 个百分点。因此不启动 783 条全量训练。

## 根因判断

1. v1 的 `near_duplicate_fact_penalty` 和 `concise_fact_penalty` 被 `protocol_gated()` 包裹。重复事实会先让 payload 失效，针对重复/超量事实的惩罚随后被置零，模型仍可能获得语义奖励。
2. v2 已移除这两个 gate，训练日志中对应惩罚出现非零值，说明修复生效；但 16 step smoke 的 rollout 组仍有同质化输出，部分组 reward std 为零，策略没有足够相对偏好信号。
3. 当前 SFT 训练集的 555 个 `answer_directly` 样本中，事实数分布为 1:54、2:97、3:109、4:80、5:69、6:69、7:45、8:32。虽然标签本身没有重复事实，但“允许并经常输出 5–8 条事实”的先验会促使 4B 模型填充列表；在验证集上表现为 8 条中后半段重复。
4. 训练使用的旧 `train.json` prompt 仍是旧版“1–8 条事实”协议，新增的去重/通常 1–4 条规则没有进入 SFT 监督目标。仅靠短 RL smoke 不足以覆盖这个分布偏差。

## 下一步方案

1. 先构建 canonical SFT 数据：保留所有必要证据 ID，只去掉重复/近重复事实；对非多事实问题将目标压到最小充分事实集合，不能为凑数量复制事实。
2. 用 canonical SFT adapter 做小规模验证，再接 v2 RL；RL 使用更高 rollout 多样性（例如 `num_generations=8`、temperature 1.0、至少 32 steps），并继续保留未 gate 的 anti-padding 惩罚。
3. 评测同时报告 raw-output 和 runtime sanitizer 后指标。运行时已有 `compact_supported_facts_payload` 去重，但不能用后处理掩盖训练协议问题。
4. 在 canonical SFT + v2 RL 的 79 条验证中，只有当 `schema+action`、evidence Jaccard 和 fact similarity 至少不低于当前最佳基线，才考虑 783 条全量训练。

