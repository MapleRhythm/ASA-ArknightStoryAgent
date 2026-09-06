# SFT / RLVR 模型选择复核（2026-09-06）

## 已有 79 条固定集指标

| 模型 | Schema valid | Action accuracy | Duplicate fact | Exact evidence set | Claim-citation alignment |
|---|---:|---:|---:|---:|---:|
| v3 stratified SFT | 78.5% | 78.5% | 10.1% | 29.9% | 0.183 |
| factref SFT | 63.3% | 78.5% | 24.1% | 20.9% | 0.164 |
| factref RLVR | 64.6% | 81.0% | 17.7% | 19.4% | 0.164 |
| GLM precision structural RLVR smoke | 65.8% | 82.3% | 20.3% | 17.9% | 0.161 |

原始指标位于 241 挂载盘：

```text
/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/eval/
```

## 结论

1. 旧 RLVR 奖励主要提升了 `next_action`，没有提升证据集合和 claim-level 绑定，甚至有下降。
2. 当前不能用 action accuracy 或 schema valid 作为生产选择标准；它们与事实正确率并不等价。
3. 下一版训练必须先切换为 Exx/evidence-id 目标，并以每个 fact 的语义支持为主奖励。
4. 只有在 Exx 绑定指标提升后，才适合加入 KTO/GRPO/RLVR；否则 RLVR 会继续放大“格式正确但事实绑定错误”。

## 下一轮训练验收门槛

- claim-level semantic support ≥ 0.80；
- unsupported/contradicted fact rate ≤ 0.20；
- duplicate fact ≤ 0.10；
- invalid evidence-id = 0；
- fixed 79 + novel blind set 均通过；
- 单题稳态端到端延迟 ≤ 20s。
