# Qwen0.6B 检索旁路端到端基线（2026-09-06）

## 配置

- 241，单卡 RTX 4090；
- 生成模型：Qwen3.5-4B + 既有 LoRA；
- 检索 embedding：Qwen3-Embedding-0.6B 1024 维旁路索引；
- BGE reranker；
- MiniRAG 关闭；
- 结论证据：10 条、单条最多 1000 字、总上限 8000 字；
- 同一组 8 条盲测题，未覆盖旧结果；
- 运行目录：

```text
/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/runs/e2e_full_evidence_codex_20260906/qwen06_quote_intended/
```

## 结果

- 8 条完成；
- 平均耗时 22.69s（首题冷启动 71.60s，后 7 题平均 15.70s）；
- 1 条明确正确（霜星）；
- 1 条基本可用但不完整（雷德，第二子问部分正确）；
- 多条出现人物身份、动机或照顾行为的无证据拼接；
- 1 条正确触发弃答（凯尔希人名/日期未写明）。

## 主要错误

1. 证据覆盖不足时，模型仍把相邻剧情拼接成完整答案，例如房东身份、古米受照顾方式。
2. 多子问没有逐项槽位约束，第一子问的证据会挤占第二子问。
3. `quote` 协议会让格式问题和证据缺失混在一起；即使证据存在，省略号或长 quote 也可能触发错误重检索。
4. 结论生成仍允许模型自行补全人物关系和因果，词法 grounding 只能发现部分越界，不能证明蕴含。
5. 入口若传入不存在的 runtime 配置会静默使用默认值，已在新分支改为直接报错；本次运行因此显式指定了全部关键参数。

## 决策

- Qwen0.6B 旁路索引继续保留，不能单凭 Recall 提升切生产；
- 先完成 Exx/evidence-id 协议在 241 主仓库的可运行迁移，再测“只输出 E-ID + 程序重建引用”；
- 训练侧优先补多子问覆盖、否定性证据和身份/因果 hard negatives；
- 生产门槛仍为事实正确率与幻觉率，而非 schema 或 Recall 单项指标。

## 原始产物

```text
/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/runs/e2e_full_evidence_codex_20260906/qwen06_quote_intended/results.jsonl
/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/runs/e2e_full_evidence_codex_20260906/qwen06_quote_intended/run.log
```
