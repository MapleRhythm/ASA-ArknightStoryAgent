# ASA continuation status (2026-09-06)

## 本轮结论

- `relevance-aware SFT` 的主要收益仍是结构行为：79 条固定集上
  `schema_valid` 从 78.48% 提升到 89.87%，重复 fact 从 17.72% 降到
  8.86%，截断从 2.53% 降到 0%。
- 事实绑定没有得到证据支持：校准 gold 重测后，`exact E-ID set`
  29.03% -> 27.42%，平均 claim-local binding 0.616 -> 0.595，
  `claim-citation` 仍低于生产门槛。因此该 adapter 不切生产，也不以此
  结果启动 RLVR。
- GLM 集合审计 smoke 使用了两个协议。`fast` 协议因 1536 token/关闭思考，
  多次出现证据编号或 fact 数量协议错误；`rich` 协议提高到 4096 token/
  高思考后，仍有非 JSON 响应，但有效样本显示 clean-SFT 为 1/1 complete，
  relevance-aware 为 2/3 complete、1/3 partial。样本量不足以声称质量提升。
- 审计请求最初失败的原因是 241 的代理变量指向失效的
  `127.0.0.1:8888`，清除代理后 GLM 请求可以到达。

## 运行入口问题

241 的 `release/ASA-ArknightStoryAgent` evidence-id 配置原先引用不存在的
`model/lora/asa-arknightstoryagent-4b-lora`。实际可用的 clean Exx adapter 位于
挂载盘 `relevance_data_20260906/models/exx_binding_clean_sft_v2...`。
临时修正配置已另存为：

```text
/mnt/wwn-0x5000cca295f3594f/zhb_asa/relevance_data_20260906/runtime_gpu_exx_clean_mount_20260906.json
```

即使修正 LoRA 路径，vLLM 首次启动仍在加载模型/检索器阶段耗时较长；不要把
初始化耗时当成回答延时。主仓库新增了 vLLM 生成资产 preflight，会在加载大型
reranker 和 embedding 之前对基座与 LoRA 路径做存在性检查，避免配置错误表现为
长时间假死。

## 证据预算诊断

从 79 条固定输入的实际 `[E#]` 顺序统计，62 条 answer 行共有 128 个 gold
evidence IDs：

- top-10 能覆盖全部 gold IDs 的题目为 59/62（95.16%）；
- top-16 能覆盖全部 gold IDs 的题目为 62/62（100%）；
- gold evidence 的最大出现位置为 E12。

这说明当前 top-10 存在可量化的 prompt coverage 缺口。该缺口不是 reranker
候选召回失败，而是候选已经存在、在进入回答 prompt 时被过早裁掉。新增的
`runtime_gpu_reranker_qwen35_4b_evidence_id_coverage_ablation.json` 将 top-k
提高到 16，并启用集合覆盖选择；它是独立 ablation，尚未替换默认配置。

## 已提交

- `468dba5`：训练选择与生产验收门槛报告
- `9a67208`：vLLM 生成资产 fail-fast preflight

分支：`fix/grounding-pool-chain-priority`，已推送 GitHub。

## 下一步门槛

1. 用同一份 rich 审计协议完成 family-held-out 的差异样本审计，并保存原始
   GLM 响应，区分“审计协议失败”和“事实不支持”。
2. 解决 241 运行入口的冷启动/模型加载问题后，再做 Exx/evidence-id 盲测；
   先以 `--no-reranker` 验证协议，再恢复 reranker 验证完整链路。
3. 只有在校准 gold、独立语义审计和新盲测三者一致显示
   `unsupported/contradicted <= 20%`、claim support >= 0.80、
   稳态 P95 <= 20s 时，才考虑生产切换或 RLVR。
