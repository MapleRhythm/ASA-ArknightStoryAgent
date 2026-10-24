# Phase 2：深入审计与隔离对照

日期：2026-09-09。主入口：[研究报告](REPORT_20260909.md)。

状态：

- 旧 56 条模型输出已做内容绑定检查；旧语义总分不再作为可靠指标。
- 12 条旧检索输出已通过当前本地 normalizer/merge/query planner 回放；未实际检索。
- 124 条自然控制＋512 条合成控制已生成，尚未调用模型。
- 21 项测试通过；无新训练、无生产代码改动。
- unknown 与所有控制样本都排除训练用途。

从仓库根目录执行：

```bash
python3 experiments/paper_idea1_alternative_proof_20260909/phase2_deep_audit/tools/research_controls.py
python3 experiments/paper_idea1_alternative_proof_20260909/phase2_deep_audit/tools/replay_query_processing.py
python3 -m unittest discover -s experiments/paper_idea1_alternative_proof_20260909/phase2_deep_audit/tests -v
```

可选文献 HTTP 核验（会重新获取页面，时间戳和页面哈希可能变化）：

```bash
python3 experiments/paper_idea1_alternative_proof_20260909/phase2_deep_audit/tools/verify_bibliography.py
```

`research_controls.py` 只重建本子目录中的派生 JSON，不覆盖上一轮原始数据、输出或语义审阅。
预测的 prompt/system、原文内容和 reference 各自绑定哈希；自然语义复核仍是 Codex 审阅，不是人类 gold。

目录：

- `audit_spec.json`：四题原子事实、标签歧义和旧审阅勘误。
- `audits/`：旧输出检查、原文重叠、查询回放、资源记录与文献响应。
- `data/`：推理输入、配对索引、合成真值表。图和标签仅在 diagnostic 中。
- `tools/`：只读分析与派生数据构造器。
- `tests/`：控制因素、内容绑定、图逻辑、查询信息丢失的测试。
- `CONTROL_MANIFEST.json`：规模、哈希与 `generation_status: not_run`。

下一轮 GPU 推理必须保持 canonical `enable_thinking=False`，不得把 diagnostic/reference 输入 reader，
不得因为显存有空余就未经授权与既有任务共享 GPU。需要记录实际 token 长度，禁止静默截断；
结果必须写入新的 run 目录，不能复用旧预测假装是新对照结果。
