# ASA 创新点研究

主入口：[研究报告](RESEARCH_REPORT.md)；下一阶段：[pilot 规范](PILOT_PROTOCOL.md)。

首推“面向问题目标、保留归因作用域的替代证明后训练”。这是待验证候选，不是已确认首创或模型增益。

本轮：

- 补齐 12 条旧检索输出的全部本地查询构造分支；
- 构造 84 条隔离语义探针，调用两种底层 validator；
- 15 项新测试及前轮 21 项测试通过；
- 保存官方文献资料，撤回并隔离未核实的近期论文断言；
- 零新模型调用、零真实检索、零训练；未修改生产管线，未占用远端 GPU。

重要纠正：“郑愁、渡魂剑”虽然退出独立 follow-up 分支，仍保留在综合查询；不能认定它们从整个系统消失。
84 条探针不是问答正确率评测，且全部排除训练。

从仓库根目录执行：

```bash
python3 experiments/innovation_discovery_20260910/tools/probe_project_mechanisms.py
python3 -m unittest discover -s experiments/innovation_discovery_20260910/tests -v
python3 experiments/innovation_discovery_20260910/tools/verify_research_artifacts.py
```

可选文献重查，必须使用新的 run-tag，保留失败历史：

```bash
python3 experiments/innovation_discovery_20260910/tools/collect_primary_literature.py \
  --group focused --run-tag fresh_literature_attempt
```

脚本读取当前工作树实现，因此更新生产源码后回放结果可能变化；先保存旧批次，比较 `RUN_MANIFEST.json` 源码哈希。
下载脚本收到 HTTP 200 但传输超时仍判 `unverified`；部分 PDF 不作为全文阅读依据。

目录：

- `data/`：隔离探针，不是训练 gold；
- `audits/`：完整查询回放、底层检查逐项结果、最终验收；
- `sources/`：出版方资料、下载哈希、阅读范围和未核实线索；
- `tools/`、`tests/`：确定性回放与检查；
- `RUN_MANIFEST.json`：本次回放的源码和运行范围。
