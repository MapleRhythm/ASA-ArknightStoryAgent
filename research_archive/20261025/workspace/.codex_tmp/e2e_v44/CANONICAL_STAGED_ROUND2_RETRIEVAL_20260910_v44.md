# v44：保持首轮状态的第二轮检索续跑

规范目录：`outputs/experiments/staged_round2_retrieval_20260910.v44`。
日期按北京时间 2026-09-10；机器回执使用 UTC 2026-09-09。

## 结果与边界

24 个 v41 首轮状态全部经过真实 canonical 方法重放；结论、trace、
证据及调用次数逐字段一致。使用原有 append/apply/prepare_next_round
继续状态机，不重抽第一轮，不手写后续 hypothesis，不额外调用 planner。

- 10 个已结束答案原样带过，包括已知错误答案；检索调用为零。
- 14 个未结束状态的真实第二轮检索全部成功。
- 新增 LLM 文本生成为零。该阶段不测量最终回答正确率。
- 完整 state、实际 dense/sparse 查询、prepared 对象、真实 12000 字符预算下
  的计划第二轮 prompt 已保存并通过 CPU 回放。
- 这是一组既有开发输出的 staged continuation；24 个状态不是 24 道独立题，
  也不是全新独立 E2E benchmark。

各未结束状态第二轮 query 数：010 为 15，017 为 13，023 为 15，
046 为 15；049 对照为 14、候选为 15。每个 query 各调用一次 dense/sparse。
查询数仍有混杂，不可把后续差异完全归因于姓名恢复。

## 验证和资源

prepare、run、check、close 全部 exit 0；`checkpoint_v44.json` 已归档。
preflight SHA：
`c6389a800db29516b9c792770784fa5907056dbab7ae2e80e2c0cfe2c63b0775`。

GPU2 在原锁内进行了新的 13 次、超过 60 秒连续空闲检查；
检索后已释放。close 回执记录 GPU2 无计算进程、15 MiB、利用率 0；
GPU0 的其他用户进程未干预。本次 run 日志无 ERROR/Traceback 行。

## 决策

仅完成检索/状态/提示协议验收，不宣称语义源文复核完成，不推广生产。
下一阶段调用原 canonical 第二轮 reader，保留其原有最终 direct fallback，
分别记录两次可能的请求和原始输出，再核对最终展示与证据。

MiniRAG 与 web 仍禁用；未改变模型、预算、校验器、E-ID 绑定、生产源码或
默认配置；未训练、重启、提交或 push。更早 v33 的生产排版改动仍保留，
不能将本阶段的“无生产改动”扩展为整个恢复任务都没有改生产。
