# ASA v22：彻底停用 MiniRAG 与训练镜像同步

日期：2026-09-09  
规范代码/数据源：241 `${REMOTE_HOME}/ASA-ArknightStoryAgent`  
训练镜像：248 `${REMOTE_HOME}/ASA-ArknightStoryAgent`

## 结论

用户明确决定 ASA 不再使用 MiniRAG。本轮已在 241 规范仓库和 248
训练镜像完成 fail-closed：

- 不加载 MiniRAG 图索引；
- 不执行图召回、图融合、章节图隔离或图驱动二次检索；
- `minimal_conclusion_prompt` 与 `direct_answer_prompt` 不再生成
  `minirag_hints`；
- 即使旧调用方传入 `enable_minirag=true` 或图索引路径，规范搜索层和
  retriever 构造层也不会调用图搜索或 `MiniRAGIndex.load`；
- dense、sparse、reranker、storyline scope、scoped chapter search、
  neighbor expansion 与 same-story sweep 保留；
- 历史 MiniRAG 源码、图索引、旧实验和失败记录没有删除，供审计使用。

这是架构切换和无调用证明，不是回答正确率已经提高的证明。

## 241 规范仓库改动

第一层关闭运行链路：

- 规范 GPU 配置把 MiniRAG 开关、权重和 top-k 归零，索引路径设为
  `null`；
- `QueryConfig` 的 MiniRAG 默认值归零/关闭；
- 每轮 retrieval 固定执行普通 dense/sparse 检索，不进入图 scope 或图
  second pass；
- `search_queries` 不调用 `minirag_search`，显式
  `search_minirag_queries` 入口返回空结果；
- finalization 丢弃外部误注入的 graph hits；
- 两个答案提示移除 `minirag_hints`；
- `run_cpu_inference.py` 对显式启用 MiniRAG 请求 fail closed。

第二层补上第一轮测试发现的漏口：

- `src/goldenglow/retrieval/hybrid.py` 和 release retriever 的
  `from_paths` 保留兼容参数但直接丢弃 `minirag_index_path`，无法执行
  `MiniRAGIndex.load`；
- 两份 CPU 活动配置也归零/关闭 MiniRAG；
- 扫描 `configs/*.json` 和 release configs 后，活动配置中
  `enable_minirag=true` 的数量为 0。

## 241 验证

第一层旧源码基线为 12 项中 9 失败、3 通过；失败覆盖配置、搜索、融合、
图扩展、提示注入和 CLI。候选及规范应用后均为 387/387 通过。

62 个冻结状态重放确认：

- evidence 与 packing 全部逐字段不变；
- 唯一提示变化是删除 `minirag_hints` 行；
- 所有提示仍满足配置的输出预算；
- 每个提示减少 23–160 tokens。

中央加载层补漏的旧源码基线为 15 项中 3 失败、12 通过；候选及规范应用后
均为 390/390 通过。最终机器回执记录：

- 活动 MiniRAG 配置：0；
- `MiniRAGIndex.load` 可达调用：0；
- MiniRAG 搜索调用：0；
- 服务重启：否；
- 新训练：0。

## 248 训练镜像

248 是无 `.git` 的训练同步镜像，不按 Git 仓库处理。同步前逐文件守卫发现：

- 7 个核心 release 文件与 241 应用前哈希一致；
- release retriever 与 241 应用前哈希一致；
- 248 的 root retriever、runner 和 4 份活动配置有训练机自身版本，未用
  241 整文件覆盖，而是在各自版本上只移植 MiniRAG 禁用改动；
- 完整 release 源码只有
  `inference/retrieval/minirag_prompt_hints.py` 与 241 不同。该文件是
  248 未同步的旧元数据清理版本；两个规范提示构造器现在不再导入或调用
  它，因此保留为不可达历史差异。

248 共同步 14 个目标文件，并新增同一无 MiniRAG 定向测试。旧镜像基线为
15 项中 12 失败、3 通过；隔离候选和应用后的 CPU 回归均为 221/221
通过。最终活动配置中 `enable_minirag=true` 的数量为 0，图加载和图搜索
可达调用均为 0。未修改模型或训练数据，未使用 GPU，未重启服务。

第一次镜像 prepare 因完整源码存在上述一个历史差异而被守卫拒绝；该失败
目录原样保留。v2 明确把它限定为唯一允许差异后才继续，没有放宽功能断言。

## 关于“79 条测试集”

现有 fixed79 是 58 个去重问题的 79 个历史 evidence states，不是 79 道
相互独立的端到端问题。其旧动作统计为 39 direct / 31 retrieve /
9 abstain；旧自动内容标签 19 complete / 11 partial / 8 unsupported /
40 no-answer / 1 error 已确认存在假阳性，不能称为 19/79 正确率。

因此，停用 MiniRAG 后不能只重放这 79 个冻结状态就宣称 RAG 召回率或最终
答案正确率变化。后续应：

1. 从 58 个唯一问题中冻结每题一个起始输入，另保留真正未调参 holdout；
2. 使用当前无 MiniRAG 代码重新执行 dense+sparse+reranker+邻居扩展和
   最多两轮生成；
3. 分开统计候选召回、进入 prompt 的证据覆盖、证据充分条件下的生成正确率、
   unsupported answer、误拒答和过度检索；
4. 对答案做 claim-level 人工/严格证据审核，不再沿用旧 19/79 自动标签。

## 当前状态

- 241 与 248 已在代码和活动配置层停用 MiniRAG；
- 历史图代码和索引保留；
- 尚未重启线上服务，因此不能声称现有常驻进程已经加载新代码；
- 尚未运行新的无 MiniRAG 完整 RAG 评测；
- 尚无新的可认证回答正确率或完整 RAG 召回率；
- 下一阶段立即建立无 MiniRAG 的可比端到端基线，再围绕答案生成瓶颈做优化。
