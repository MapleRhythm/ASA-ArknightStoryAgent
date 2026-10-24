# 一手文献核验与近邻边界

核验日期：2026-09-10，Asia/Shanghai。是针对候选机制的定向检索，不是穷尽综述，不保证找到截至此日所有最新论文。

本轮可见 web 工具返回为空，不能据此声称阅读成功或给出有效 web 引用。
曾有中途进度和本文件旧稿误称读到近期论文原文，现撤回这些说法。
下面 L01–L06 以及后续实际 HTTP 取得的 L16–L20 有落盘资料与下载 manifest 支撑。
L07–L15 仍然隔离；成功取得其他文献不追认它们。这不表示已完成近期文献查全。

`manifest.json` 记录 ACL 官方维护仓库的 XML 及早期 arXiv HTTP 失败。
`manifest_papers.json` 记录 PDF 下载与提取状态。CONQRR 的 PDF 只下载了部分，**不能作为完整 PDF 阅读依据**；
其书目与摘要来自完整的官方 XML。`manifest_recent.json` 的超时不代表那些会议或论文不存在。
没有可核验的后续 web 阅读记录，不用叙述性进度或上下文摘要替代原文证据。

## L01：FactGraph，NAACL 2022

题目：*FactGraph: Evaluating Factuality in Summarization with Semantic Graph Representations*。
作者：Leonardo F. R. Ribeiro 等。
定位：`https://aclanthology.org/2022.naacl-main.236/`
本地全文：`2022.naacl-main.236.pdf`、`2022.naacl-main.236.txt`。
阅读范围：摘要、引言、图1、相关工作、方法3.1–3.2；并非逐页复现。

已覆盖：文本和语义图共同用于事实一致性评估。图1直接对比“媒体报道某人考虑解职”和“某人说将解职”，已经包含模态/转述节点丢失问题。
因此“加主体、关系、模态图来验事实”不能当作本项目的新贡献。

## L02：FRANK，NAACL 2021

题目：*Understanding Factuality in Abstractive Summarization with FRANK: A Benchmark for Factuality Metrics*。
作者：Artidoro Pagnoni、Vidhisha Balachandran、Yulia Tsvetkov。
定位：`https://aclanthology.org/2021.naacl-main.383/`
本地全文：`2021.naacl-main.383.pdf`、`2021.naacl-main.383.txt`。
阅读范围：摘要、引言、错误分类2.1–2.3。

已覆盖：谓词、实体、情境、共指、篇章连接、源外信息等错误分类，实体类别明确包含主客体倒置。
因此 ASA 的错误分类应引用和适配这些维度，不宣称首次发现主体/时间/否定错误。

## L03：文学引语归因，ACL 2023

题目：*Improving Automatic Quotation Attribution in Literary Novels*。
作者：Krishnapriya Vishnubhotla 等。
定位：`https://aclanthology.org/2023.acl-short.64/`
本地全文：`2023.acl-short.64.pdf`、`2023.acl-short.64.txt`。
阅读范围：摘要、任务分解、PDNC 与5.1–5.3。

已覆盖：角色识别、共指、引语识别、说话人归因的分解与评测。
ASA 已有说话人字段，但外层说话人并不自动等于内层观点持有人；候选方法必须超越仅恢复 speaker 标签。

## L04：Rewrite-Retrieve-Read，EMNLP 2023

出版页题目：*Query Rewriting in Retrieval-Augmented Large Language Models*；
PDF 首页题目使用 “for” 而不是 “in”，以各来源原样记录，不另造两个工作。
作者：Xinbei Ma 等。
定位：`https://aclanthology.org/2023.emnlp-main.322/`
本地全文：`2023.emnlp-main.322.pdf`、`2023.emnlp-main.322.txt`。
阅读范围：摘要、引言、框架图、方法3.1与3.2。

已覆盖：固定检索/reader，通过 reader 反馈训练小型 query rewriter。
“用回答奖励训练检索查询”不是新方向。

## L05：CONQRR，EMNLP 2022

题目：*CONQRR: Conversational Query Rewriting for Retrieval with Reinforcement Learning*。
定位：`https://aclanthology.org/2022.emnlp-main.679/`
阅读范围：官方 XML 中完整书目及摘要。PDF 下载超时且不完整，未声称阅读方法全文。

已覆盖：用检索奖励优化会话查询重写，适配现成检索器。是候选查询方法的必要早期对照。

## L06：ALCE，EMNLP 2023

题目：*Enabling Large Language Models to Generate Text with Citations*。
作者：Tianyu Gao 等。
定位：`https://aclanthology.org/2023.emnlp-main.398/`
本地全文：`2023.emnlp-main.398.pdf`、`2023.emnlp-main.398.txt`。
阅读范围：摘要、任务定义、指标介绍。

已覆盖：答案质量与多段引用质量分开评价。ASA 的引用编号存在率不能替代引用蕴含，亦不能将“多证据引用”本身当创新。

## 隔离区：旧稿 L07–L15，没有核验成功

下表仅用于识别并阻止污染传播，**不是参考文献表**。
简称和编号来自旧稿，连同其对应关系都尚未核实；删除旧稿的作者、日期、方法和阅读章节断言。
不得用这些线索证明“已有工作做过”，也不得因未取到它们而宣称“没有人做过”。

| 旧编号 | 旧稿声称的简称 | 旧稿声称的 arXiv 编号 | 当前状态 |
|---|---|---|---|
| L07 | CUE-R | 2604.05467 | HTTP 失败记录；无可核验原文 |
| L08 | Pair-ID | 2608.08944 | HTTP 失败记录；无可核验原文 |
| L09 | GRAB-RAG | 2608.22228 | HTTP 失败记录；无可核验原文 |
| L10 | MetaRAG | 2608.24214 | 仅旧稿线索，未核实 |
| L11 | Bridge Evidence | 2607.15253 | 仅旧稿线索，未核实 |
| L12 | DPS | 2508.09497 | 仅旧稿线索，未核实 |
| L13 | SAGE | 2506.19783 | 仅旧稿线索，未核实 |
| L14 | SPA | 2606.08620 | 仅旧稿线索，未核实 |
| L15 | GenerationPrograms | 2506.14580 | 仅旧稿线索，未核实 |

## 仍未完成的核查

- DVCQR 的旧搜索说法没有证据；本轮后来从官方 XML 实际取得的独立条目见 L20，不追认旧 web 阅读。
- 缺少 2025–2026 作用域蕴含、嵌套事实性、语义压缩的系统查全，当前候选只能叫“有区分点的可检验假设”。
- 旧 `2021.emnlp-main.315` 引用错误已在前轮证实；本轮未找到原声称 false-negative-contexts 题目的可靠对应，继续停用，而非猜编号。
- 本轮没有独立复现任何论文报告的模型增益，不把论文数字搬到 ASA。

## 本轮后续实际取得的近期近邻

证据文件：

- `focused_attempt2/manifest_focused.json`：2024/2025 TACL、2025/2026 ACL 官方 XML 均完整取得；
  通过标题关键词定向筛选，不是全文查全。该次作者字段包含 XML 机构内容，以下作者名以 PDF 首页为准。
- `near_neighbors_attempt1/manifest_near_neighbors.json`：三份完整 PDF/TXT，两份超时 PDF。
- 早期失败历史没有覆盖。HTTP 状态 200 但下载超时仍标为 `unverified`，不能冒充已读全文。

### L16：来源—模态作用域，ACL 2026

XML 题目：*Modal Dependency Parsing as Structured Prediction over Source-Cue Scope*；
PDF 首页使用 *Source–Cue Scopes*。作者：Jayeol Chun、Nianwen Xue。
编号：`2026.acl-long.1362`。
本地：`near_neighbors_attempt1/2026.acl-long.1362.txt`。
已读：摘要、引言、相关工作、3.1–3.2 方法和紧接其后的作用域标注说明。

它明确建模来源、模态线索和嵌套作用域，并据此约束事件归因；英文和中文均在实验范围内。
因此“把 who-believes-what 做成结构图”以及“按最小包围作用域连接来源”不能作为 ASA 新贡献。
我们的待检验区别应是：固定问答目标下的多套合法证明、作用域干预后的监督关系及 reader 行为，
而不是复述它的解析目标。解析输出也不自动保证文本或标注无误。

### L17：LCR-RAG，ACL 2026

题目：*LCR-RAG: Enhancing Logical Consistency in Retrieval-Augmented Generation via Neuro-symbolic Reinforcement Learning*。
编号：`2026.acl-long.814`。
本地：`near_neighbors_attempt1/2026.acl-long.814.txt`。
已读：摘要、引言、3.1–3.3 方法、实验设置入口。

它将 NLI、RDF/规则验证反馈用于 PPO 查询编辑；查询策略选择操作，再由冻结模型改写。
方法3.2的事实校验为每个 claim 取 BM25 top-1 证据句，形式逻辑部分检测答案三元组内的不一致。
这与“可见集合中允许多个合法且保留归因的支持集”不相同，但不等于我们已证明它不能处理该问题。
不能把“外部逻辑反馈＋查询修复”当原创；也不能直接照搬其多组件系统或论文性能到 ASA。

### L18：SynQA，ACL 2025

题目：*On Synthesizing Data for Context Attribution in Question Answering*。
编号：`2025.acl-long.828`。
本地：`near_neighbors_attempt1/2025.acl-long.828.txt`。
已读：摘要、引言、2.1–2.2、3.1指标入口。

先选支持句、生成 QA，再加入相似干扰文本并训练小型归因模型的路线已有研究。
不能把“先证据后问题＋合成难例＋小模型微调”当原创。其 attribution 任务与答不答的动作学习不同。
ASA 需要独立验证剩余候选是否真的无替代支持，不能因来源文档未重复就认定不存在替代证明。

### L19：CAQA，ACL 2025

题目：*Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs*。
编号：`2025.acl-long.837`。
官方 XML 摘要已读：`focused_attempt2/2025.acl.xml` 中该条目。
首次 PDF 超时；在未取得完整文件前，不声称看过方法章节。

摘要明确覆盖知识图谱自动构造、细粒度归因类别、细微归因差异，以及 evaluator 微调。
它是本项目复杂归因评测和合成数据的直接近邻，不能只比较词法规则。
具体类别和数据构造细节应以完整原文复核，不从摘要推测。

### L20：DVCQR，ACL 2026

题目：*DVCQR: Dual-View Conversational Query Rewriting with Stage-wise Reinforcement Learning*。
编号：`2026.acl-long.1054`。
已读：`focused_attempt2/2026.acl.xml` 中的书目与完整摘要；PDF 超时，未按全文使用。

摘要描述针对 sparse/dense 两种检索器生成互补查询，并分阶段用检索反馈优化。
因此“给 BM25 和 dense 分别生成查询”不是足够的区分点。
对 ASA 的候选只能进一步检验完整后处理对实际查询动作的影响，不宣称执行感知思想首创。
