# ASA 项目交接文档

日期：2026-10-25。编写目的：为后续开发者提供项目当前状态的完整入口，
包括仓库结构、发布状态、正在进行的实验、已有证据的边界、待完成事项和下一步方向。

本文档不包含 API 密钥、SSH 凭证、私有绝对机器路径或原始调试会话标识。
远端服务器使用逻辑名（241 = 规范源；248 = 训练镜像），实验路径相对于实验根目录。

---

## 1. 项目概述

ASA（ArknightStoryAgent）是面向《明日方舟》剧情的本地优先 RAG 问答系统。
核心能力（引自 README）：从大量剧情、档案、语音文本中定位具体事件，识别角色别名和跨章节关系，
回答多跳推理问题，并将每个事实绑定到模型实际可见的证据。

支持三种运行模式：GPU 本地（vLLM + Qwen3.5 4B + LoRA + reranker）、
CPU 本地（llama.cpp + GGUF）、CPU API（本地检索 + 远程 OpenAI 兼容 API）。
配有交互式 Web UI。

详细架构见 `ARCHITECTURE.md`，完整部署流程见 `README.md`。

## 2. 仓库结构

### 2.1 两个 worktree

同一 Git 仓库的两个工作树：

| 工作树 | 分支 | HEAD | 用途 |
| --- | --- | --- | --- |
| `ASA-ArknightStoryAgent/` | `fix/grounding-pool-chain-priority` | `83f2f90`（本交接文档提交前的代码/研究文档基线） | 研究：检索/grounding/训练数据实验 |
| `ASA-API-Release/` | `release/api-ui-20260914` | `ff22ed8` | 发布：面向用户的 API/WebUI 精简版 |

### 2.2 研究工作区当前状态（接管基线）

- 接管时的 26 个修改和已审查的工具/测试/报告，现已拆为 6 个原子提交；工作树没有已修改或暂存文件。
- 保留的未跟踪项目是 raw evaluation/prompt/prediction/data、第三方全文、运行日志和运维记录；它们不进入公开 Git 提交，按 `PUBLICATION_BOUNDARIES_20261025.md` 等待私有归档与许可审查。
- 六个接管基线提交：`03d053c`（可见证据 grounding）、`ee2844d`（训练 lineage 审计）、`ed1f0e4`（renderer 诊断）、`931c8ab`（GLM 标注管线）、`56a78e6`（实验协议）、`83f2f90`（研究文档与发布边界）。
- 本交接文档是独立文档提交；研究分支只允许 fast-forward 推送，不能改写 `main` 或发布分支。

### 2.3 发布工作区当前状态

- 工作区干净，无修改/未跟踪/暂存
- Tag `api-v1.1.0`（2026-09-14）：支持自定义 API 连接
- 已核对 CPU/API standard 配置：CPU dense + BM25、MiniRAG 开启、reranker/邻接扩展关闭、`rerank_top_k=20`、最多 2 轮
- 不能由 README 的“先答后检索”描述推断当前 standard 配置；详见第 3 节的不一致项
- 发布进程实际选择了何种配置尚未端到端复验
- 外部入口：API 模式（`api-mode/`）和 Web UI（`web/`）

### 2.4 远端角色约定

参见工作区外路径 `../.coverage_v8_remote/REPOSITORY_ROLES.md`（位于 ASA/ 根，不在 Git 仓库内）：

- **241**（规范源）：代码、检索候选、API 标注、checkpoint、manifest、评测报告的创建和保留地
- **248**（训练镜像）：接收 241 的不可变训练输入，运行训练/评测，结果回传 241 审核后再提升
- 同步方向：241 → 248，单向，不 `--delete`

当前规范工作根：241 上的 `rank_v8_coverage/`。

## 3. 发布版链路关键差异

README 描述了 GPU/CPU/API 三模式，但实际发布版 runtime 配置与研究分支的默认有差异：

| 维度 | 发布版实际默认 | README 部分描述 |
| --- | --- | --- |
| CPU/API standard 配置 | `configs/runtime_cpu_api_no_reranker.json` 实际为 `pipeline_mode=standard`、MiniRAG 开启、reranker 关闭、邻接扩展关闭、`rerank_top_k=20`、最多 2 轮 | README 将 CPU API 默认描述为“先答后检索再校正”，并说切到 `standard` 才回到完整 planning；这与现有配置不一致，发布前应修正文档或配置，不能凭 README 推断运行行为 |
| Reranker | CPU/API 关闭；GPU 默认开启 | 一致 |
| Web 上下文 | 发布配置关闭 | GPU 默认关闭说明在 RELEASE_NOTES |
| 最大证据 chunk | `rerank_top_k=20` | README 未在模式说明中完整列出 |
| 最大轮次 | 2 | README 已说明默认 2 轮 |

研究分支有完整块 packing、可见证据 grounding、strict fallback 封闭、lineage 审计等能力。
本轮定向回归测试已验证，但尚未完成端到端运行复验；不能直接等同于发布版行为。

## 4. 模型与训练状态

### 4.1 发布版 LoRA（历史记录）

当前发布 LoRA（`cutoff6656`，2026-06-07）基于约 200 条旧数据和 Qwen3.5 4B + LoRA rank8 训练。
GPU LoRA 路径：`model/lora/asa-arknightstoryagent-4b-lora/`。
CPU GGUF 未确认由该权重合并导出（RELEASE_NOTES 已标注）。
此 LoRA 版本为历史状态，后续 Bonsai2 数据和 180 条扩充候选均未用于任何已发布模型。

### 4.2 Bonsai2 P-tuning 实验

实验根目录：241 上的 `bonsai2_ptuning_20261012/`（与 `rank_v8_coverage/` 的位置关系未经本次确认）。

27B 冻结主干上的 soft-prompt 实验，两个配置：

| 配置 | 训练数据 | 训练步数 | 完整正确 | 部分正确 | 错误 | 完整正确率 | JSON 有效 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 输入层 | 1117 条 | 1117 步 | 24 | 13 | 9 | 52.17% | 45/46 |
| 逐层 v2 | 1117 条 | 1117 步 | 15 | 23 | 8 | 32.61% | 16/46 |

验证集：同 46 题开发集（43 问题组），20 段完整证据无截断。

#### 关键观察

v2 完整正确率下降 19.57 个百分点。主要表现为过度扩写后引入主体/受话人/阶段/地点错误。
例如：

- 第 13 题：督察/朋友答对，又把陈接班身份给了星熊
- 第 35 题：战斗态度对比答对，却把谁不愿让谁独自面对危险反过来
- 第 41 题：反抗机制答对，又把对博士的恨与报复给了普瑞赛斯

#### 证据边界

- **非单因素因果**：两版同时变动多项优化器设置，不能归因于逐层注入本身
- **非盲测**：46 题反复用于开发，非独立测试集
- **非独立金标**：助手评分，非人工标注
- **非检索全链路**：固定 20 段证据，不经过检索系统
- **不可外推**为泛化准确率或上线指标

详细逐题判断见 `V2_OPTIMIZED1117_CONTENT_REVIEW_20261022.json`，
复核报告见 `V2_OPTIMIZED1117_CONTENT_REPORT_20261022.md`。

## 5. 扩数实验（2026-10-22）

### 5.1 出题

从 60 个完整 20 段训练侧证据组，生成 180 道新题。计划类型：
人物/施受关系 60、核心动机 30、事件阶段 30、部分可答 30、缺证边界 30。

使用 GLM-5.3 经 Coding Plan 端点出题。两轮动作边界交替，每轮 90 题。
仅送训练侧证据和已有问题上下文，不发送验证集内容/答案/评分。
train/val 按问题、证据文档、剧情族隔离。

结果：60 组全部完成。180 题中 171 条 GLM 标注及自复核通过、5 条隔离、4 条 `incomplete_output` 未完成。

### 5.2 助手语义复核抽查（20 条）

助手逐条阅读原文复核 20 条样本（非随机抽样，非独立人工金标）：

- 15 条 GLM 通过样本中 13 条内容可接受，2 条应返修
- 5 条既有隔离均有原文依据
- 新增 2 条隔离提案（第 3 题、第 158 题）
- 第 45、114 题建议重新归类题型（非内容拒绝）
- 171 条中 34 条超过 3 个事实，需留意扩写引入错误

20 条非随机样本，不可用 13/15 外推整批正确率。

### 5.3 修复流程

用户确认后执行远端独立修复候选生成，三轮 GLM 复核运行完成：

- 修复 10 条内容标签（原 7 条 + 恢复时发现的 108、144、179 三条）
- 纠正 2 条题型（45、114）
- 修正 1 处题面歧义（178）
- 原 4 条未完成样本补齐独立复核：175 保留原正确标签，其余 3 条改正内容
- 实际采用 14 个修改/恢复样本均有干净全文 GLM 复核回执

### 5.4 最终候选

位于远端实验目录：`input_failure_repairs_followup_20261022_attempt01/final_candidate/`

文件清单：

| 文件 | 说明 |
| --- | --- |
| `all180.jsonl` | 全量 180 条候选 |
| `assistant_reviewed24.jsonl` | 已经助手内容审核的 24 条子集 |
| `manifest.json` | 行数、SHA、修复计数、状态标志 |
| `postrepair_review.json` | 逐条修复依据及真实复核 response_id |
| `REPORT.md` | 修复结果摘要 |

关键 SHA-256：

- `all180.jsonl`：`161c3803bba3de516ec7512549218e48b313a14cb58a5310293fa38a6b38a3b5`
- 原始 171 条 `train.jsonl`：`05dd60657291e3a9066f5a1ed76ddd3204624fc51e8273b6a71020734a125339`

上述修复流程的历史成功记录来自 10/22 会话日志（finalize exit 0、counts/hash exit 0）。
远端 `final_candidate/` 目录内容在本次接管中未经独立复验。

### 5.5 当前状态

| 维度 | 状态 |
| --- | --- |
| `ready_for_training` | **false** |
| 已审 24 条 / 待审 156 条 | 未完成全量助手逐条语义复核 |
| 合并到基准训练集 | **未执行** |
| 训练 | **未执行** |
| 发布模型变更 | **未执行** |
| 原始 171 条导出 | **未删除、未修正** |

GLM 自复核不等于独立人工金标；助手逐条语义复核也不是独立人工金标。
即使 156 条助手审核全部完成，`ready_for_training` 仍需用户明确授权才可变更。

## 6. 本地实验产物索引

`.codex_tmp/asa_bonsai2_ptuning_20261012/` 下关键产物（均经 `ls` 确认存在）：

| 目录/文件 | 说明 |
| --- | --- |
| `curated_reader_data_20261019_v1/` | 精选阅读器数据 v1 |
| `input_failure_expansion_20261022_snapshot/` | 出题阶段本地快照（非自动同步） |
| `input_failure_review_snapshot_20261022/` | 审核阶段完整快照（含账本、回执、验证集） |
| `INPUT_FAILURE_EXPANSION_HANDOFF_20261022.md` | 出题交接文档 |
| `INPUT_FAILURE_DATA_REPORT_20261022.md` | 20 条助手语义复核报告 |
| `V2_OPTIMIZED1117_CONTENT_REPORT_20261022.md` | v2 内容复核报告 |
| `finalize_input_failure_repairs_20261022.py` | 修复候选 finalize 脚本 |

> 注：目录下还有 `checkpoint-1`~`checkpoint-6`（各含 `prompt_encoder.safetensors`，
> 约 11MB）和 `manifest.json`（描述 6 train / 2 val 的 tiny offline pilot，非主 Bonsai2
> 1117 步实验的 checkpoint）。主 Bonsai2 训练 adapter 权重在远端 241，本地不存在。

本地不存在 `final_candidate/` 目录——该产物仅在远端 241。

### 6.1 远端拉取优先表（历史记录，未经本次实时确认）

下表是后续在获得远端访问授权后应优先核验和拉取的逻辑路径。所有路径相对于 241 的实验根；它们是历史会话记录，不表示本次已重新列目录或已同步到本地。

| 逻辑源 | 相对路径/逻辑项 | 用途与拉取边界 |
| --- | --- | --- |
| 241 | `bonsai2_ptuning_20261012/` | 核心 P-tuning 实验根；先取 manifest、报告和不可变输入指纹，再决定是否复制大资产 |
| 241 | `input_failure_repairs_followup_20261022_attempt01/final_candidate/` | 180 条修复后候选；先校验 `manifest.json` 与 `all180.jsonl` SHA，仍不可直接训练 |
| 241 | 输入层 P-tuning run（历史目录名待实时 manifest 确认） | 与 1117-step 输入层结果对应；只在 live listing 后记录精确目录名 |
| 241 | 逐层 v2 P-tuning run（历史目录名待实时 manifest 确认） | 与 1117-step v2 结果对应；不可假定本地存在同名副本 |
| 241 | `curated_reader_data_20261019_v1/` | 精选 reader 数据；先核验来源、split 与许可，再作为训练输入候选 |
| 241 | shallow 归档/运行副本（历史标签，精确目录名待实时确认） | 仅用于追溯；不把历史存在表述为本次已确认的可用资产 |

## 7. 代码结构入口

### 检索链路

- `src/asa_arknight_story_agent/retrieval/hybrid.py`：混合检索器（dense + BM25）
- `src/asa_arknight_story_agent/retrieval/hybrid_components/`：RRF 融合、证据链构建、打分调整
- `src/asa_arknight_story_agent/retrieval/minirag.py` / `minirag_components/`：MiniRAG 图检索（现有 CPU/API standard 配置启用；实际发布进程选择仍待端到端确认）

### 推理链路

- `src/asa_arknight_story_agent/inference/pipeline/`：推理状态、轮次编排
- `src/asa_arknight_story_agent/inference/generation/`：假设生成、结论生成、prompt 渲染
- `src/asa_arknight_story_agent/inference/evidence/`：证据准备、选择、渲染
- `src/asa_arknight_story_agent/inference/grounding/`：答案 grounding、引用检查、回退
- `src/asa_arknight_story_agent/inference/planning/`：问题理解、实体抽取、续检索规划
- `src/asa_arknight_story_agent/inference/model_runtime/`：llama.cpp 与 vLLM 运行器

### 数据构建

- `data/story_parser.py`：语料构建（剧情正文 + 档案 + 语音）
- `data/operator_aliases.py`：干员别名表
- `scripts/build_retrieval_index.py`：索引构建

### 训练数据相关脚本

- `scripts/audit_exx_set_grounding.py`：grounding 审计
- `scripts/build_binding_verifier_pairs.py`：binding verifier 对构建
- `scripts/build_exx_clean_binding_sft.py`：clean binding SFT 数据构建

## 8. 已完成但需注意的事项

### 8.1 接管提交基线

接管时的 26 个修改已完成审查并拆分为 6 个原子提交：`03d053c`、`ee2844d`、`ed1f0e4`、`931c8ab`、`56a78e6`、`83f2f90`。本交接文档随后作为独立提交；后续提交、私有归档范围和未跟踪 raw 资产规则见 `PUBLICATION_BOUNDARIES_20261025.md`。

不要为历史远端结果虚构本地副本。任何新修改应基于此提交基线，并继续使用显式文件 staging。

## 9. 待完成事项

按优先级排列：

### P0 — 阻塞训练

1. **完成 156 条待审数据的助手逐条语义复核**
   - 当前 24/180 已审，156 条 `assistant_content_review` 仍为 pending
   - 20 条非随机抽查发现 2/15 应返修，需要更系统的复核
   - 助手复核完成后仍需用户明确授权才可将 `ready_for_training` 改为 true

2. **审核合格后另存新版训练集**
   - 依赖 P0.1 复核结果
   - 原始 171 条从未进入基准训练集；合格候选需经去重、隔离审计后另存为新版本，
     待用户授权合并和训练

### P1 — 阻塞发布

3. **从修复后数据训练新 adapter**
   - 依赖 P0.2 合并决策
   - 训练环境在 248，需从 241 同步不可变训练输入
   - 训练完成后需独立评测（不能复用 46 题开发集）

4. **构建独立测试集**
   - 当前 46 题已被反复使用，不具备盲测性质
   - 需要新的、与训练数据隔离的评测问题

5. **端到端复验研究分支的 grounding/evidence 改动**
   - 相关 26 个接管修改已在 6 个提交中固化，定向回归已通过
   - 仍需确认完整运行对发布版的影响，再决定是否 cherry-pick 到发布分支

### P2 — 发布与分发

6. **GitHub 仓库**
   - 已有仓库：https://github.com/MapleRhythm/ASA-ArknightStoryAgent
   - 当前状态：Git SSH 已可读，研究分支 dry-run push 已通过；本交接文档提交将随本次 fast-forward push 发布。不得推送 `main`、不得 force。

7. **HuggingFace 模型**
   - 已有项目：`MapleRhythm/ASA-ArknightStoryAgent`、对应 4b-lora / gguf / reranker 仓库
   - 当前状态：官方 TCP 连接仍超时，尚未上传；不把未同步的远端或本地 raw 资产虚构为 HF 已归档。

8. **确认 CPU GGUF 来源并重新导出**（如需）

### P3 — 改进方向

9. **改进扩写控制**
   - v2 实验的主要失败模式是过度扩写引入事实错误
   - 考虑对生成长度和 fact 数量施加约束

## 10. 关键文件快速索引

| 文件 | 位置 | 说明 |
| --- | --- | --- |
| `README.md` | 研究工作区根 | 部署与使用指南 |
| `ARCHITECTURE.md` | 研究工作区根 | 链路代码结构 |
| `docs/RELEASE_NOTES.md` | 研究 docs/ | GPU LoRA cutoff6656 发布说明 |
| `docs/PROJECT_ARCHITECTURE_METHODS_RESULTS_20260907.md` | 研究 docs/ | 9 月架构/方法/结果快照（有隔离警告） |
| `docs/API_RELEASE_V1_1.md` | 发布 docs/ | api-v1.1.0 发布说明 |
| `REPOSITORY_ROLES.md` | 工作区外 `../.coverage_v8_remote/` | 241/248 角色与同步规则 |
| `INPUT_FAILURE_EXPANSION_HANDOFF_20261022.md` | 实验目录 | 10/22 出题交接 |
| `V2_OPTIMIZED1117_CONTENT_REPORT_20261022.md` | 实验目录 | v2 内容复核 |
| `finalize_input_failure_repairs_20261022.py` | 实验目录 | 修复候选 finalize 脚本 |

## 11. 操作约束提醒

- 241 → 248 单向同步，不在 248 编辑规范数据
- 训练输出回传 241 时进入 `incoming_from_248/<run_id>/`，校验后再提升
- 每个训练快照必须包含 manifest（源路径、SHA-256、记录数、协议版本、创建时间、run_id）
- 不要对已封存脚本（如 `expand_input_failures_20261022.py`）热修改——其 SHA 与 plan 绑定
- `ready_for_training=false` 状态在用户明确授权前不可变更（助手复核通过也不自动变更）

---

*本文档由 Pi 助手于 2026-10-25 编写。如有疑问请参阅上述各引用文件的原始内容。*
