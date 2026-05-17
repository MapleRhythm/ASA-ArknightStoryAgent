# ASA-ArknightStoryAgent 开发代理指南

本文件用于指导 AI 开发代理在本仓库内做代码修改、数据处理、训练和推理调试。项目目标是做《明日方舟》剧情问答 Agent：回答必须基于检索证据，不能把模型记忆、二创设定或猜测写成剧情事实。

## 1. 当前项目定位

- 核心任务：中文剧情问答、剧情关系解释、事件因果分析、多轮追问消解。
- 核心原则：先保证证据正确，再考虑表达风格。
- 生成模型职责：生成 hypothesis、生成 conclusion 决策、基于证据生成最终答案。
- 检索系统职责：从剧情原文、档案、语音和元数据构建候选证据，并通过 reranker 排序。
- 不要把项目改成泛聊天 Bot，也不要让模型在没有证据时自由发挥。

## 2. 当前技术架构

主代码位于 `src/goldenglow/`：

- `src/goldenglow/config.py`：全局路径与 `QueryConfig`。
- `src/goldenglow/data/story_parser.py`：剧情文本解析与 chunk 构建。
- `src/goldenglow/data/sft_teacher.py`：teacher 数据生成通用逻辑。
- `src/goldenglow/retrieval/hybrid.py`：dense / sparse 召回、RRF 融合、reranker 重排。
- `src/goldenglow/retrieval/reranker.py`：交叉编码器 reranker 封装。
- `src/goldenglow/inference/cpu_pipeline.py`：多轮 RAG 推理主链路。

当前在线链路：

1. 用户问题进入推理入口。
2. 生成初始 `HypothesisDocument`。
3. 根据 hypothesis 构造检索 query。
4. 执行 dense + sparse 召回。
5. 使用 RRF 融合候选。
6. 使用 reranker 对候选证据或证据链排序。
7. 生成 `ConclusionResult`，决定 `answer_directly` / `retrieve_more` / `clarify_user` / `abstain`。
8. 若 `retrieve_more` 且未达到上限，则使用 follow-up hypothesis 进入下一轮检索。
9. 达到可回答、需澄清、放弃或最大检索轮次后停止。

默认总检索轮次由 `inference.max_retrieval_rounds` 控制，当前配置为 `3`。不要为了修某个样例强制增加检索轮次。

## 3. 环境与设备

本项目默认区分两个 conda 环境：

- `train`：训练、数据生成、GPU 检索、reranker 训练、vLLM 推理、API 模式调试。
- `reasoning`：CPU / llama.cpp 推理、轻量检索调试、离线运行验证。

常用 Python 路径：

- `train`: `/home/zhb/miniconda3/envs/train/bin/python`
- `reasoning`: 以本机 conda 环境为准；若脚本需要精确解释器，优先显式传 `PYTHON_BIN` 或使用环境内 `python`。

当前机器按项目使用约定视为 3 块 RTX 4090。多卡训练默认使用：

```bash
TRAIN_GPUS=0,1,2
```

单卡调试或推理优先使用 `CUDA_VISIBLE_DEVICES` 显式指定，例如：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhb/miniconda3/envs/train/bin/python ...
```

Python overlay 约定：

- 主 overlay 目录：`.python_packages/train`
- 第三方轻量依赖目录：`.python_packages/third_party`
- train 环境覆盖目录：`.vendor/train_override`
- 临时过滤 overlay：`outputs/.cache/python_overlay_filtered*`

`scripts/run_cpu_inference.py`、`api-mode/run_api_inference.py`、`scripts/build_retrieval_index.py` 会在检测到 `train` 环境或 `/envs/train/bin/python` 时自动把 `.python_packages/train` 和 `.vendor/train_override` 加入 `sys.path`。可用 `GOLDENGLOW_USE_TRAIN_OVERRIDE=0/1` 显式关闭或开启。

训练脚本如 `scripts/llama_factory/run_train.sh`、`scripts/run_reranker_sft.sh` 会使用 `PYTHON_OVERLAY_DIR`，并在需要时生成过滤后的 `FILTERED_OVERLAY_DIR`，避免 overlay 中的重型包相互污染。不要直接删除 `.python_packages/train` 或 `outputs/.cache/python_overlay_filtered*`，除非正在明确重建依赖环境。

如果需要安装或重建 vLLM overlay：

```bash
conda activate train
PYTHON_OVERLAY_DIR=.python_packages/train bash scripts/install_train_vllm.sh
```

## 4. 运行入口

本地模型模式：

- 入口：`scripts/run_cpu_inference.py`
- 配置：`configs/runtime_inference.json`、`configs/runtime_inference_gpu.json`
- 支持后端：`llama.cpp`、`vllm`
- GPU 脚本：`scripts/run_gpu_inference.sh`

API 模式：

- 入口：`api-mode/run_api_inference.py`
- 配置：`api-mode/runtime_api.json`
- 当前支持后端：`openai_compatible_api` / `chat_completions`、`responses_api` / `responses`
- 当前 API 配置使用火山 Ark Responses API：
  - `api_base_url`: `https://ark.cn-beijing.volces.com/api/v3/responses`
  - `api_key_env`: `ARK_API_KEY`
  - `model`: `doubao-seed-2-0-mini-260428`

运行 API 模式示例：

```bash
export ARK_API_KEY="..."

CUDA_VISIBLE_DEVICES=1 \
/home/zhb/miniconda3/envs/train/bin/python api-mode/run_api_inference.py \
  --runtime-config api-mode/runtime_api.json \
  "炎景公主一事具体指什么"
```

## 5. 数据与索引

主要数据路径：

- `data/ArknightsGameData/zh_CN/gamedata/story/`：剧情原文。
- `data/ArknightsGameData/zh_CN/gamedata/excel/`：角色、章节、关卡、回顾等元数据。
- `indexes/arknights_story/documents.jsonl`：检索文档。
- `indexes/arknights_story/faiss.index`：FAISS 向量索引。
- `indexes/arknights_story/bm25_tokens.pkl`：BM25 tokens。
- `indexes/arknights_story/index_meta.json`：索引元信息。
- `indexes/arknights_story/operator_aliases.json`：干员别名。

构建索引：

```bash
python scripts/build_retrieval_index.py --device cpu
```

调试检索：

```bash
python scripts/query_retrieval.py "岁兽是什么，为什么会成为危机"
```

剧情 chunk 必须尽量保留叙事结构与元数据，不要粗暴把原始脚本整包塞进向量库。修改 parser 或 chunker 时，必须确认 `source_path`、`activity_name`、`story_name`、`stage_code`、`avg_tag`、`clean_text` 等字段仍可追踪。

## 6. 检索与重排规则

检索基础链路是：

```text
query -> dense_search + sparse_search -> reciprocal_rank_fusion -> reranker -> evidence
```

当前 reranker 默认优先使用：

- `model/reranker/bge-reranker-v2-m3-evidence-chain-answerability`

回退路径：

- `model/reranker/bge-reranker-v2-m3`

相关配置字段：

- `retrieval.device`
- `retrieval.enable_reranker`
- `retrieval.reranker_model_path`
- `retrieval.dense_top_k`
- `retrieval.sparse_top_k`
- `retrieval.fusion_top_k`
- `retrieval.rerank_top_k`
- `retrieval.rerank_batch_size`
- `retrieval.reranker_max_length`

实现检索优化时遵守以下规则：

- 不要为单个问题、单个角色、单个活动写特殊分支。
- 不要用硬编码关键词强行改排序结果来修个例。
- 排序能力优先通过数据、reranker 训练、query schema 和 prompt 改进。
- 如果需要改变召回或排序策略，应做成配置项或模型可学习信号。
- 必须保留 dense / sparse / fusion / rerank 分数，便于诊断。
- 对事实类、关系类、因果类问题的差异，优先通过 `query_type`、训练数据和模型决策表达，不要写死样例规则。

## 7. 多轮 RAG 推理

核心数据结构在 `src/goldenglow/inference/cpu_pipeline.py`：

- `HypothesisDocument`
- `ConclusionResult`
- `InferenceResult`

允许的 hypothesis `intent`：

- `plot_fact`
- `plot_reasoning`
- `timeline`
- `character_relation`
- `event_summary`
- `compare`
- `persona_chat`
- `out_of_scope`

允许的 `query_type`：

- `fact`
- `relation`
- `causality`
- `reasoning`
- `reveal`
- `mystery`
- `answerability`

允许的 conclusion action：

- `answer_directly`
- `retrieve_more`
- `clarify_user`
- `abstain`

`retrieve_more` 只能由模型 conclusion 决定；代码不应因为某类问题自动强制追加检索轮。达到 `max_retrieval_rounds` 后必须停止扩展检索。

## 8. Prompt 与 JSON 输出

模型输出 JSON 时必须稳定、可解析、字段名固定。当前 tool task 类型：

- `user_question_hypothesis_generation`
- `follow_up_hypothesis_generation`
- `conclusion_generation`

修改 prompt 时优先修正以下问题：

- 字段缺失或枚举值不合法。
- 把答案泄漏进 hypothesis。
- conclusion 在证据不足时硬答。
- `retrieve_more` 没有给出可检索的 `missing_slots`。
- `clarify_user` 被误用于“检索证据不足”而非“用户问题歧义”。

不要要求模型输出 chain-of-thought。API 模式下也不要依赖 reasoning 内容，最终可用信息必须在 `content` 或 Responses API 的文本输出中。

## 9. Evidence Chain Reranker 数据

证据链数据与训练相关路径：

- `scripts/evidence_chain_dataset.py`
- `scripts/run_evidence_chain_batch.py`
- `scripts/train_evidence_chain_reranker.py`
- `scripts/run_reranker_sft.sh`
- `scripts/evaluate_evidence_chain_reranker.py`
- `docs/evidence_chain_reranker_dataset.md`
- `data/processed/evidence_chain_reranker/`

当前主要数据集：

- `data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/`

关键产物：

- `annotations.cleaned.jsonl`
- `validation_issues.jsonl`
- `reranker_pairwise.jsonl`
- `reranker_listwise.jsonl`
- `quality_report.json`

证据链训练目标是学习“这条证据链是否足以回答问题”，而不是学习某个固定关键词。负例应覆盖：

- `background_only`
- `answer_adjacent`
- `partial_answer`
- `same_entity_distractor`
- `shuffled_order`
- `misleading_chain`

如果线上排序失败，优先抽样检查数据质量、负例强度、query_type 分布和训练评估，不要直接在 runtime 写问题特化规则。

## 10. SFT 与本地模型训练

主训练链路仍保留：

- 基座：`model/qwen3.5-4b`
- LoRA 输出：`model/lora/`
- 训练入口：`scripts/llama_factory/run_train.sh`
- 数据转换：`scripts/llama_factory/prepare_sft_dataset.py`
- 训练配置：`src/config/llama_factory_config.yaml`

相关数据生成脚本：

- `scripts/generate_sft_from_teacher.py`
- `scripts/generate_prompt_supplement_from_teacher.py`
- `scripts/generate_prompt_supplement_from_teacher_merged.py`
- `scripts/merge_sft_datasets.py`
- `scripts/migrate_dataset_task_labels.py`
- `scripts/repair_teacher_v2_dataset.py`

训练数据要求：

- 保留原始输入、teacher 输出、清洗结果和版本信息。
- 抽样检查事实忠实性，特别是角色种族、亲属关系、事件因果、时间线。
- 风格数据不能污染事实标签。
- hypothesis 数据只写检索线索，不要写最终答案。
- conclusion 数据要严格区分 `answer_directly`、`retrieve_more`、`clarify_user`、`abstain`。

## 11. API 模式注意事项

`api-mode/run_api_inference.py` 会复用本地检索和 `CPUInferencePipeline`，只替换生成模型。

OpenAI-compatible chat completions：

- `backend`: `openai_compatible_api` 或 `chat_completions`
- URL 会规范到 `/chat/completions`
- payload 使用 `messages`

Responses API：

- `backend`: `responses_api` 或 `responses`
- URL 会规范到 `/responses`
- payload 使用 `input`
- 文本从 `output_text` 或 `output[].content[].text` 提取

不要把 API key 写入仓库；只能通过环境变量或 CLI 传入。请求日志默认写入 `outputs/api_mode_runs/`，用于复盘 prompt、payload 和模型输出。

## 12. 回答生成原则

最终答案必须遵守：

- 先给结论，再解释证据。
- 明确区分“原文直接写明”“多段证据归纳”“现有证据不足”。
- 如果证据不足，直接说明不足，不要用语气包装成事实。
- 不要把上一轮错误答案当作事实延续。
- 多轮追问中要用上下文做指代消解，但歧义过大时应澄清。

澄闪风格只属于表达层：

- 可以礼貌、轻柔、克制。
- 不要过度口癖化。
- 不要为了角色语气改变剧情事实。

## 12.1 当前痛点（接手 AI 必读）

截至 2026-05-13 的两个核心痛点：

1. **推理类问题（causality / reasoning / reveal / answerability）容易乱回答** —— 4B 在证据不足或证据片面时倾向硬答，缺乏 abstain 与 clarify 能力。
2. **事实类问题（fact / relation）难召回到正确文档** —— BM25 单字噪声已修，但 dense + reranker 路径在干员别名、profile 来源、关系桥接上仍弱。

端到端召回 baseline：
- `outputs/retrieval_eval/baseline.json` recall@1 = 0.237 / recall@10 = 0.511 / MRR = 0.325
- 详细对比见 `outputs/retrieval_eval/comparison_report.md`

已经做过的修复（请勿重复）：
- BM25 tokenizer 去掉中文单字（`src/goldenglow/retrieval/hybrid.py:tokenize_for_bm25`）
- 削弱 `_compute_original_query_match_bonus` 的字符级 bonus、调小 fact/relation 的 `_original_query_bonus_scale`
- `validate_conclusion_grounding` 实装实体级 grounding 校验（`src/goldenglow/inference/cpu_pipeline.py:1637`）
- `operator_aliases.json` 已接入 hypothesis 生成，会自动展开干员别名到 keywords
- evidence chain reranker 训练数据已增加 chain structure metadata（`scripts/evidence_chain_dataset.py` 的 `_classify_evidence_type / _extract_chain_structure / _build_chain_text_with_metadata`）

## 12.2 推荐演进路线：最强组合

下一阶段目标是组合下列五个互相正交、各自独立可上线的增强项。详细方案文档与调研结论见对话记录归档。

### A. **MiniRAG 异构图召回**（强匹配 4B 小模型）
- 将 chunk-node 与 entity-node 放进同一张图，召回时按图传播，无需大模型抽取
- 直接受益：事实/关系类 recall@1 预期 16% → 35%+
- 切入点：复用 `indexes/arknights_story/documents.jsonl` 与 `operator_aliases.json`
- 索引层新建 `indexes/arknights_story_minirag/`，召回入口写到 `src/goldenglow/retrieval/minirag.py`，保留 `ArknightsHybridRetriever` 接口

### B. **CRAG-lite（Knowledge Refinement）**
当前项目已有 CRAG 中的 Evaluator（reranker score + conclusion next_action）与半阉的 Query Rewriting（follow_up_hypothesis 只改 keywords）。**缺的是 Knowledge Refinement**：
- 把 rerank top-k 的每个 chunk 按句子切（已有 `LINE_SPLIT_RE`）
- 对每个句子用 reranker 单独打分
- 砍掉低分句子，重组成"精炼证据包"再喂给 4B
- 切入点：`src/goldenglow/inference/cpu_pipeline.py` 在 `select_prompt_evidence` 后插入 refine 步骤
- 副作用：4B 的 context 噪声预计下降 60%+，hallucination 直接降

### C. **Self-RAG 反思 token**
让 4B 在生成时主动输出 `[Retrieve]` / `[Relevant]` / `[Supported]` / `[Useful]` 反思 token。
- 修改：教师 prompt 在 conclusion / hypothesis 输出中插入反思 token
- 重训 4B（一轮 SFT，沿用 `scripts/llama_factory/run_train.sh`）
- 完美契合现有 hypothesis / retrieval_decision / conclusion 三阶段流程

### D. **DPO reranker**
- 当前 reranker 是 pairwise softplus loss（`scripts/train_evidence_chain_reranker.py`）
- 升级到 DPO loss：`positive_score > negative_score` 转为偏好对，直接用现有 `reranker_pairwise.jsonl` + chain_structure
- 预期：listwise 排序更稳，推理乱答减少（因 evidence 排序质量提升）

### E. **推理时 MMR + Self-Consistency**
零训练成本，立刻可上线：
- **MMR**：把当前 `select_prompt_evidence` 改成 MMR，平衡 relevance 与 diversity，避免 12 段 evidence 都讲一句话的不同变体
- **Self-Consistency**：conclusion 阶段同一 query 跑 5 次（temperature 0.7），投票选 next_action；只对 conclusion 做，推理变慢约 5×（可接受）
- **Lost-in-the-middle 缓解**：evidence 按金字塔结构排（top1 放最前、top2 放最后、top3+ 放中间）

### 落地建议
- **Sprint 1（1 周）**：E1（MMR）+ E2（Self-Consistency）+ E3（pyramid）+ B（CRAG knowledge refinement）—— 全是推理时改动，无需重训
- **Sprint 2（2 周）**：A（MiniRAG）+ D（DPO reranker）—— 中等改动，受益最大
- **Sprint 3（1 月+）**：C（Self-RAG 反思 token）—— 需要重做 SFT 数据并重训 4B

### 当前落地状态

Sprint 1 已在推理链路中落地：

- `select_prompt_evidence` 前置增强：支持 MMR evidence 选择。
- prompt 渲染前支持 CRAG-lite Knowledge Refinement：把 chunk 切成句子 strip，并用 reranker 选高分句重组证据。
- prompt evidence 支持 pyramid 顺序：top1 放前，top2 放末尾，其余放中间。
- conclusion 阶段支持 Self-Consistency：同一 prompt 多次采样，按 `next_action` 投票。

配置入口：

- `configs/runtime_inference.json`: CPU/llama.cpp 默认启用 MMR + pyramid；CRAG 与 Self-Consistency 保守关闭。
- `configs/runtime_inference_gpu.json`: GPU/vLLM 默认启用 MMR + pyramid + CRAG + 5 次 Self-Consistency。
- `api-mode/runtime_api.json`: API 模式默认启用 MMR + pyramid + CRAG + 5 次 Self-Consistency。
- CLI 覆盖入口：`scripts/run_cpu_inference.py` 支持 `--enable-mmr`、`--enable-crag-refinement`、`--enable-pyramid-order`、`--self-consistency-samples` 等参数。

Sprint 2 已完成基础落点：

- MiniRAG：新增 `src/goldenglow/retrieval/minirag.py` 与 `scripts/build_minirag_index.py`，图索引输出到 `indexes/arknights_story_minirag/graph.json`，并通过 `retrieval.enable_minirag` 接入 hybrid 召回。
- DPO reranker：`scripts/train_evidence_chain_reranker.py` 支持 `--loss-type dpo --dpo-beta 0.1`；`scripts/run_reranker_sft.sh` 支持环境变量 `LOSS_TYPE=dpo DPO_BETA=0.1`。

后续不要重复实现 Sprint 1 或 MiniRAG/DPO 的基础入口；下一步应进入 Sprint 3：Self-RAG 反思 token 数据生成与 4B 重训。

## 12.3 数据生成 Pipeline 评估（接手 AI 必读）

**核心结论**：当前 SFT 数据有训练-推理分布 mismatch 与 action 失衡问题，必须先修数据再做新方案。

### P0 缺陷（必须先修）

| ID | 问题 | 数据证据 | 影响 |
|---|---|---|---|
| P0-1 | conclusion 训练 evidence 数量太少 | median=1, 73% 样本仅 1 条 evidence，线上喂 12 段 | 4B 在长 context 下能力下降 |
| P0-2 | clarify_user 样本几乎不存在 | 1089 条 conclusion 中仅 7 条（0.6%）clarify_user；97 条含代词题里 0 走 clarify | 代词题/重名题一律硬答 |
| P0-3 | retrieval action 严重失衡 | retrieve_more 70.5% / answer_directly 27.1% / abstain 1.8% / clarify_user 0.6% | 教师默认 strategy 偏向硬答 |
| P0-4 | 教师只见 top-3 chunk | `configs/sft_teacher_generation.json:max_evidence_docs_per_request=3` | 教师 evidence 视野与线上不一致 |

### P1 缺陷（建议同步修）

| ID | 问题 | 数据证据 |
|---|---|---|
| P1-1 | hypothesis keywords 含问句词噪声 | 380 / 1500 个 hypothesis 的 keywords 含 `什么/为什么/关系/身份/原因` |
| P1-2 | hypothesis entity 只 1 个偏多 | 53% 样本只有 1 个 entity；relation 题需要 ≥ 2 |
| P1-3 | style/knowledge 类暴露检索过程 | 46 条 canon_qa / persona_grounded_qa 含 "根据证据 / 根据检索结果" 等暴露词 |
| P1-4 | reranker shuffled_order score 太高 | median=0.95，与 gold(1.0) 只差 0.05 → 梯度不足，学不到顺序敏感性 |
| P1-5 | reranker 用合成 negative 而非 in-domain hard negative | 当前 background_only / answer_adjacent 都是教师从原文挑的，不是真实 retriever top-K 里的混淆 chunk |

### 推荐修复顺序（最高 ROI 三件事）

1. **改 `configs/sft_teacher_generation.json`** 把 `max_evidence_docs_per_request` / `retrieval_top_k` 从 3 提到 12，让训练分布对齐线上
2. **改教师 prompt**（`src/goldenglow/data/sft_teacher.py` 的 `build_conclusion_prompt_bundle`）：
   - 显式注入歧义触发器，让 clarify_user 占比 ≥ 12%
   - 显式注入证据不足场景，让 abstain 占比 ≥ 10%
3. **改教师 prompt**（同文件 `build_initial_hypothesis_prompt_bundle / build_follow_up_hypothesis_prompt_bundle`）：
   - keywords 黑名单（禁问句词）
   - entities ≥ 2 强制（多实体桥接）
   - 强制让教师调用 `operator_aliases.json` 展开别名

### 长期修复

- **数据反馈回路**：`scripts/evaluate_retrieval_recall.py` 的 missed_queries.jsonl 应回流到 `generate_sft_from_teacher.py`（新增 `--seed-queries` 参数）
- **自动质量检测**：新增 `scripts/audit_sft_data.py`，对生成的 SFT 数据做分布校验（clarify_user 比例 / 暴露词 / 问句词 keywords 比例），不通过则阻断 merge
- **教师 reasoning 蒸馏**：教师 prompt 输出 `[REASONING]...[/REASONING][OUTPUT]{json}[/OUTPUT]` 两段，让 4B 学到推理过程而不仅是 JSON 形式
- **三元组数据收集**：教师同时输出 `(head, relation, tail, source_evidence_id)`，为 MiniRAG / GraphRAG 攒图谱原料

## 13. 常用验证命令

Python 语法检查：

```bash
/home/zhb/miniconda3/envs/train/bin/python -m py_compile \
  src/goldenglow/inference/cpu_pipeline.py \
  src/goldenglow/retrieval/hybrid.py \
  scripts/run_cpu_inference.py \
  api-mode/run_api_inference.py
```

JSON 配置检查：

```bash
/home/zhb/miniconda3/envs/train/bin/python -m json.tool configs/runtime_inference_gpu.json >/tmp/runtime_gpu.checked
/home/zhb/miniconda3/envs/train/bin/python -m json.tool api-mode/runtime_api.json >/tmp/runtime_api.checked
```

检索链路调试：

```bash
python scripts/query_retrieval.py "炎景公主一事具体指什么"
```

API 模式端到端：

```bash
export ARK_API_KEY="..."
CUDA_VISIBLE_DEVICES=1 \
/home/zhb/miniconda3/envs/train/bin/python api-mode/run_api_inference.py \
  --runtime-config api-mode/runtime_api.json \
  "炎景公主一事具体指什么"
```

端到端召回评估（必须用 GPU + train env）：

```bash
# baseline 索引备份在 indexes/.baseline_arknights_story/
cp -r indexes/arknights_story indexes/.baseline_arknights_story  # 改索引前先备份

CUDA_VISIBLE_DEVICES=1 /home/zhb/miniconda3/envs/train/bin/python \
  scripts/evaluate_retrieval_recall.py \
  --output outputs/retrieval_eval/<tag>.json \
  --device cuda \
  --tag <tag_for_run>
# gold 数据：data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/reranker_listwise.jsonl
# 命中判定：字符 trigram Jaccard ≥ 0.25
# 输出含 overall + by_query_type 的 recall@1/5/10/20 与 MRR
```

任何改 hybrid retrieval / reranker / hypothesis 的变更**完成后必须**跑 baseline vs improved 对比，对比表写到 `outputs/retrieval_eval/comparison_report.md`。

## 14. 开发工作方式

AI 修改本项目时必须：

- 先读现有代码和配置，再修改。
- 优先做可运行、可验证的小改动。
- 不要回滚用户已有修改。
- 不要引入不可观测的黑箱流程。
- 路径、模型、阈值优先配置化。
- 变更检索、prompt、数据清洗或训练脚本后，至少做语法检查和相关配置检查。
- 如果某个问题需要模型能力提升，优先从数据、prompt、训练目标和评估集入手。
- **本项目所有教师数据（SFT + reranker）均由 Minimax 教师模型生成，没有人工标注层**。提改进方案时优先改教师 prompt / 教师生成约束 / 教师产物质量过滤，而不是"加人工"。
- 凡涉及检索 / reranker / hypothesis / 召回评估的工作，**完成后必须**跑 `scripts/evaluate_retrieval_recall.py` 做 before/after 对比，不要只靠单 query 调试。

禁止：

- 硬编码某个问题、角色、活动或答案。
- 跳过检索直接让模型凭记忆回答剧情事实。
- 把没有证据的推断写成官方设定。
- 为了单个 bad case 增加全局启发式规则。
- 把 API key、私有 endpoint token 或本地绝对密钥路径提交到仓库。
- 在没跑 recall 评估的情况下声称"召回更好"。

## 14.1 接手 AI 工作指引

如果你是新接手本项目的 AI，请按下列顺序消化背景：

1. **必读章节**：§1（项目定位）→ §2（架构）→ §12.1（当前痛点）→ §12.2（最强组合方案）→ §12.3（数据 pipeline 评估）
2. **必看产物**：`outputs/retrieval_eval/baseline.json`、`outputs/retrieval_eval/comparison_report.md`、`data/processed/sft_data/teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_teacher_v1/stats.json`
3. **必跑命令**：在改任何东西前先跑一次 `scripts/evaluate_retrieval_recall.py` 重现 baseline，确保环境正常
4. **推荐起步**：按 §12.2 的 Sprint 1（MMR + Self-Consistency + pyramid + CRAG knowledge refinement）顺序做，这些都是推理时改动，不需要重训，风险最低
5. 任何方案落地前先按 §14 第 8 条做 before/after 评估对比，写到 `outputs/retrieval_eval/comparison_report.md`，再交付

## 15. 一句话决策规则

当实现方案有取舍时，默认选择：

```text
更忠于剧情证据、更少硬编码、更容易追踪调试、更能通过数据和模型泛化的方案。
```
