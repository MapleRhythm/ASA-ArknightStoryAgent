# ASA-ArknightStoryAgent 开发代理指南

本文件给接手仓库的 AI / 开发代理使用。项目是《明日方舟》剧情问答 RAG Agent，核心要求是：**答案必须可追踪到检索证据，不能把模型记忆、猜测、二创设定或百科误读写成官方剧情事实。**

## 1. 项目定位

- 任务：中文剧情问答、因果解释、人物关系、事件真相、时间线、多轮追问消解。
- 本地模型：Qwen3.5-4B + LoRA，负责结构化 hypothesis / conclusion / follow-up 与最终答案。
- API teacher：DeepSeek / OpenAI-compatible API，用于对照评测、数据生成和 SODA 蒸馏。
- 检索系统：剧情原文、档案、语音、元数据、MiniRAG 图、web context 共同构成证据池。
- 约束：证据够就回答可确认部分，证据不足就明确不足；不要为了“看起来聪明”补不存在的设定。

## 2. 当前主架构

核心代码：

- `src/goldenglow/config.py`：路径与 `QueryConfig`。
- `src/goldenglow/retrieval/hybrid.py`：dense / sparse / MiniRAG / reranker 混合检索。
- `src/goldenglow/retrieval/minirag.py`：MiniRAG 图召回与章节 scope。
- `src/goldenglow/retrieval/storyline.py`：故事线标签与 sparse scope。
- `src/goldenglow/inference/cpu_pipeline.py`：本地与 API 共用的多轮 RAG pipeline。
- `api-mode/run_api_inference.py`：API 生成器入口，只替换 generator，不替换检索链路。

标准链路：

```text
question
-> user_question_hypothesis_generation
-> build retrieval queries
-> dense + BM25 + MiniRAG
-> MiniRAG chapter isolation / graph expansion / scoped second retrieval
-> optional storyline sparse scope / neighbor expansion / web context
-> fusion + reranker
-> prompt evidence selection: dedupe / source weighting / MMR / pyramid / pinning
-> conclusion_generation
-> answer_directly | retrieve_more | clarify_user | abstain
-> follow_up_hypothesis_generation if needed
```

当前最大检索轮次由 `inference.max_retrieval_rounds` 控制，但核心链路会 clamp 到最多 `2` 轮。达到轮次上限后，若当前证据可支持部分回答，应输出可确认部分；否则输出证据不足说明，不要机械返回“达到检索轮次上限”。

## 3. 运行配置

主要 runtime：

- `configs/runtime_inference_gpu.json`：本地 GPU / vLLM 链路。
- `configs/runtime_inference.json`：本地轻量链路。
- `api-mode/runtime_deepseek_api.json`：DeepSeek API teacher 链路。
- `api-mode/runtime_api.json`：通用 API 模式配置。

关键字段：

- `retrieval.reranker_model_path`
- `retrieval.minirag_index_path`
- `retrieval.minirag_chapter_isolation`
- `retrieval.minirag_auto_second_retrieval`
- `retrieval.enable_storyline_sparse_scope`
- `retrieval.enable_neighbor_expansion`
- `inference.max_retrieval_rounds`
- `inference.prompt_evidence_top_k`
- `inference.prompt_evidence_max_chars_per_doc`
- `inference.prompt_conclusion_evidence_max_total_chars`
- `inference.conclusion_prompt_mode`
- `inference.answer_grounding_mode`
- `inference.web_context`

本地与 API 模式应尽量使用同一套 retrieval / inference 配置。不要只在 API mode 修好，再让本地链路继续跑旧配置。

## 4. 环境约定

默认使用 `train` 环境进行训练、GPU 检索、vLLM 推理和 API 调试：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate train
export PYTHONPATH=.python_packages/train:src
```

常用环境变量：

```bash
export DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CONDA_NO_PLUGINS=true
```

训练默认 3 卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2
```

单卡 vLLM / 检索调试：

```bash
CUDA_VISIBLE_DEVICES=0
```

启动大模型前必须看 `nvidia-smi`。不要在已有训练 / vLLM 进程占满显存时强行拉起新 engine。

## 5. API Mode 规则

`api-mode/run_api_inference.py` 已对齐本地链路，支持 MiniRAG 章节隔离、故事线 scope、web context、MMR、pyramid、evidence pinning 和 conclusion prompt mode。

API teacher prompt 分两类：

- JSON 任务：附加 `API_MODE_SYSTEM_APPENDIX`，强调 schema、字段、证据边界、不要过度 abstain。
- direct-answer 任务：附加 `API_MODE_QA_SYSTEM_APPENDIX`，避免最终答案被强制输出 JSON。

DeepSeek 注意事项：

- 环境变量用 `DEEPSEEK_API_KEY`。
- key 应为纯 `sk-...`；如果外部记录成 `ds:sk-...`，脚本外运行时先去掉 `ds:`。
- reasoning 模型可能返回空 `content` 但有 `reasoning_content`；API runner 已做一次重试，要求最终内容写入 `content`。

API 模式示例：

```bash
export DEEPSEEK_API_KEY="sk-..."

PYTHONPATH=.python_packages/train:src \
python api-mode/run_api_inference.py \
  --runtime-config api-mode/runtime_deepseek_api.json \
  --answer-only \
  "澄闪在卡拉顿城识破的阴谋是什么"
```

## 6. 检索与证据规则

检索优化原则：

- 不要为单个 bad case 写角色、活动、问题特化逻辑。
- 不要用硬编码关键词强行改排序结果。
- 召回和排序问题优先通过 query schema、reranker 数据、MiniRAG/storyline scope、数据清洗解决。
- 必须保留 trace、分数、来源、doc id，便于复盘。

当前常见噪声：

- 泛词召回：问题词如“为什么/是什么/关系/事件”压过实体词。
- 百科/档案压过剧情原文：可降权但不要默认删除，百科在缺官方直证时仍有辅助价值。
- 图扩展串章节：MiniRAG 应优先活动级/章节级 scope，第二轮候选也应继承 scope，除非置信度不足。
- 过度拒答：结论层要学会输出“可确认部分”，而不是缺全貌就 abstain。
- 过早回答：本地 4B 首轮对原因/真相/危机/作用机制等复杂题可能直接答错并截断召回；优先通过 SODA/KTO 的负例让模型自己学会 `retrieve_more` 边界，不要在 runtime 中硬编码补检索文档。

## 7. MiniRAG 与故事线

MiniRAG 图没有把关系永久切成单独图文件；runtime 主要从 evidence chunk 的 metadata 推导章节 / 活动 scope。

当前策略：

- MiniRAG：强活动/章节隔离，第一轮命中章节后进行图扩展和 scoped second retrieval。
- BM25：可按故事线做 sparse scope，避免泛词把其他主线或活动拉进来。
- dense：默认不强做故事线隔离，避免语义召回被过早截断。

故事线 scope 应作为召回约束，不应写成答案逻辑。

## 8. Web Context

`inference.web_context` 可在第一轮召回后启用网络剧情解析作为辅助证据。

原则：

- web context 进入证据池前需要 rerank / threshold。
- “萌百”等来源不要默认剔除，但可降权。
- web context 不能覆盖官方剧情原文；官方原文足够时优先原文。
- 不要把网页摘要直接当官方事实，答案中应保守表述。

## 9. Prompt 与结构化输出

当前 tool task：

- `user_question_hypothesis_generation`
- `follow_up_hypothesis_generation`
- `conclusion_generation`

允许的 conclusion action：

- `answer_directly`
- `retrieve_more`
- `clarify_user`
- `abstain`

修改 prompt 时注意：

- hypothesis 只写检索线索，不要写答案。
- conclusion JSON 字段必须固定，禁止额外字段污染训练：`confidence`、`decision`、`slot_values`、`follow_up_question` 等。
- `retrieve_more` 必须有具体可检索的 `missing_slots` 和可用 follow-up hypothesis。
- `clarify_user` 只用于用户问题歧义，不用于普通证据不足。
- 不要要求模型输出 chain-of-thought。

## 10. 本地 4B 训练路线

当前顺序：

1. SFT：学会 runtime JSON 协议、检索意图、结论动作边界。
2. API-grounded conclusion 数据：补复杂剧情问答和可确认部分回答。
3. KTO / SODA：用当前学生在线错误作为负样本，API teacher 输出作为正样本，修过度拒答、JSON 漂移和错误 action。

不要用 SODA 直接从原始基座开始做最终版。若要复现“从基座开始”，应先做 teacher-positive SFT，再做 SODA/KTO。

当前 SODA 入口：

- `scripts/generate_soda_blackbox_distillation.py`
- `scripts/run_soda_blackbox_distill.sh`
- `src/config/llama_factory_soda_blackbox_deepseek_v1_config.yaml`

推荐规模：

- smoke：`SODA_SAMPLE=50`
- 第一版正式：`SODA_SAMPLE=300`
- 第二版增强：`SODA_SAMPLE=500`
- 不建议未经抽样直接上千。

命令：

```bash
export DEEPSEEK_API_KEY="sk-..."

SODA_SAMPLE=300 \
GEN_CUDA_VISIBLE_DEVICES=0 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 \
bash scripts/run_soda_blackbox_distill.sh
```

## 11. Reranker 路线

证据链 reranker 训练目标是“证据是否足以回答问题”，不是关键词匹配。

相关脚本：

- `scripts/build_rank_mix_reranker_dataset.py`
- `scripts/build_rank_clean_reranker_dataset.py`
- `scripts/train_evidence_chain_reranker.py`
- `scripts/evaluate_evidence_chain_reranker.py`

训练参数使用当前脚本真实参数名：

- `--model-name-or-path`
- `--train-file`
- `--output-dir`
- `--num-train-epochs`
- `--per-device-train-batch-size`
- `--learning-rate`
- `--max-length`
- `--loss-type softplus|dpo`

不要使用旧参数名如 `--base-model`、`--epochs`、`--batch-size`。

## 12. 常用验证

语法检查：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate train

PYTHONPATH=.python_packages/train:src python -m py_compile \
  src/goldenglow/inference/cpu_pipeline.py \
  src/goldenglow/retrieval/hybrid.py \
  api-mode/run_api_inference.py \
  scripts/run_cpu_inference.py \
  scripts/generate_soda_blackbox_distillation.py
```

API smoke：

```bash
PYTHONPATH=.python_packages/train:src \
python api-mode/run_api_inference.py \
  --runtime-config api-mode/runtime_deepseek_api.json \
  --max-retrieval-rounds 1 \
  --answer-only \
  "岁兽是什么"
```

多轮召回评测：

```bash
PYTHONPATH=.python_packages/train:src \
CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_multiround_retrieval_recall.py \
  --runtime-config configs/runtime_inference_gpu.json \
  --output outputs/eval_multiround_retrieval/current_sample50.json \
  --sample 50 \
  --max-rounds 2 \
  --device cuda
```

任何声称“召回更好 / 链路更好 / 拒答率下降”的改动，都需要至少跑固定问题集或 sample50 对比。

## 13. 开发禁区

禁止：

- 硬编码某个问题、角色、活动、答案或关键词修复。
- 跳过检索直接让模型凭记忆回答剧情事实。
- 把 API key、私有 endpoint token 或密钥路径写进仓库。
- 在未记录 trace 的情况下修改排序 / 过滤逻辑。
- 为了降低拒答率关闭 grounding 校验。
- 用没有抽样检查的数据直接训练大轮次。

允许但必须配置化：

- 调整 MiniRAG / BM25 / dense / reranker 的权重。
- 调整 prompt evidence 选择策略。
- 调整 teacher prompt 的全局行为边界。
- 调整 SODA 正负样本比例与筛选规则。

## 14. 接手顺序

新接手本项目时，按顺序看：

1. `configs/runtime_inference_gpu.json`
2. `api-mode/runtime_deepseek_api.json`
3. `src/goldenglow/inference/cpu_pipeline.py`
4. `src/goldenglow/retrieval/hybrid.py`
5. `src/goldenglow/retrieval/minirag.py`
6. `scripts/generate_soda_blackbox_distillation.py`
7. `outputs/eval_multiround_retrieval/` 里的最近评测结果

默认决策规则：

```text
更忠于剧情证据、更少硬编码、更可观测、更能通过数据和模型泛化的方案优先。
```
