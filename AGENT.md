# Goldenglow 项目 AI 开发指南

本文件用于指导任何 AI 开发代理在本仓库内实现、修改和扩展项目。

项目定位：这是一个面向《明日方舟》剧情问答的本地 Agent。目标不是做泛知识聊天，而是做“基于剧情证据的中文问答”，并在回答风格上注入澄闪的语气，但不能为了语气牺牲事实准确性。

## 1. 不可偏离的技术约束

除非用户明确要求替换，否则默认技术栈固定为：

- 主对话模型：`Qwen3.5-4B`
- 主模型微调方式：`LoRA`
- LoRA 训练框架：本地 `LLaMA-Factory`
- 向量编码模型：`bge-small-zh-v1.5`
- 向量检索：`FAISS-CPU`
- 推理框架：`llama.cpp`
- 检索调度与工具编排：原生 `Python`

补充约束：

- CPU / 兼容性优先的离线推理主路径仍然是 `llama.cpp`
- `train` 环境下允许额外挂接 `vLLM` 作为 GPU 推理加速后端
- 若 `vLLM` 与训练依赖冲突，优先通过仓库内 overlay 目录 `.python_packages/train` 安装，而不是破坏现有训练环境

开发时不要擅自将核心方案替换为：

- 远程 API 推理
- 重型工作流框架
- LoRA 训练框架替换（如 `Axolotl`、`DeepSpeed`）
- Elasticsearch / Milvus / Chroma 等替代 FAISS 的主方案
- Transformers 作为线上主推理框架

`LLaMA-Factory` 约束补充：

- 训练环境必须本地部署执行，不使用远程托管训练服务
- 与 `Qwen3.5-4B` 的训练模板、分词与权重导出格式必须保持兼容，确保训练后的 LoRA 能挂载到 `llama.cpp` 推理链路
- 若出现训练便利性与推理兼容性冲突，优先保证 `llama.cpp` 挂载可用性

可以做的事：

- 为 `llama.cpp` 增加转换、量化、推理封装
- 用 Python 增加 BM25 / 关键词检索分支，形成混合检索
- 预留可配置的交叉编码器重排接口
- 增加离线评测、索引构建、数据清洗脚本

## 2. 仓库现状

当前仓库中已知的关键目录：

- `data/`：剧情与游戏数据
- `data/ArknightsGameData/zh_CN/gamedata/story/`：剧情原文 `.txt`
- `data/ArknightsGameData/zh_CN/gamedata/excel/`：章节、活动、角色等元数据 `.json`
- `data/processed/sft_data/`：SFT 主数据集、补充中间能力数据集与合并后的训练数据
- `model/qwen3.5-4b/`：Qwen 基座模型权重
- `model/lora/`：LoRA 训练输出目录
- `model/gguf/`：推理使用的 GGUF 模型
- `indexes/arknights_story/`：检索索引、文档与 BM25 产物
- `configs/`：运行时配置，例如 `runtime_inference.json`
- `src/config/`：训练配置，例如 `llama_factory_config.yaml`
- `scripts/`：原生 Python 脚本、训练脚本与评测脚本
- `scripts/llama_factory/`：LLaMA-Factory 数据转换、训练与评测入口
- `scripts/run_cpu_inference.py`：当前实际推理入口
- `src/goldenglow/inference/cpu_pipeline.py`：当前在线推理主链路
- `third_party/llama.cpp/`：本地推理依赖

AI 在实现功能时，应优先围绕这些路径工作，不要假定仓库存在未出现的 Web 服务、数据库或前端。

当前仓库的主流程已经具备以下可运行入口：

- 检索索引构建：`scripts/build_retrieval_index.py`
- 主 SFT 数据生成：`scripts/generate_sft_from_teacher.py`
- 补充中间能力数据生成：`scripts/generate_prompt_supplement_from_teacher.py`
- 数据集合并：`scripts/merge_sft_datasets.py`
- LLaMA-Factory 训练：`scripts/llama_factory/run_train.sh`
- CPU 全流程推理：`scripts/run_cpu_inference.py`
- GPU 全流程推理：`scripts/run_gpu_inference.sh`
- 检索延迟测试：`scripts/benchmark_retrieval_latency.py`

其中主 SFT 生成与 supplement 生成在 tool 类样本上必须统一使用以下三种 `task_type`：

- `user_question_hypothesis_generation`
- `follow_up_hypothesis_generation`
- `conclusion_generation`

## 3. 项目目标

这个 Agent 需要完成以下链路：

1. 接收用户问题
2. 结合多轮对话上下文生成首份“假设文档”
3. 用假设文档驱动首轮混合检索与重排
4. 判断当前证据是否足够直接回答
5. 若证据不足，则基于“原问题 + 多轮对话上下文 + 当前假设文档 + 当前证据 + 当前未解点 + 前几轮检索历史”生成补充检索假设文档
6. 用新的补充检索假设文档执行下一轮检索，并继续判断是否还需要补充检索
7. 将前几轮的检索上下文持续带入后续 planner / follow-up hypothesis 生成
8. 整个检索循环默认最多执行 3 轮总检索
9. 在证据足够时生成最终剧情答案
10. 在表达层面体现澄闪的语气

这里的“假设文档”不是最终答案，而是检索中间产物。它的职责是把自然语言问题拆解成更适合召回的结构化检索线索。

## 4. 回答原则

这是剧情问答 Agent，不是自由发挥型角色扮演 Bot。始终遵守以下原则：

- 优先事实，其次语气
- 优先证据，其次印象
- 优先原文剧情，其次表格元数据
- 先回答“剧情里明确写了什么”，再补充“合理推断是什么”

回答时必须尽量区分三类信息：

- 明确剧情事实
- 基于多段证据的归纳
- 无法确认的推测

如果证据不足，必须明确说“不确定”或“现有检索证据不足以支持这个结论”，不要硬编。

## 5. 澄闪语气注入规则

LoRA 的目标之一是注入澄闪的语气，但语气层必须服从事实层。

澄闪风格建议：

- 语气轻柔、礼貌、略带犹豫感
- 更像认真解释剧情，而不是夸张卖萌
- 可以有亲近感，但不要过度口癖化
- 不要把角色语气写成二创段子或强行撒娇

禁止出现的偏差：

- 为了“像澄闪”而改变剧情事实
- 把不确定内容说得像官设
- 用角色口吻掩盖检索不足

如果风格 LoRA 与事实能力冲突，优先保证事实能力。必要时可拆分为多个适配器，例如：

- `style`：仅负责澄闪表达风格
- `tool` 或 `dialogue`：负责意图识别、工具调用、多轮对话格式
- `knowledge`：补充少量高质量、人工确认过的知识

## 6. 数据使用规则

剧情问答的数据源优先级如下：

1. `data/ArknightsGameData/zh_CN/gamedata/story/` 下的剧情原文
2. `data/ArknightsGameData/zh_CN/gamedata/excel/story_table.json`
3. `data/ArknightsGameData/zh_CN/gamedata/excel/story_review_table.json`
4. `data/ArknightsGameData/zh_CN/gamedata/excel/story_review_meta_table.json`
5. `data/ArknightsGameData/zh_CN/gamedata/excel/stage_table.json`
6. `data/ArknightsGameData/zh_CN/gamedata/excel/chapter_table.json`
7. `data/ArknightsGameData/zh_CN/gamedata/excel/character_table.json`
8. 其他与剧情定位有关的元数据表

实现数据清洗时必须注意：

- 原始剧情 `.txt` 含有舞台控制指令、音效、镜头、角色切换等标记
- 需要从脚本中提取“可检索文本”与“结构化事件”
- 不要粗暴按行切割后直接入库
- 不要丢失说话人、关卡编号、活动代号、剧情段落位置

建议至少保留以下元数据字段：

- `source_path`
- `story_id`
- `stage_id`
- `stage_name`
- `chapter_id`
- `chapter_name`
- `activity_id`
- `activity_name`
- `speaker`
- `segment_type`
- `sequence_id`
- `raw_text`
- `clean_text`

## 7. 文本解析与切块原则

剧情切块不能只按 token 数硬切，必须保留叙事结构。

优先策略：

- 先解析成“剧情段”或“对话块”
- 再按语义窗口做二次切块
- 保留相邻上下文重叠

建议规则：

- 不拆开同一角色连续发言
- 尽量让一个 chunk 对应一个完整的小情节、问答片段或叙述单元
- 对特别长的段落再做窗口化切分
- chunk 之间保留少量 overlap，避免问答跨段断裂

如果需要多级检索，可同时建立：

- `scene-level` 索引：较大颗粒度，利于找背景
- `utterance/block-level` 索引：较细颗粒度，利于精确问答

## 8. 推荐系统架构

推荐采用如下模块化架构：

1. `data_ingest`
2. `parser`
3. `chunker`
4. `embedder`
5. `faiss_index`
6. `sparse_retriever`
7. `hybrid_retriever`
8. `reranker`
9. `dialogue_state`
10. `intent_classifier`
11. `hypothesis_builder`
12. `answer_generator`
13. `evaluation`

最关键的在线链路如下：

1. 用户问题进入对话状态管理器
2. 基于用户问题与多轮对话上下文生成初始假设文档
3. 用假设文档同时触发稠密检索与稀疏检索
4. 用融合策略合并候选
5. 对 Top-N 使用交叉编码器重排
6. 让 retrieval planner 判断当前证据是否足够直接回答
7. 若当前证据不足，则基于“原问题 + 多轮对话上下文 + 当前假设文档 + 当前证据 + 当前未解点 + 历史检索摘要”生成补充检索假设文档
8. 用新的补充检索假设文档再次召回，并把前几轮检索历史继续注入下一轮 planner
9. 整个检索过程默认最多执行 3 轮总检索；达到上限后不再继续扩展检索
10. 组织最终证据上下文
11. 让 Qwen 生成最终回答

## 9. 假设文档要求

假设文档是本项目的关键中间结构。它应当显式包含：

- 用户原问题
- 意图类型
- 多轮上下文中解析出的指代对象
- 可能涉及的角色名、组织、地点
- 主题关键词
- 预期回答类型

当前实现中，假设文档优先使用精简 schema，不要求填充过多低价值字段。推荐输出为结构化 JSON，而不是自由文本。例如：

```json
{
  "question": "缪尔赛思为什么会帮助博士？",
  "intent": "plot_reasoning",
  "entities": ["缪尔赛思", "博士"],
  "keywords": ["帮助", "动机", "合作", "原因"],
  "expected_answer_type": "原因/动机"
}
```

假设文档必须服务于检索，不要把它写成华丽提示词。

允许的可选字段：

- `aliases`
- `dialogue_context`

不建议默认强制要求的字段：

- `possible_arcs`
- `exclude_terms`

原因是这些字段在 API 蒸馏时更容易引入猜测性噪声，不如优先保证 `entities`、`keywords` 与 `expected_answer_type` 的稳定性。

当需要继续检索时，推荐使用联合决策输出，而不是把“是否继续检索”和“follow-up hypothesis”完全拆散。例如：

```json
{
  "question": "烛煌的真实身份是什么？",
  "next_action": "retrieve_more",
  "missing_slots": ["太师是谁", "烛煌与太师的关系"],
  "clarification_question": "",
  "follow_up_hypothesis": {
    "question": "烛煌的真实身份是什么？",
    "intent": "plot_fact",
    "entities": ["烛煌", "太师"],
    "keywords": ["烛煌", "太师", "身世", "太师是谁", "烛煌 太师 什么关系"],
    "expected_answer_type": "身份关系"
  }
}
```

## 10. 检索策略

检索链路必须采用混合检索，而不是单一路径。

最低要求：

- 稠密检索：`bge-small-zh-v1.5` + `FAISS-CPU`
- 稀疏检索：BM25、关键词匹配或等价轻量实现
- 融合策略：可采用加权归一化或 `RRF`
- 重排：交叉编码器 reranker

注意：

- 交叉编码器模型暂未在仓库中固定，必须做成配置项，不要把代码写死到某个 Hugging Face 名称
- 如果仓库后续新增 reranker 模型，应默认放在 `model/` 下可配置路径
- 检索分数、融合分数、重排分数应保留到日志中，便于诊断

建议在线检索顺序：

1. 用原问题检索一次
2. 用假设文档扩展后的 query 再检索一次
3. 合并候选
4. 重排
5. 让 planner 输出 `next_action`
6. 若 `next_action=retrieve_more`，则基于“原问题 + 对话上下文 + 当前假设文档 + 当前证据 + 未解点 + 历史检索摘要”生成补充检索 hypothesis，并执行下一轮检索
7. 重复 5-6，并在每一轮把前几轮检索历史继续带入 planner / hypothesis 生成
8. 当 planner 返回 `answer_directly` / `clarify_user` / `abstain` 时结束；若达到总检索轮次上限，也必须停止扩展检索
9. 截取高可信证据送入生成

当前运行时代码中的相关配置通过 `configs/runtime_inference.json` 控制，至少包括：

- `retrieval.device`
- `retrieval.enable_reranker`
- `retrieval.dense_top_k`
- `retrieval.sparse_top_k`
- `retrieval.fusion_top_k`
- `retrieval.rerank_top_k`
- `retrieval.rerank_batch_size`
- `inference.max_retrieval_rounds`

## 11. Function Calling 设计原则

多轮对话、意图识别和 RAG 调用需要走 function calling，但工具调用本身必须由 Python 控制，不要把检索细节交给模型自由发挥。

推荐工具边界：

- `detect_intent`
- `build_hypothesis`
- `retrieve_dense`
- `retrieve_sparse`
- `merge_candidates`
- `rerank_candidates`
- `compose_context`
- `generate_answer`

要求：

- 工具输入输出使用稳定 JSON schema
- Python 负责执行工具
- 模型只负责决定“何时调用”和“填哪些参数”
- 检索结果、重排结果、最终上下文都要可追踪

不要让主模型直接：

- 伪造检索结果
- 口头声称“已检索”但没有实际工具执行
- 自行编造 source id 或剧情出处

## 12. 多轮对话规则

多轮对话至少要维护以下状态：

- 最近讨论的角色
- 最近讨论的章节或活动
- 最近一次明确问题的核心主题
- 指代消解结果，例如“她”“那件事”“这里”
- 用户是否在继续追问同一话题
- 当前检索轮次与前几轮检索历史摘要
- 前几轮检索中尚未解决的 `missing_slots`

当用户提问模糊时：

- 如果上下文足够，直接消解并继续回答
- 如果歧义过大，先澄清再检索

不要在以下情况下硬答：

- “她”可能指多个角色
- 同名事件或相似活动可能混淆
- 时间线前后冲突明显

## 13. 意图识别范围

至少支持以下意图类型：

- `plot_fact`
- `plot_reasoning`
- `timeline`
- `character_relation`
- `event_summary`
- `compare`
- `follow_up`
- `clarification_needed`
- `persona_chat`
- `out_of_scope`

意图识别结果应直接影响：

- 是否检索
- 检索范围
- 是否需要澄清
- 最终回答模板

## 14. 生成层规则

最终生成答案时应遵守：

- 先给结论，再给依据
- 尽量引用检索到的关键证据
- 如果结论来自多段归纳，要说明是归纳而非单句原文
- 有歧义时要点明版本或剧情阶段

推荐答案结构：

1. 简短直接回答
2. 证据说明
3. 如有必要，补充背景
4. 如证据不足，明确保留

语气可以偏澄闪，但内容组织必须清楚、克制。

## 15. llama.cpp 相关约束

线上或本地主推理路径默认使用 `llama.cpp`。

因此实现时要考虑：

- 模型可能需要转换为 `GGUF`
- LoRA 挂载需要兼容 `llama.cpp`
- function calling / tool calling 的提示格式要与所用 chat template 对齐
- 不要先用一个依赖很重的 Transformers 服务再包一层 Python，当作正式方案

可以接受的辅助用途：

- 用其他脚本做离线数据处理
- 用训练脚本生成 LoRA 权重
- 用转换脚本把基座模型转成推理所需格式

## 16. LoRA 数据与训练原则

LoRA 数据集应至少覆盖三类样本：

- 澄闪风格表达样本
- 高质量剧情知识样本
- function calling / 意图识别 / 多轮对话样本

在当前流程中，还应额外补充“中间能力”样本，至少覆盖以下三类：

- `user_question_hypothesis_generation`
- `follow_up_hypothesis_generation`
- `conclusion_generation`

SFT 数据集构建要求：

- 监督微调（SFT）样本默认通过在线模型 API 蒸馏生成，不直接用人工随意编写问答替代
- 蒸馏流程需保留输入提示、模型输出、清洗规则与版本信息，便于追溯与复现

蒸馏模型与流程要求：

- 在线蒸馏 API 默认使用 `Doubao-Seed-2.0-lite`；若任务准确性不足，可切换到同级高质量模型（如 Qwen 官方 API）并记录变更原因
- 标准流程为：剧情原文输入 -> 提示词设计 -> API 调用蒸馏 -> 输出清洗 -> 样本校验
- 蒸馏样本落盘到 `data/processed/sft_data/`，并按 `style` / `knowledge` / `tool` 分类存储
- 必须进行人工抽样校验，重点检查是否忠于剧情原文、是否引入幻觉、是否错误强化二创设定

当前仓库中的数据生成流程分为两层：

- 主数据集：`scripts/generate_sft_from_teacher.py`
- 补充中间能力数据集：`scripts/generate_prompt_supplement_from_teacher.py`

两层数据在 tool 标签上必须完全对齐，只允许使用：

- `user_question_hypothesis_generation`
- `follow_up_hypothesis_generation`
- `conclusion_generation`

补充数据集的职责不是替代主 SFT 数据，而是专门增强：

- 是否继续检索
- 是否要求用户澄清
- 在需要继续检索时生成 follow-up hypothesis
- 在多轮检索中利用“前几轮检索历史 + 当前未解点”生成新的补充检索 hypothesis

补充数据集应尽量覆盖以下运行时形态：

- 首轮检索后直接回答
- 首轮检索证据不足，进入第 2 轮补充检索
- 第 2 轮仍证据不足，带着前两轮检索历史进入第 3 轮补充检索
- 达到最多 3 轮总检索后停止继续扩展

当前默认建议直接运行 `scripts/generate_prompt_supplement_from_teacher.py`，该脚本在生成 supplement 后会自动调用合并逻辑产出主训练集；如只想单独生成 supplement，可显式加 `--skip-merge`。

如需单独重跑合并，仍可直接使用：

- `scripts/merge_sft_datasets.py`

当前主训练数据集建议使用：

- `data/processed/sft_data/teacher_v2_plus_prompt_supplement_v2`

补充数据集中的字段应尽量精简，避免把“答案结论”直接泄漏进 hypothesis 或 decision 样本。

LoRA 训练执行要求：

- LoRA 训练默认调用本地 `LLaMA-Factory`，不要替换为远程训练服务
- 训练配置（数据路径、模板、超参数、输出目录）应以可复现方式落盘管理
- 当前主训练入口为 `scripts/llama_factory/run_train.sh`
- `scripts/transformers_peft/` 保留为兼容/调试路径，不是主训练链路

`LLaMA-Factory` 配置与产物要求：

- 训练配置文件默认放置在 `src/config/llama_factory_config.yaml`
- 数据转换脚本默认使用 `scripts/llama_factory/prepare_sft_dataset.py`
- 默认训练目标为单机多卡（当前 3x4090）；`CPU` 路径仅用于小样本调试与流程验证，不作为主训练路径
- 配置需显式包含并版本化关键参数，例如：`model_name_or_path`、`finetuning_type=lora`、`lora_rank`、`lora_alpha`、`lora_dropout`、`per_device_train_batch_size`、`gradient_accumulation_steps`、`learning_rate`、`cutoff_len`、`num_train_epochs`
- LoRA 权重输出目录统一为 `model/lora/`
- 训练日志需完整保留 `LLaMA-Factory` 训练过程信息（如 loss、step、学习率、评估指标与时间戳），用于复现与调优

当前默认训练产物约定：

- 训练输入目录：`data/processed/sft_data/teacher_v2_plus_prompt_supplement_v2`
- LLaMA-Factory 数据目录：`data/processed/llama_factory/teacher_v2_plus_prompt_supplement_v2`
- LoRA 输出目录：`model/lora/teacher_v2_plus_prompt_supplement_v2_qwen35_4b`

训练时注意：

- 风格数据不要污染事实标签
- 知识数据优先来源于整理后的剧情证据
- 工具调用样本要强调 JSON 格式稳定性
- 多轮样本要覆盖指代消解、纠错追问、范围缩小
- 补充样本里，`clarify_user` 只用于真正的用户歧义，不要把“证据不足”误标成“需要澄清”
- `retrieve_more` 样本应尽量显式给出 `missing_slots`
- `follow_up_hypothesis` 只应包含服务检索的结构化线索，不应提前写入答案结论

如果单一 LoRA 导致能力相互干扰，优先拆任务，不要强行把所有目标揉进一个退化的适配器。

## 17. 推荐目录规划

如果仓库后续需要补齐代码结构，优先按下面方式组织：

```text
data/
  ArknightsGameData/
  processed/
    sft_data/
model/
  qwen3.5-4b/
  lora/
  embeddings/
  reranker/
indexes/
outputs/
scripts/
  llama_factory/
src/
  config/
    llama_factory_config.yaml
  data/
  retrieval/
  rerank/
  dialogue/
  llm/
  app/
tests/
```

目录职责建议：

- `scripts/`：一次性脚本、索引构建、模型转换、数据预处理
- `scripts/llama_factory/`：`LLaMA-Factory` 训练启动脚本、配置模板与本地多卡训练辅助脚本
- `configs/runtime_inference.json`：实际使用时的检索候选数、reranker 开关与多轮检索安全上限
- `data/processed/sft_data/`：在线 API 蒸馏并清洗后的 SFT 样本，按 `style` / `knowledge` / `tool` 分类管理
- `src/config/llama_factory_config.yaml`：集中管理 `LLaMA-Factory` 训练参数，保证可复现
- `src/data/`：剧情解析、清洗、切块
- `src/retrieval/`：embedding、FAISS、BM25、融合
- `src/goldenglow/retrieval/`：embedding、FAISS、BM25、融合与 reranker
- `src/goldenglow/inference/`：llama.cpp 调用封装、hypothesis / planner prompt、在线多轮检索与答案生成
- `indexes/`：FAISS 索引与稀疏索引产物
- `outputs/`：调试输出、评测结果、缓存

## 18. 开发优先级

按以下顺序推进，除非用户另有要求：

1. 剧情原文解析与清洗
2. chunk 与 metadata 设计
3. embedding 与 FAISS 建库
4. 稀疏检索分支
5. 混合检索融合
6. 假设文档生成
7. 交叉编码器重排
8. 答案生成
9. 多轮状态管理
10. function calling
11. LoRA 训练与挂载
12. 离线评测与调优

不要一开始就花大量时间做 UI。先把可验证的剧情问答主链路做通。

## 19. 测试与验收标准

至少建立以下验收维度：

- 检索召回率：相关剧情片段能否进入 Top-K
- 重排质量：高相关片段是否稳定排在前面
- 答案真实性：是否忠于剧情原文
- 幻觉率：是否出现无依据设定
- 多轮一致性：追问时是否能延续上下文
- 语气稳定性：澄闪风格是否自然但不过度

最低测试集应包含：

- 主线剧情问题
- 活动剧情问题
- 角色关系问题
- 时间线问题
- 含指代的追问
- 有歧义需澄清的问题
- 无法从现有语料确认的问题

## 20. 代码实现要求

默认使用 Python 编写调度逻辑，保持实现简单、可测、可替换。

要求：

- 核心流程模块化
- 配置集中管理
- 日志可读
- 中间结果可落盘复现
- 每个检索阶段都能单独调试

优先选择：

- 标准库
- 轻量依赖
- 明确的数据类或 schema
- 可从命令行直接运行的脚本

避免：

- 把核心逻辑塞进 notebook
- 把 prompt、schema、路径写死在多个文件里
- 用隐式全局状态管理多轮对话

## 21. AI 修改代码时的工作方式

AI 在本项目中进行开发时，应遵守以下行为：

- 先阅读现有目录与数据格式，再动手
- 优先做本地可运行、可验证的实现
- 修改数据解析或检索逻辑时，必须考虑对剧情准确率的影响
- 涉及回答策略时，先保证“有依据”，再考虑“像澄闪”
- 若某个模型文件或 reranker 文件尚未存在，应通过配置预留，而不是伪造路径

如果用户只说“实现剧情问答”，默认理解为：

- 先打通 RAG 主链路
- 再补多轮与 function calling
- 最后再做风格微调整合

## 22. 禁止事项

禁止：

- 把没有检索依据的内容写成剧情事实
- 用 LoRA 风格覆盖剧情证据
- 擅自替换既定模型或推理框架
- 在未确认格式的情况下硬编码交叉编码器模型名
- 跳过数据清洗直接把原始脚本文本整包塞进向量库
- 把工具调用做成不可观测黑箱

## 23. 一句话决策规则

当实现方案有取舍时，默认选择：

“更忠于剧情证据、更容易追踪调试、更符合既定本地技术栈”的方案。
