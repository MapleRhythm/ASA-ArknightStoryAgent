# 当前全流程与数据集细节审阅文档

生成日期：2026-06-03

本文档用于交给另一个 AI 或审阅者检查当前《明日方舟》剧情 RAG / SODA 蒸馏项目。文档只描述工程状态、数据资产、训练结果和已知问题，不包含任何 API key。

## 1. 项目目标

目标是在本地 4B 模型规模下实现一个剧情问答 Agent：

- 能根据 `data/ArknightsGameData/zh_CN/gamedata` 中的剧情、档案和相关文本检索证据。
- 能在证据不足时继续检索或拒答，而不是凭模型内部知识早答。
- 能在证据充分时回答可确认事实，而不是过度 abstain。
- 训练数据不直接模仿 teacher 原始答案，而是尽量按当前 evidence state 训练正确动作。

当前核心矛盾：

- 4B 模型已经能学习 `retrieve_more` 行为，但最终答案仍容易把模型先验混进证据答案。
- RAG 召回能覆盖部分 hard case，但如果 final answer prompt 没有 claim-level grounding，仍会出现 unsupported answer。
- teacher/API 具备自身剧情知识，直接蒸馏会污染 student，因此引入 evidence-only verifier。

## 2. 当前运行链路

实现主入口：

- `src/goldenglow/inference/cpu_pipeline.py`
- `scripts/run_cpu_inference.py`
- `scripts/run_eval50_hard10_gpu_abstain_flow.sh`
- `configs/runtime_inference_gpu.json`

标准 runtime 链路：

```text
question
-> user_question_hypothesis_generation
-> retrieval round 1
   -> dense / BM25 / MiniRAG / scoped chapter search / neighbor / same-story sweep
   -> fusion
   -> reranker
   -> prompt evidence selection
-> conclusion_generation
   -> answer_directly | retrieve_more | clarify_user | abstain
-> if retrieve_more:
   -> merge model-provided follow_up_hypothesis
   -> retrieval round 2
   -> conclusion_generation again
-> final answer
```

当前最大检索轮次：

- `max_retrieval_rounds = 2`

当前评测默认关闭 web context：

- 配置文件里 `inference.web_context.enabled = true`
- 但最新 eval 脚本通过 `--disable-web-context` 关闭
- 因此本文里的评测结论默认是本地文本 / 本地索引，不依赖外网

## 3. 检索与证据包配置

索引与语料路径：

- 原始文本：`data/ArknightsGameData/zh_CN/gamedata`
- 主索引目录：`indexes/arknights_story`
- chunk 文档：`indexes/arknights_story/documents.jsonl`
- dense index：`indexes/arknights_story/faiss.index`
- BM25 token：`indexes/arknights_story/bm25_tokens.pkl`
- MiniRAG 图：`indexes/arknights_story_minirag_v3/graph.json`
- reranker：`model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch`

`documents.jsonl` 是索引后的剧情 chunk，不是训练数据。RAG 检索时 dense/BM25/reranker 都围绕这些 chunk 取证据。

当前 `configs/runtime_inference_gpu.json` 关键参数：

```json
{
  "retrieval": {
    "device": "cuda",
    "enable_reranker": true,
    "dense_top_k": 120,
    "sparse_top_k": 120,
    "fusion_top_k": 80,
    "rerank_top_k": 32,
    "rerank_batch_size": 4,
    "reranker_max_length": 1024,
    "enable_minirag": true,
    "minirag_weight": 0.35,
    "minirag_chapter_isolation": true,
    "minirag_auto_second_retrieval": true,
    "enable_storyline_sparse_scope": true,
    "enable_scoped_chapter_search": true,
    "enable_neighbor_expansion": true,
    "enable_same_story_sweep": true
  },
  "inference": {
    "max_retrieval_rounds": 2,
    "prompt_evidence_top_k": 12,
    "prompt_evidence_max_chars_per_doc": 1800,
    "prompt_conclusion_evidence_max_total_chars": 24000,
    "conclusion_prompt_mode": "minimal",
    "answer_grounding_mode": "weak",
    "use_model_hypothesis": true,
    "use_model_conclusion_generation": true
  }
}
```

证据包长度：

- conclusion prompt 默认放 `top_k = 12` 个证据块
- 每块最多 `1800` 字符
- conclusion 总证据上限 `24000` 字符
- follow-up hypothesis 里使用 `evidence_brief`，每块压到约 `260` 字符，总计约 `1200` 字符

需要注意：

- `evidence_brief` 是从 prompt evidence 渲染出来的短证据视图，用于模型判断是否需要继续检索。
- `minirag_hints` 是 MiniRAG 图/关系提示，不等价于事实证据。当前 verifier prompt 明确规定：`minirag_hints` 不能作为事实支撑。
- 当前 answer grounding 是 `weak`，只做有限实体/身份类保护，不是完整 claim-level verifier。

## 4. 当前 prompt / 输出 schema

### 4.1 第一轮 hypothesis

任务名：

- `user_question_hypothesis_generation`

输出 schema：

```json
{
  "question": "...",
  "intent": "character_relation | compare | event_summary | out_of_scope | persona_chat | plot_fact | plot_reasoning | timeline",
  "query_type": "fact | relation | causality | reasoning | reveal | mystery | answerability",
  "entities": ["短实体"],
  "keywords": ["短关键词"],
  "expected_answer_type": "...",
  "dialogue_context": ""
}
```

要求：

- 只写检索线索，不回答问题。
- entities/keywords 要短、可检索、去重。
- 输出必须是合法 JSON。

### 4.2 conclusion_generation

任务名：

- `conclusion_generation`

当前 schema：

```json
{
  "question": "...",
  "next_action": "answer_directly | retrieve_more | clarify_user | abstain",
  "answer": "...",
  "missing_slots": ["..."],
  "clarification_question": "",
  "follow_up_hypothesis": {
    "question": "...",
    "query_type": "...",
    "entities": ["..."],
    "keywords": ["..."],
    "expected_answer_type": "...",
    "dialogue_context": ""
  }
}
```

字段规则：

- `answer_directly` / `abstain`：`answer` 非空，`follow_up_hypothesis = null`
- `retrieve_more`：`answer = ""`，`missing_slots` 非空，`follow_up_hypothesis` 是下一轮检索 JSON
- `clarify_user`：`clarification_question` 非空

当前实现会容忍部分 4B JSON 错误：

- 修复漏引号、漏冒号、残缺 JSON。
- 如果 `retrieve_more` 缺失 `follow_up_hypothesis`，会用启发式补一个。
- 这些 fallback 让 pipeline 更稳，但也会掩盖模型自身 schema 能力不足。

### 4.3 当前还没有上线的新 answer schema

最近讨论过的新 schema 还未作为当前训练/评测主流程：

```json
{
  "supported_facts": [
    {
      "id": "F1",
      "fact": "...",
      "evidence_refs": [
        {
          "evidence_id": "E1",
          "sentence_id": "E1.S6",
          "quote": "证据原句"
        }
      ]
    }
  ],
  "inferred_facts": [
    {
      "id": "I1",
      "fact": "...",
      "premise_fact_ids": ["F1", "F2"],
      "inference_type": "purpose"
    }
  ],
  "final_answer": "..."
}
```

建议下一版引入该 schema 或 evidence-only self-check：

- `supported_facts` 必须引用原证据块和原文 quote。
- `inferred_facts` 只能引用已支持事实，不允许引入新实体、新动机、新因果。
- 用户不需要看到 unsupported 字段；unsupported 只用于内部自检或训练过滤。
- 不建议把最终答案硬限制成 supported facts 拼接，因为推理题需要跨证据归纳；但可以要求推理结论列出 premise fact ids。

## 5. verifier-aware SODA 数据构造流程

核心思想：

- teacher 原始输出只是候选答案，不是训练标签。
- verifier 输出才是当前 evidence state 下的训练标签。
- verifier 必须 evidence-only，不能使用 teacher 自身剧情知识补证据。

主要脚本：

- `scripts/generate_soda_blackbox_distillation.py`
- `scripts/build_soda_api_verifier_dataset.py`
- `scripts/run_soda_eval50_len1800_api_verifier_flow.sh`
- `scripts/run_soda_extra_api_verifier_flow_3gpu.sh`
- `scripts/score_soda_kto_pairs_with_teacher.py`

### 5.1 Step 1：student 按真实 runtime 跑完整链路

对每个问题，让当前 4B student 在真实 RAG pipeline 中执行：

```text
question
-> hypothesis
-> retrieval round 1
-> conclusion
-> maybe follow_up_hypothesis
-> retrieval round 2
-> final conclusion
```

记录每个模型调用状态：

- 原 question
- task_type
- round
- exact prompt
- evidence_brief
- minirag_hints
- student_output
- retrieval_trace

这样数据来自 student 自己真实会走到的状态，而不是 teacher 离线构造的理想状态。

### 5.2 Step 2：teacher replay 同一个 prompt

把 student 当时看到的 exact prompt 给 API teacher：

- same question
- same round
- same hypothesis
- same evidence_brief
- same minirag_hints
- same output schema

teacher 只生成一个候选输出：

```json
{
  "next_action": "answer_directly",
  "answer": "..."
}
```

注意：

- teacher 可能用自身剧情知识早答。
- 所以 teacher replay 结果不能直接当 chosen。

### 5.3 Step 3：evidence-only verifier 判定正确动作

`scripts/build_soda_api_verifier_dataset.py` 中 verifier prompt 的核心约束：

- 只能依据 `allowed_evidence_brief` 判断。
- `teacher_policy_output` 只是候选，不是标准答案。
- hypothesis、entities、keywords、minirag_hints、student_output、teacher_output 都不是事实证据。
- 如果答案关键实体、身份、因果、动机或结果无法在 evidence 中找到支撑，标为 unsupported / teacher_prior_knowledge。
- evidence 不足但有明确检索方向，`correct_action = retrieve_more`。
- evidence 足够回答核心问题，`correct_action = answer_directly`。

verifier 输出字段：

```json
{
  "evidence_sufficient": false,
  "correct_action": "retrieve_more",
  "supported_answer": "",
  "missing_slots": ["缺少直接说明 X 的原文"],
  "student_action_error": "premature_answer",
  "teacher_action_error": "teacher_prior_knowledge",
  "teacher_answer_uses_prior_knowledge": true,
  "use_for_training": true,
  "label_reason": "证据不足，student/teacher 都早答"
}
```

### 5.4 Step 4：按 verifier 重新标 chosen/rejected

不是 teacher 说什么就训练什么，而是按 verifier 结果重标：

情况 A：student 早答，teacher 也早答，但 verifier 说证据不足。

```text
chosen: retrieve_more + missing_slots + follow_up_hypothesis
rejected: student answer_directly / teacher answer_directly
```

情况 B：student 继续检索，teacher 答案，verifier 说当前证据足够。

```text
chosen: answer_directly + supported_answer
rejected: student retrieve_more
```

情况 C：teacher 答案使用隐藏知识。

```text
如果 evidence 不足：chosen = retrieve_more
如果 evidence 只支持部分：chosen = supported partial answer
teacher answer 不作为 chosen
```

情况 D：round1 证据不足，round2 证据充分。

```text
round1 chosen = retrieve_more
round2 chosen = answer_directly
```

### 5.5 Step 5：teacher full-chain 只作 oracle

可以让 teacher 自己完整跑一遍链路：

```text
teacher question -> teacher hypothesis -> teacher retrieval -> teacher answer
```

用途：

- 找 student 没召回到的关键原文。
- 提取更好的 query / missing entity / story name。
- 生成后续 hard questions。

限制：

- 不直接把 teacher full-chain answer 当 student chosen。
- 除非 teacher 使用的证据也被放进 student prompt 或 student 重新召回到了。

## 6. 主要数据集资产

### 6.1 eval50 scoped-sweep verifier clean 数据

目录：

- `data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_clean_v1`

来源：

- 50 个召回评测问题
- 4B student 使用 SODA LoRA 跑真实 RAG
- teacher replay
- evidence-only verifier 重标
- 人工/脚本清理 4 个明显 unsupported patch

规模：

```json
{
  "records_total": 208,
  "records_train": 190,
  "records_val": 18,
  "kto_tags": {"True": 119, "False": 89},
  "task_counts": {
    "user_question_hypothesis_generation": 100,
    "conclusion_generation": 108
  }
}
```

主要 verifier reason：

```json
{
  "verifier_chosen_answer_directly": 50,
  "verifier_chosen_retrieve_more": 19,
  "reject_student_unsupported_answer": 17,
  "reject_student_over_retrieve": 7,
  "reject_teacher_unsupported_answer": 5,
  "reject_teacher_teacher_prior_knowledge": 4
}
```

### 6.2 extra300 verifier 数据

目录：

- `data/processed/llama_factory/soda_eval50_len1800_api_verifier_extra300_v1_soda_lora_merged`

来源：

- 自建 extra hard question pool，约 300 问
- 三张 GPU 并行 student rollout + teacher replay + verifier

规模：

```json
{
  "records_total": 1294,
  "records_train": 1190,
  "records_val": 104,
  "verifier_records": 432,
  "kto_tags": {"True": 725, "False": 569},
  "task_counts": {
    "user_question_hypothesis_generation": 583,
    "conclusion_generation": 711
  }
}
```

主要 verifier reason：

```json
{
  "verifier_chosen_answer_directly": 238,
  "verifier_chosen_retrieve_more": 185,
  "verifier_chosen_abstain": 8,
  "reject_student_unsupported_answer": 101,
  "reject_student_premature_answer": 28,
  "reject_teacher_unsupported_answer": 38,
  "reject_teacher_teacher_prior_knowledge": 38
}
```

### 6.3 mix eval50 + extra300 QC v2 数据

目录：

- `data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1_qc_v2`

规模：

```json
{
  "records_total": 1500,
  "records_train": 1379,
  "records_val": 121,
  "verifier_records": 500,
  "teacher_full_chain_records": 50,
  "kto_tags": {"True": 843, "False": 657},
  "task_counts": {
    "user_question_hypothesis_generation": 683,
    "conclusion_generation": 817
  }
}
```

处理：

- 从 clean eval50 verifier 数据和 extra300 verifier 数据合并。
- 后续做 QC v2。
- 有 `teacher_full_chain.jsonl`，但 teacher full-chain 仅作 oracle / 审计，不直接作为 chosen。

主要 verifier reason：

```json
{
  "verifier_chosen_answer_directly": 288,
  "verifier_chosen_retrieve_more": 204,
  "verifier_chosen_abstain": 8,
  "reject_student_unsupported_answer": 118,
  "reject_student_premature_answer": 30,
  "reject_student_over_retrieve": 43,
  "reject_teacher_unsupported_answer": 43,
  "reject_teacher_teacher_prior_knowledge": 42
}
```

### 6.4 teacher-scored KTO 数据

teacher scoring 输入：

- `data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1_qc_v2`

teacher scoring 输出：

- `data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1_qc_v2_teacher_scored`

评分配置：

```json
{
  "groups": 606,
  "batch_size": 10,
  "task_type": "all",
  "max_evidence_chars": 3000,
  "max_prompt_chars": 1400
}
```

评分标准：

- 0-5 分。
- KTO 使用：`best_score >= 4`，`margin >= 1.5`，`confidence >= 0.65`。
- SFT candidate 使用：`best_score >= 4`，`unsupported = false`。
- 低分差 pair 不适合 KTO，但若最佳单条本身高分、无 unsupported，可补 SFT。

过滤后：

```json
{
  "scored_records": {"train.json": 1379, "val.json": 121},
  "filtered_records": {"train.json": 462, "val.json": 32},
  "sft_candidate_records": {"train.json": 490, "val.json": 45},
  "filters": {
    "min_margin": 1.5,
    "min_confidence": 0.65,
    "min_best_score": 3.5,
    "min_sft_score": 4.0
  }
}
```

最终 KTO mix：

- `data/processed/llama_factory/soda_teacher_scored_kto_mix_v1`

规模：

```json
{
  "records": {"train": 600, "val": 40},
  "kto_tag": {
    "train": {"true": 301, "false": 299},
    "val": {"true": 19, "false": 21}
  },
  "task_type": {
    "train": {
      "user_question_hypothesis_generation": 105,
      "conclusion_generation": 495
    },
    "val": {
      "user_question_hypothesis_generation": 10,
      "conclusion_generation": 30
    }
  }
}
```

这个 KTO 数据已用于一次训练：

- `model/lora/teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2`

但该模型在 eval 中有格式/稳定性问题：

- eval50 errors = 4
- hard10 errors = 1

所以当前不建议直接作为 release 主模型。

### 6.5 API-QC SFT 数据

这部分是最新两阶段 SFT 使用的数据。

#### schema SFT QC 数据

目录：

- `data/processed/llama_factory/schema_sft_patch_v1_api_qc_v1`

输入：

- `data/processed/llama_factory/schema_sft_patch_v1`

API QC 参数：

```json
{
  "records_to_score": 1782,
  "batch_size": 8,
  "model": "deepseek-chat",
  "max_input_chars": 6500,
  "max_output_chars": 2500
}
```

输出：

```json
{
  "train.json": 1675,
  "val.json": 107,
  "actions": {"fix": 1477, "keep": 305},
  "rejected": 0
}
```

高频问题：

- 断词残片
- unsupported_claim
- redundant_keywords / redundant_entities
- dirty keywords/entities
- evidence sufficient 时过度检索
- missing_slots 不具体

#### conclusion chosen SFT QC 数据

目录：

- `data/processed/llama_factory/conclusion_chosen_sft_v1_api_qc_v1`

输入：

- `data/processed/llama_factory/conclusion_chosen_sft_v1`

API QC 参数：

```json
{
  "records_to_score": 300,
  "batch_size": 8,
  "model": "deepseek-chat",
  "max_input_chars": 7500,
  "max_output_chars": 2500
}
```

输出：

```json
{
  "output_splits": {"train.json": 221, "val.json": 13},
  "rejected_splits": {"train.json": 63, "val.json": 3},
  "actions": {"fix": 222, "keep": 70, "drop": 8}
}
```

高频问题：

- unsupported_claim = 77
- missing_slots 不具体 = 27
- over_retrieve = 22
- dirty_keywords = 15
- answer unsupported / over claim

去重：

- `exact conversations md5 across train+val`

#### 可用但未纳入最新两阶段 SFT 的大 SFT QC 数据

目录：

- `data/processed/llama_factory/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3_plus_detail_conclusion_patch_v1_abstain_mini8_retrieval_trim_v1_plus_teacher_scored_sft_v1_api_qc_v1`

规模：

```json
{
  "records_to_score": 1997,
  "output_splits": {"train.json": 1863, "val.json": 133},
  "rejected_splits": {"train.json": 1, "val.json": 0},
  "actions": {"fix": 1482, "keep": 514, "drop": 1}
}
```

说明：

- 这是已产出的数据资产。
- 最新有效评测的两阶段 SFT 没有直接使用它。
- 后续可作为更大 SFT 或 schema/grounding 训练候选，但需要抽样审计。

## 7. 训练栈与模型资产

基础模型：

- `model/qwen3.5-4b`

上一版 SODA LoRA：

- `model/lora/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_qwen35_4b_lr1e6_beta001_epoch3`

最新两阶段 SFT：

### Stage 1：schema SFT

配置：

- `src/config/llama_factory_schema_sft_patch_v1_api_qc_from_soda_lora_config.yaml`

输入 LoRA：

- `model/lora/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_qwen35_4b_lr1e6_beta001_epoch3`

输出：

- `model/lora/schema_sft_patch_v1_api_qc_from_soda_lora_qwen35_4b_lr8e6_epoch1`

训练参数：

```yaml
stage: sft
template: qwen3_nothink
cutoff_len: 5632
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 8.0e-6
num_train_epochs: 1.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
fp16: true
gradient_checkpointing: true
```

结果：

```json
{
  "train_loss": 0.3295041055873383,
  "eval_loss": 0.29202502965927124,
  "train_runtime": 2529.1284
}
```

### Stage 2：conclusion chosen SFT

配置：

- `src/config/llama_factory_conclusion_chosen_sft_v1_api_qc_from_schema_sft_cutoff3072_config.yaml`

输入 LoRA：

- `model/lora/schema_sft_patch_v1_api_qc_from_soda_lora_qwen35_4b_lr8e6_epoch1`

输出：

- `model/lora/conclusion_chosen_sft_v1_api_qc_from_schema_sft_qwen35_4b_lr2e6_epoch2_cutoff3072`

训练参数：

```yaml
stage: sft
template: qwen3_nothink
cutoff_len: 3072
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2.0e-6
num_train_epochs: 2.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
fp16: true
gradient_checkpointing: true
```

结果：

```json
{
  "train_loss": 1.0031517148017883,
  "eval_loss": 0.737480103969574,
  "train_runtime": 2885.6469
}
```

限制：

- `cutoff_len = 5632` 在本地 4090 OOM。
- `cutoff_len = 4096` 仍 OOM。
- 最终用 `cutoff_len = 3072` 完成，因此长 conclusion 样本有截断风险。

一键训练/评测脚本：

- `scripts/run_api_qc_sft_4b_train_eval.sh`

## 8. 最新评测结果

最新评测输出目录：

- `outputs/eval_api_qc_sft_4b_cutoff3072_mem52_20260603_134858`

评测问题：

- eval50：`outputs/eval_soda_api_verifier_v2/eval50_questions.txt`
- hard10：`outputs/eval_soda_api_verifier_v2/hard10_questions.txt`

评测配置：

- backend：vLLM
- LoRA：`model/lora/conclusion_chosen_sft_v1_api_qc_from_schema_sft_qwen35_4b_lr2e6_epoch2_cutoff3072`
- GPU memory utilization：`0.52`
- max batched tokens：`4096`
- max retrieval rounds：`2`
- web context：disabled

指标：

| model | split | errors | abstain_like | one_round | two_round | two_round_answer |
|---|---:|---:|---:|---:|---:|---:|
| new API-QC SFT | eval50 | 0 | 9 | 32 | 18 | 10 |
| old conclusion SFT | eval50 | 0 | 11 | 35 | 15 | 5 |
| teacher-scored KTO | eval50 | 4 | 11 | 25 | 21 | 13 |
| new API-QC SFT | hard10 | 0 | 1 | 6 | 4 | 4 |
| old conclusion SFT | hard10 | 0 | 1 | 8 | 2 | 2 |
| teacher-scored KTO | hard10 | 1 | 1 | 2 | 7 | 7 |

new API-QC SFT 原始 summary：

```json
{
  "eval50": {
    "count": 50,
    "errors": 0,
    "abstain_like": 9,
    "rounds": {"1": 32, "2": 18},
    "final_actions": {"abstain": 8, "answer_directly": 42},
    "action_sequences": {
      "answer_directly": 32,
      "retrieve_more>answer_directly": 10,
      "retrieve_more>abstain": 8
    },
    "avg_elapsed_sec": 71.223
  },
  "hard10": {
    "count": 10,
    "errors": 0,
    "abstain_like": 1,
    "rounds": {"1": 6, "2": 4},
    "final_actions": {"answer_directly": 10},
    "action_sequences": {
      "answer_directly": 6,
      "retrieve_more>answer_directly": 4
    },
    "avg_elapsed_sec": 69.808
  }
}
```

结论：

- 最新 SFT 相比 old conclusion SFT 更愿意第二轮检索。
- eval50 的 `retrieve_more > answer_directly` 从 5 提升到 10。
- eval50 的 abstain_like 从 11 降到 9。
- hard10 的二轮回答从 2 提升到 4。
- 但事实性不是稳定提升，仍有 unsupported confident answer。
- 当前不建议盲目替换 release，需要先解决 final-answer grounding。

## 9. 已知 failure cases

### 9.1 PCS 是什么

现象：

- 检索能召回 `act42side` 相关证据。
- 模型最终回答包含“普瑞赛斯开发”。

问题：

- 当前证据更支持“前文明制造 / 基于仿生学的实验性质产物 / 普瑞赛斯校准并用于源石项目”。
- “普瑞赛斯开发”是 unsupported overclaim。

根因：

- final answer prompt 让模型“总结剧情知识”，不是先列证据支持事实再合成答案。
- weak grounding 对“开发/制造/校准”这种动词关系没有强检查。

### 9.2 真龙为什么要启动不反

现象：

- 新模型仍可能一轮回答，答案偏泛化危机解释。

风险：

- 没有优先围绕“启动/不反/代价/目的/危机”的直接原文组织回答。
- 多轮 RAG 不能保证模型会产生正确二轮 query。

### 9.3 特雷西斯为什么要建造碎片大厦

现象：

- 新 SFT 可能给 unsupported confident answer。
- teacher-scored KTO 更保守，可能 abstain。

说明：

- SFT 改善了动作稳定性，但可能牺牲安全性。
- KTO 更倾向于拉开 chosen/rejected 行为，但目前有格式错误和 eval errors。

### 9.4 岁兽是什么

现象：

- 可能回答过细、把分支或推测混写成确定设定。

风险：

- 概念定义题需要区分“证据中的直接定义”和“跨证据归纳”。

### 9.5 JSON schema 问题

eval50/hard10 中出现过一次 invalid hypothesis JSON：

- 输出用括号或不合法 JSON。
- pipeline fallback 修复后继续运行。

风险：

- 指标里 errors 为 0 不代表模型原生 schema 完全正确。
- 需要单独统计 raw JSON failure rate。

### 9.6 现有启发式后处理

`src/goldenglow/inference/cpu_pipeline.py` 中存在部分针对 reveal / suiling / event reference 的本地后处理和 fallback。

风险：

- 这些逻辑能缓解局部问题，但可扩展性有限。
- 审阅时应区分“模型真实学到的行为”和“pipeline 后处理修正”。

## 10. 当前主要问题判断

当前瓶颈排序：

1. final answer evidence grounding 不够强。
2. conclusion stage 训练样本被 `cutoff_len=3072` 截断，长 evidence prompt 学习不足。
3. teacher/API QC 能修复大量数据，但仍需抽样防止 teacher prior knowledge 混入。
4. KTO 数据有更强负例，但训练后格式稳定性不如 SFT。
5. RAG 召回仍有错章/同角色跨章节污染，但 reranker + scope 已经显著改善，不是唯一瓶颈。

对“为什么强制两轮 RAG 仍会早答”的解释：

- 当前 pipeline 不是无条件两轮，而是最多两轮。
- 第一轮 conclusion 可以 `answer_directly`。
- 即使生成第二轮，第二轮 query 质量取决于 model 的 `follow_up_hypothesis`。
- 强制检索并不等于强制 evidence grounding；第二轮证据回来后，最终答案仍可能夹带先验。

## 11. 建议下一步改造

### 11.1 引入 evidence-grounded answer self-check

不一定要硬规则拼接答案，但建议 final answer 前增加一次本地模型自检：

```text
answer draft
-> extract supported_facts with evidence quote
-> extract inferred_facts with premise fact ids
-> rewrite final_answer, removing unsupported claims
```

如果不想多一次模型调用，可以把 final answer prompt 改成：

```text
先输出 supported_facts，再输出 inferred_facts，最后输出 final_answer。
final_answer 只能使用 supported_facts 和 inferred_facts 中的信息。
```

### 11.2 数据层面增加 negative cases

重点补：

- 证据中只支持 A 校准/参与，但模型回答 A 开发/制造。
- 证据中只支持部分事实，但模型扩写动机/因果。
- 证据已足够时过度 abstain。
- 证据不足时早答。
- 同角色错章节召回后的错误回答。

### 11.3 训练层面建议

可选路线：

1. 在大显存机器上重新训 conclusion SFT，`cutoff_len >= 5632`。
2. 用 teacher-scored KTO 数据做小学习率 KTO，但必须加强 JSON/schema 约束评测。
3. 使用 SFT + KTO 分阶段：先 schema/conclusion SFT 保格式，再 KTO 拉动作偏好，最后少量 SFT 回稳格式。
4. 引入 supported_facts schema 后重建 conclusion SFT 数据，而不是继续修旧 schema。

## 12. 复现入口

### 12.1 跑当前两阶段 SFT + eval

```bash
bash scripts/run_api_qc_sft_4b_train_eval.sh
```

关键环境变量：

```bash
CUDA_VISIBLE_DEVICES=1
GPU_MEMORY_UTILIZATION=0.52
RUN_SCHEMA=1
RUN_CONCLUSION=1
RUN_EVAL=1
```

### 12.2 只跑 eval50/hard10

```bash
RUN_NAME=eval_api_qc_sft_4b_cutoff3072_mem52_manual \
LORA_PATH=model/lora/conclusion_chosen_sft_v1_api_qc_from_schema_sft_qwen35_4b_lr2e6_epoch2_cutoff3072 \
GPUS=0 \
GPU_MEMORY_UTILIZATION=0.52 \
MAX_NUM_BATCHED_TOKENS=4096 \
MAX_RETRIEVAL_ROUNDS=2 \
DISABLE_WEB_CONTEXT=1 \
ENFORCE_EAGER=1 \
bash scripts/run_eval50_hard10_gpu_abstain_flow.sh
```

### 12.3 生成 SODA verifier 数据

需要先在 shell 中设置 API key 环境变量：

```bash
export DEEPSEEK_API_KEY='...'
```

不要把 key 写进脚本或文档。

跑 eval50 verifier flow：

```bash
bash scripts/run_soda_eval50_len1800_api_verifier_flow.sh
```

跑 extra300 三 GPU flow：

```bash
bash scripts/run_soda_extra_api_verifier_flow_3gpu.sh
```

### 12.4 teacher batch scoring

```bash
python scripts/score_soda_kto_pairs_with_teacher.py \
  --input-dir data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1_qc_v2 \
  --output-dir data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1_qc_v2_teacher_scored \
  --batch-size 10
```

## 13. 给审阅 AI 的重点问题

请重点检查以下问题：

1. verifier prompt 是否真的足够 evidence-only，是否还有 teacher prior knowledge 泄漏路径。
2. `evidence_brief` 只取 prompt evidence 的短摘要，是否会导致 verifier 误判 evidence insufficiency。
3. 当前 conclusion schema 是否应该升级到 `supported_facts/inferred_facts/final_answer`。
4. KTO 数据里的 positive/negative 是否足以教会 4B 保守检索，而不是过度 abstain。
5. 是否应把 teacher-scored SFT candidates 合并到 SFT 训练，还是先人工抽样再用。
6. cutoff_len 3072 对 conclusion SFT 是否造成关键证据截断，是否必须转到大显存机器训 5632+。
7. local reranker 是否能作为弱 verifier，用于降低 API verifier 成本；如果可以，应如何避免 reranker 只判断相关性、不判断事实支持。
8. 当前启发式后处理是否应剥离出评测开关，避免高估模型能力。

