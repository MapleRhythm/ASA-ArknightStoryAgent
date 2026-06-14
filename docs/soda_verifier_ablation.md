# Verifier-Aware SODA 消融实验协议

本文档定义 verifier-aware SODA 的可复现实验，用于分别验证数据构造质量与模型运行效果。核心原则是配对控制：同一批 student rollout、同一批 teacher replay、同一份 prompt evidence 保持不变，仅替换标签构造策略或训练后模型。

## 实验命题

- 数据质量命题：直接采用 teacher-as-chosen 的 Raw SODA 会把 teacher prior、unsupported answer、premature answer 和 action mismatch 写入正样本；evidence-only verifier 能显式剔除或重标这些噪声。
- 负对照命题：只检查 JSON 合法性、action 枚举和 quote 是否出现在 evidence 中，无法识别 evidence insufficiency、teacher prior 和 unsupported reasoning。
- 模型效果命题：在相同 runtime、相同问题集、相同检索预算和相同解码参数下，使用 verifier-aware SODA 训练的 LoRA 应降低 unsupported / premature / over-abstain 等 evidence-only scorer 错误，并改善 action/support/reward 分数。

## 数据质量消融

固定输入：

- Raw rollout pair：`raw_pairs.jsonl`
- Verifier verdict：`api_verifier_records.jsonl`
- 最终 KTO 数据：`train.json`、`val.json`

对照策略：

- `Raw SODA teacher-as-chosen`：teacher 输出恒为 positive，student 输出恒为 negative。
- `Output-only heuristic`：仅以 JSON 合法性、action 枚举和 exact quote presence 接受 teacher。
- `Verifier-aware relabel`：依据 verifier 的 `correct_action`、`student_action_error`、`teacher_action_error` 与 `teacher_answer_uses_prior_knowledge` 重建 chosen / rejected。

复现命令：

```bash
python scripts/analyze_soda_verifier_ablation.py
```

产物：

- `outputs/soda_verifier_ablation_20260609/report.md`
- `outputs/soda_verifier_ablation_20260609/summary.json`
- `outputs/soda_verifier_ablation_20260609/audit_cases.md`
- `outputs/soda_verifier_ablation_20260609/audit_cases.jsonl`

当前结果摘要：

| 数据集 | Prompts | Raw unsafe chosen | Output-only unsafe accepted | Verifier-aware unsafe chosen | Raw unsupported/prior positive | Student safer than teacher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `eval50_clean` | 69 | 12 / 17.39% | 12 / 17.39% | 0 / 0.00% | 10 / 14.49% | 6 / 8.70% |
| `eval50_plus_extra300_qc_v2` | 500 | 108 / 21.60% | 108 / 21.60% | 0 / 0.00% | 98 / 19.60% | 53 / 10.60% |

该消融的证据边界是 verifier-defined label quality。它能证明 verifier gate 在其 evidence-only 合同下移除了 Raw SODA 正样本噪声；若要证明 verifier verdict 本身的绝对正确性，还需要额外人工盲审或独立裁判复核。`audit_cases.md` 已导出可人工复核样本。

## 模型效果消融

严格对照需要同时满足以下条件：

- 同一问题集：`outputs/eval_soda_api_verifier_v2/eval50_questions.txt` 与 `hard10_questions.txt`
- 同一 runtime config：本次严格运行使用 `configs/runtime_inference_gpu_no_reranker.json`
- 同一检索预算：默认 `MAX_RETRIEVAL_ROUNDS=2`
- 同一 GPU 数、batch token、memory utilization、web context 开关
- 可选同一评分器：`scripts/score_runtime_answers_with_teacher.py --include-all-actions --prompt-style compact`

执行命令：

```bash
PYTHON_BIN=/home/zhb/miniconda3/envs/train/bin/python \
GOLDENGLOW_USE_TRAIN_OVERRIDE=1 \
RUNTIME_CONFIG=configs/runtime_inference_gpu_no_reranker.json \
RUN_TEACHER_SCORE=0 \
RUN_EVAL50=1 \
RUN_HARD=1 \
GPUS=0,2 \
GPU_MEMORY_UTILIZATION=0.80 \
MAX_NUM_BATCHED_TOKENS=4096 \
bash scripts/run_soda_verifier_model_ablation.sh
```

GPU driver 不可用时，可使用 CPU/GGUF fallback。该路径会顺序合并、转换并量化 Raw SODA 与 verifier-aware SODA，两套模型使用同一 llama.cpp CPU runtime 跑相同问题集：

```bash
PYTHON_BIN=/home/zhb/miniconda3/envs/train/bin/python \
RUN_EVAL50=0 \
RUN_HARD=1 \
bash scripts/run_soda_verifier_cpu_gguf_ablation.sh
```

CPU/GGUF 路径默认只保留 q4 中间产物，并清理 f16 GGUF，以降低磁盘峰值；如需重建，设置 `FORCE_EXPORT=1`。

严格模型效果输出目录：

- Raw SODA：`outputs/eval50_hard10_raw_soda_550_verifier_ablation`
- Verifier-aware SODA：`outputs/eval50_hard10_verifier_soda_v2_ablation`
- Evidence-only scorer：`outputs/runtime_teacher_scores/soda_verifier_model_ablation`
- CPU/GGUF Raw SODA：`outputs/soda_verifier_cpu_gguf_ablation/raw_soda_550_cpu_gguf`
- CPU/GGUF Verifier-aware SODA：`outputs/soda_verifier_cpu_gguf_ablation/verifier_soda_v2_cpu_gguf`

当前严格运行结果：

| 数据集 | Raw count/errors | Verifier count/errors | Raw answer_directly | Verifier answer_directly | Raw abstain | Verifier abstain | Raw avg sec | Verifier avg sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eval50` | 50 / 0 | 50 / 0 | 26 | 27 | 24 | 23 | 88.815 | 77.646 |
| `hard10` | 10 / 0 | 10 / 0 | 5 | 7 | 5 | 3 | 68.007 | 64.053 |

该结果是严格同运行时的 behavioral ablation：两组只切换 LoRA，问题集、runtime config、GPU 数、检索轮数与解码参数保持一致。Verifier-aware SODA 在 hard10 上将 `answer_directly` 从 5 提升到 7，并将 `abstain` 从 5 降到 3；eval50 上行为差异较小。一次 verifier eval50 schema 解析异常已按同配置单题重试并替换，最终四个 strict 输出文件均为完整行数且 `errors=0`。

Evidence-only scorer 已使用同一批 strict 输出完成质量评估：

| Run | Count | Positive | Neutral | Negative | Unsupported | Premature | Over-abstain | Avg action | Avg support | Avg coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_soda` | 60 | 22 | 8 | 30 | 13 | 8 | 9 | 3.533 | 2.417 | 2.733 |
| `verifier_soda` | 60 | 25 | 6 | 29 | 11 | 5 | 9 | 3.750 | 3.017 | 3.217 |

Paired delta（`verifier_soda - raw_soda`）为：`action_score +0.2167`，`support_score +0.6000`，`negative -1`，`unsupported -2`，`premature -3`，`over_abstain 0`。这说明 verifier-aware 数据不仅改变了 action policy，在 evidence-only 质量评分上也提升了证据支撑与覆盖，同时减少 unsupported 与 premature answer。

报告脚本会自动读取上述目录，并生成两层模型效果证据：

- Runtime summary：错误数、abstain-like 数、final action 分布。
- Teacher scorer summary：positive / neutral / negative、unsupported、premature、over-abstain、avg action score、avg support score、avg reward score。
- Paired historical probe：若存在同题历史输出，报告会单独列出，但只有 runtime signature 相同才可视为严格模型效果消融。

若 scorer 产物包含 `raw_soda` 与 `verifier_soda` 两组运行，报告会按同一 `dataset/index` 自动计算 paired delta。主要判定量为：

- `avg_support_delta_verifier_minus_raw > 0`
- `avg_action_delta_verifier_minus_raw > 0`
- `avg_reward_delta_verifier_minus_raw > 0`，若使用 additive scorer
- `negative_delta < 0`
- `unsupported_delta < 0`
- `premature_delta < 0`
- `over_abstain_delta < 0`

## 当前环境状态

本机 GPU 已恢复，严格模型效果消融已在 2 张 RTX 4090 上完成。`scripts/run_soda_verifier_model_ablation.sh` 保留 CUDA / vLLM 预检，后续重跑时会提前拦截 driver、CUDA 或 vLLM ABI 问题。

Evidence-only teacher scorer 已完成 compact 模式评分。当前结论同时包含 runtime behavioral evidence 与 evidence-only quality evidence；若后续需要 reward-style 加性分数，可用 `--prompt-style additive` 追加一轮 scorer。

## 已纳入的历史模型行为证据

当前报告额外纳入了一组 hard10 同题历史 probe：

- Raw SODA：`outputs/retrieval_eval/soda_blackbox_hard10_20260530.jsonl`
- Verifier-aware SODA：`outputs/eval_soda_api_verifier_v2/hard10_answers.jsonl`

两者问题完全同序，行为统计为：

| Probe | Pairs | Raw errors | Verifier errors | Raw abstain-like | Verifier abstain-like | Raw answer_directly | Verifier answer_directly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hard10 overlap | 10 | 0 | 0 | 3 | 1 | 8 | 10 |

该结果说明 verifier-aware SODA 在这组历史 hard10 上减少了过度保守的 abstain-like 行为。但 Raw 运行使用 `web_context_enabled=True`、`enforce_eager=False`，verifier-aware 运行使用 `web_context_enabled=False`、`enforce_eager=True`，因此该结果只能作为补充行为证据，不替代上文定义的严格同运行时消融。
