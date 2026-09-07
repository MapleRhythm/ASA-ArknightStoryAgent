# ASA reader × evidence P0 诊断执行说明

日期：2026-09-07

## 当前状态

依据 `ASA_Architecture_Training_Optimization_20260907` 的首个未完成 P0，
离线诊断框架已经补齐，并已在 241 的三张 RTX 4090 上完成 strict9
流程 smoke：同一 Qwen3.5-4B 基座分别运行 untouched base 与当前
clean-SFT adapter，共 `9 个问题族 × 3 种证据条件 × 2 个 reader`。
所有 54 次生成均未截断；矩阵中的 27 行均经人工确认当前证据充分。

这次 smoke 的人工修正完整回答率为：

| Reader | 最小充分证据 | + hard distractor | 实际 prompt |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B base | 4/9 = 44.4% | 0/9 | 2/9 = 22.2% |
| clean-SFT | 4/9 = 44.4% | 1/9 = 11.1% | 1/9 = 11.1% |

详细 provenance、自动 GLM 读数、两条人工 override 和被排除实验见
`docs/STRICT9_READER_EVIDENCE_FINDINGS_20260907.md`。这不是线上系统正确率，
也不是约 60 题的正式 P0；9B 资产仍未找到，因此 3-reader 对照尚未完成。
在正式诊断完成前，不据此改变生产 MiniRAG、top-k、MMR 或检索轮数。

后续已在同一 minimal evidence 上完成自然语言短答对照：base 为 3/9
complete，clean-SFT 为 5/9；对应 Exx 都是 4/9。只有杰克逊被自然语言明确
救回，女妖、芙蕾达、莫斯提马和老天师的实体/关系错误仍在。因此 Exx 协议
负担只得到局部支持，不能作为主要失败解释。详见
`docs/STRICT9_READER_NATURAL_LANGUAGE_CONTROL_20260907.md`。

新增旁路工具：

- `scripts/build_reader_evidence_diagnostic.py`
- `scripts/build_reader_natural_language_control.py`
- `scripts/generate_exx_predictions_vllm.py` 的 base-reader 支持
- `scripts/summarize_reader_evidence_diagnostic.py`

这些工具不修改生产 runtime、检索器或默认模型。

## 防止错误标签的约束

构造器不会执行以下不可靠推断：

1. 不把 GLM 校准引用的并集自动称为“最小充分证据”；
2. 不把任意非 gold 候选自动称为“真实同场景干扰”；
3. 不要求唯一 gold ID；每个事实可以标注多个合法 `support_groups`；
4. 不允许一个问题族重复进入矩阵，避免宏观结论被重复题加权；
5. 若证据预算截掉 reviewer 声明的必要证据，直接失败，不生成伪 Full 样本；
6. 若一个事实需要 3 条以上证据，要求先拆成可核验原子事实，不静默突破
   当前 Exx 每 fact 1–2 个 E-ID 的协议。

## 审核标注格式

每个 snapshot 对应一条 JSON/JSONL 标注：

```json
{
  "id": "snapshot-task-id",
  "question_family": "activities/act-name",
  "review_status": "approved",
  "reviewer": "reviewer-name",
  "minimal_sufficient_evidence_ids": [
    "activities/example#chunk-0001",
    "activities/example#chunk-0002"
  ],
  "distractor_evidence_ids": [
    "activities/example#chunk-0008"
  ],
  "answer_facts": [
    {
      "fact": "一个人工核对的必要原子事实。",
      "support_groups": [
        ["activities/example#chunk-0001"],
        [
          "activities/example#chunk-0003",
          "activities/example#chunk-0004"
        ]
      ]
    }
  ],
  "notes": "记录时间、主体、关系、否定与替代证明的复核说明。"
}
```

`minimal_sufficient_evidence_ids` 必须完整支持所有 `answer_facts`。
`distractor_evidence_ids` 必须是真实语料中的同角色/同场景难例，但不能属于
任何已接受的支持组。

## 构造三种 evidence 条件

```bash
python3 scripts/build_reader_evidence_diagnostic.py \
  --snapshots /path/to/family-held-out.snapshots.jsonl \
  --annotations /path/to/reviewed-reader-evidence.jsonl \
  --runtime-config configs/runtime_gpu_reranker_qwen35_4b_evidence_id.json \
  --output-dir /path/to/reader_evidence_matrix_20260907
```

输出的每道题有三种条件：

- `minimal_sufficient`
- `sufficient_with_distractors`
- `actual_prompt`

前两种证据顺序按固定 seed 打乱并同步重映射 E-ID；`actual_prompt` 保留冻结
snapshot 的选择顺序。输出 `audit.json` 记录输入 SHA256、预算、审核策略和
各条件充分性。

## 运行 reader 对照

4B 原模型和当前 adapter 共用同一 base engine：

```bash
python3 scripts/generate_exx_predictions_vllm.py \
  --gold /path/to/reader_evidence_matrix_20260907/reader_evidence_matrix.json \
  --base-model /path/to/Qwen3.5-4B \
  --base-reader qwen35_4b_base \
  --adapter clean_sft=/path/to/clean-sft-adapter \
  --output-dir /path/to/reader_evidence_4b_predictions
```

9B 原模型单独运行，避免把两个不同基座塞入同一 vLLM engine：

```bash
python3 scripts/generate_exx_predictions_vllm.py \
  --gold /path/to/reader_evidence_matrix_20260907/reader_evidence_matrix.json \
  --base-model /path/to/Qwen3.5-9B \
  --base-reader qwen35_9b_base \
  --output-dir /path/to/reader_evidence_9b_predictions
```

两次运行都使用 non-thinking 的 canonical Exx prompt、temperature 0，并记录
模型/adapter SHA256、prompt token、生成 token、finish reason 和截断状态。
vLLM runner 通过 checkpoint tokenizer 的 `apply_chat_template` 显式传入
`enable_thinking=false`；对当前 Qwen3.5 template，这会在 assistant 起始处
写入空的 `<think></think>` 块。不能再用手写的空 assistant prefix 冒充
canonical non-thinking prompt，否则 base reader 会进入长 thinking 并失配。

若机器已有 Transformers/PEFT 但没有 vLLM，可分别运行同一矩阵；base reader
必须省略 `--adapter`，不要用空目录或伪 adapter 代替：

```bash
python3 scripts/generate_exx_predictions_transformers.py \
  --gold /path/to/reader_evidence_matrix_20260907/reader_evidence_matrix.json \
  --base-model /path/to/Qwen3.5-4B \
  --reader-name qwen35_4b_base \
  --output /path/to/qwen35_4b_base.predictions.json

python3 scripts/generate_exx_predictions_transformers.py \
  --gold /path/to/reader_evidence_matrix_20260907/reader_evidence_matrix.json \
  --base-model /path/to/Qwen3.5-4B \
  --adapter /path/to/clean-sft-adapter \
  --reader-name clean_sft \
  --output /path/to/clean_sft.predictions.json
```

Transformers runner 与 vLLM runner 都保留每行 `diagnostic` 元数据，并对基座
目录顶层资产逐文件记录 SHA256 和聚合哈希。两种 reader 必须写入不同的新
输出文件，runner 会拒绝覆盖已有预测或 metadata。

## 语义审计与汇总

对三个 prediction 文件分别运行现有 evidence-only set audit：

```bash
python3 scripts/audit_exx_set_grounding.py \
  --predictions /path/to/qwen35_4b_base.predictions.json \
  --output /path/to/qwen35_4b_base.set-audit.json
```

然后生成统一报告：

```bash
python3 scripts/summarize_reader_evidence_diagnostic.py \
  --gold /path/to/reader_evidence_matrix_20260907/reader_evidence_matrix.json \
  --prediction qwen35_4b_base=/path/to/qwen35_4b_base.predictions.json \
  --prediction clean_sft=/path/to/clean_sft.predictions.json \
  --prediction qwen35_9b_base=/path/to/qwen35_9b_base.predictions.json \
  --audit qwen35_4b_base=/path/to/qwen35_4b_base.set-audit.json \
  --audit clean_sft=/path/to/clean_sft.set-audit.json \
  --audit qwen35_9b_base=/path/to/qwen35_9b_base.set-audit.json \
  --output /path/to/reader_evidence_summary.json
```

汇总把 protocol 指标与回答语义指标分开，并按人工标注的
`reference_sufficiency` 分层。主读数是：

- `reference_condition_success_per_all_rows`
- `complete_per_reference_sufficient`
- `appropriate_action_per_reference_insufficient`
- `audit_coverage`
- `any_nonentailed_fact`
- `any_unsupported_or_contradicted`
- `any_irrelevant_fact`
- action correct、over-retrieve、over-abstain、premature-answer
- strict JSON、schema、非法 E-ID、截断

不能用 schema/action accuracy 替代回答正确率。对
`reference_sufficiency=sufficient` 的行，`complete` 才表示完整回答；
对 `insufficient` 的行，正确行为是证据集合审计为不足且动作合适，而不是
把 `set_support=none` 误报成“回答失败”。`complete_per_all_rows` 仍保留为
保守下界，但不能单独作为完整系统正确率。audit 未成功的行不会被当作成功，
并通过 `audit_coverage` 暴露；报告必须同时查看这些分母。

## 结果解释

完成矩阵后按以下规则进入下一阶段：

| 观察 | 后续 |
| --- | --- |
| 原 4B 在最小充分证据上明显优于 adapter | 审计 chat template、loss mask、旧字段污染和训练退化 |
| 自然语言对照正确、Exx 失败 | 优先检查协议负担、引用上限、parser/validator |
| 最小充分证据好，加入干扰后下降 | 构造 RAFT 风格 Full + hard distractor SFT |
| 三个 reader 在实际 prompt 差、在 Full 明显改善 | 优先 child→parent、上下文前缀和 bundle packing |
| 4B Full 仍差、9B 明显改善 | 评估 9B reader 的质量/延迟收益 |
| 所有 reader 在人工 Full 上仍失败 | 回查问题歧义、时间限定、说话人可信度与答案 rubric |

在该 P0 诊断完成前，不启动正式 RLVR，也不修改生产 top-k、MiniRAG、
set coverage 或检索轮数。
