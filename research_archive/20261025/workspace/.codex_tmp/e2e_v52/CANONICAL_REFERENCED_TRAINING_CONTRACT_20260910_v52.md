# v52：当前 LoRA 登记数据与运行输出格式的差异

规范实验：`outputs/experiments/referenced_training_contract_20260910.v52`。
规范仓库：`${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_HOME}/ASA-ArknightStoryAgent`。
沿用北京时间 2026-09-10 文件系列；JSON 回执使用 UTC。

## 能确认的来源范围

当前 LoRA 模型卡声明的 dataset 名、对应 KTO YAML 配置、merged 目录中的
`dataset_info.json` 一致，指向：

`data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_merged`。

不是同目录旁的 `clean_v1`，也不能因名字中有 gpu3 就把它当作单 GPU shard。
`build_summary.json` 列出的输入是 gpu0／gpu1／gpu2 三份 shard。
配置输出目录与当前使用的 LoRA 完全相符，采用 `qwen3_nothink`、KTO、3 epochs，
并声明从另一份 SODA LoRA 继续训练；trainer state 记录 epoch 3、step 144。
配置只被读取，`do_train` 没有执行，`training_args.bin` 没有加载或反序列化。

这里证明的是**当前文件的声明链一致**，不是训练时数据内容的 SHA 证明。
本轮没有完成父 LoRA 的全部训练历史审计；不能说整个模型只训练过以下 208 条数据。
2026-06-03 的历史说明文档仅作路径线索，其 MiniRAG／weak 运行配置未恢复使用。

## 对全部 208 条当前记录的检查

train 190、val 18；正向 KTO 119、负向 89。按 `kto_tag` 分组，不按可能过时的
`meta.preference_role` 判断训练期望。分组汇总与现有 build summary 一致。

| 项目 | 登记数据中的正向 conclusion 样本 | v51 实际运行请求 |
| --- | --- | --- |
| 记录数／范围 | train 63、val 6 | 六个冻结的 canonical prompt |
| schema | `conclusion_v2` | `grounded_action_v1` |
| 答案字段 | 普通 `answer`，含 missing_slots／follow_up 等字段 | `supported_facts`、`inferred_facts`、`final_answer` |
| 引用 | 69 条均无 `evidence_refs` 字段 | 每条 fact 要求 E-ID 与原文 quote |
| 证据格式 | 69 条均是 `1. 文档路径...` 式编号 | `[E1] ...` 式证据块 |

63 条正向 conclusion 训练目标中，47 条直接回答、16 条继续检索；
6 条对应验证目标为 3 条直接回答、3 条继续检索。它们都可被严格解析为 JSON object，
但**没有一条采用当前结构化 fact／reference 目标格式**。

这只说明本轮核查的登记阶段没有提供这种正向目标，不证明模型完全没学过此格式，
也不证明它是当前错归、制造者断言或 E-ID 错绑的因果来源。

有一条无效 JSON 和一条 conclusion 任务却输出 hypothesis 字段的记录，二者
`kto_tag` 都为 false，是负例，不能当成“被教为正确”的训练标签错误。
另外 12 条负例仍保留 `preference_role=teacher_positive` 元数据；
本次统计没有用该旧元数据覆盖 KTO 标签。

## train／val 重叠的精确边界

- train 有 49 个不同问题，val 有 8 个；7 个问题文字在两边相同。
- val 的 18 条记录中，14 条属于训练中出现过的问题。
- **完全相同的输入 prompt 为 0**。因此不能把同题不同状态概括成整条输入泄漏。
- val 仍有 1 个问题未在 train 中出现：卓娅前往彼得海姆中学的原因。
- v51 的六个问题与这份当前登记数据没有完全相同的问题文字。
  没有按数字 ID 后缀连接数据，也不据此宣称语义独立、未见过改写题或未受父模型训练影响。

模型卡的 validation loss 不是语义正确率；以上统计同样不是正确率。
`audit_v1.json` 保存全部 208 个记录的文件内位置、行对象 SHA、标签、schema、
问题与实际 prompt digest；归档前以新进程逐字段重算。

preflight SHA256：
`8f571c15d5fd0f4067551a3144d929a28c614805f2c1540692034b865a4cbcab`。

## 处理决定

记录格式差异和来源证明的边界，不据此立即训练、改写标签或切换模型。
原始 train／val、模型、配置、旧 manifest 和历史结果未改。
没有新 LLM、检索或 GPU 调用；没有服务重启、commit 或 push。
继续保留 v49 的 trace 修复和原有严格引用校验；v51 通用归属提示不推广。
