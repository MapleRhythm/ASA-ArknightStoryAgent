# GLM 完整证据标注管线

这是一条独立的离线数据管线，不占 GPU、不改正式链路，也不会替换正在训练的数据。

阶段：准备问题与 20 段索引全文 → GLM 新标注 → 按固定随机样本做 GLM 语义复核 → 导出训练/验证集。

沿用旧标注脚本 `scripts/relabel_conclusion_to_exx.py` 的设计：从原索引补齐可见证据全文、只让教师读取问题与证据、生成 Exx 三种动作、按样本 ID 断点续跑，最后导出按 split 分开的 ShareGPT 训练集。沿用 `scripts/compress_exx_sft_with_api.py` 的紧凑事实结构，但这里是从证据重新标注，不把旧答案当标注输入。事实是否成立交给 GLM 语义抽检；本地只检查 JSON/动作/引用编号结构，不增加剧情关键词判题或生产校验规则。

## 默认设置

- 请求普通 BigModel API，模型 `glm-5.3`，`thinking=enabled`、`reasoning_effort=high`。
- 显式选择 `config.coding_plan.json` 可切换官方 Coding Plan 端点；不会自动回退普通计费端点。该配置为单并发、每请求最多三次尝试，先用 `--preflight-only` 检查认证和模型权限。
- 2 路并发；每请求最多 3 次尝试；失败指数退避，最长 30 秒。
- 一批累计最多 400 次请求尝试、600 万 token 预算。包括重试和抽检，不是 400 个成功标签。
- 10% 标签用相同完整证据交给 GLM 再核对动作和引用。这个复核不是独立人工金标。
- 每次输出最多 8192 token；若结束原因为 `length`，隔离原始输出，不截短标签、不自动修补。
- 问题、20 段全文、原始回复、模型/请求 ID、usage、输入哈希和错误类别都有记录；不保存 API key、请求授权头或隐藏思维链。

`accounted_tokens` 是预算账，不等同于实际消费：有合法 usage 时用实际 total_tokens；超时或未知计费状态保留保守预约。预约按输入 UTF-8 字节界限加输出上限估算，崩溃后也不释放未知请求。没有人民币价格估算或自动充值。

## 输入与隔离

`prepare --source` 可重复指定，支持：

- JSON/JSONL 任务：question、evidence/documents（doc_id/document_id/id）、round。
- 旧 ShareGPT 记录：只读用户问题和证据文档 ID，不读旧答案用于标注。

每段正文都重新从 `--documents` 的 `clean_text` 取全文。保留所有原候选文档，再在原来源范围内追加邻居至 20 段，随机排列 E 编号。不是实时检索 benchmark；旧文本或答案不作为新标签。

没有够 20 段、超上下文预算、包含排除来源的样本整条跳过，不截断证据。`--single-family` 是明确的活动内试验范围，遇到跨活动输入整条排除，不悄悄丢掉其中某些文档。

拆分连接**全部来源 family 和重复问题**，验证集与训练集的 family、文档和问题均不交叉。`--holdout-family` 可重复指定，保护早先验证来源；其整个关联分组不能进训练。

历史三轮任务可用 `--round-policy phase-to-two`（默认）映射到当前两轮协议：仍能检索 → `1/2`，最终轮 → `2/2`。源轮次保留在 `source_round`；标签全部重新生成，不复用旧动作。`preserve` 则保留原轮次，只适合明确需要历史协议的独立数据集。

重复问题在同一新轮次下只保留一次。原始独立问题数、样本数及剔除原因分别统计，不把多轮样本宣称为独立新题。

## 准备与运行

下面在仓库中执行；Python 环境需有 `tokenizers`。准备阶段使用本地模型 tokenizer 做保守上下文筛选，不把它的 token 数当 GLM 账单。大型数据的 `--out` 必须指向挂载盘目录。

```bash
python scripts/glm_annotation_pipeline/pipeline.py prepare \
  --source /path/to/tasks.json \
  --source /path/to/legacy_train.json \
  --documents /path/to/index/documents.jsonl \
  --tokenizer /path/to/tokenizer.json \
  --out /mounted/disk/glm_full20_labels_new \
  --train 300 --val 50 --single-family \
  --holdout-family activities/act18mini \
  --holdout-family activities/act19mini \
  --holdout-family obt/main
```

若数量不足，默认报错；显式 `--allow-fewer` 才按实际数量产出，并记录 requested/planned。

先在当前环境安全设置 `BIGMODEL_API_KEY`，不要把 key 写进命令历史、配置或日志。

一次真实样本预检，通过后后台完成整批：

```bash
python scripts/glm_annotation_pipeline/start_pipeline.py \
  --out /mounted/disk/glm_full20_labels_new \
  --config scripts/glm_annotation_pipeline/config.json
```

启动器将配置固定为输出目录中的 `launch-config-*.json` 快照，预检和后台进程使用同一份快照。后续修改共享配置不会改变正在运行的批次。启动器只在本次预检新增已接受标签时启动后台，不能用之前的成功记录掩盖本次失败；整批已接受时不会额外调用 API。

只预检：追加 `--preflight-only`。直接前台运行或断点续跑：

```bash
python scripts/glm_annotation_pipeline/pipeline.py run \
  --out /mounted/disk/glm_full20_labels_new \
  --config scripts/glm_annotation_pipeline/config.json
```

每次只处理少量未完成样本：追加 `--limit 10`。成功样本、成功请求缓存不会重复计费调用。完成语义复核前不会把对应样本导出为已接受标签。

余额/资源包不足（1113）、认证/配置错误立即停止并保存。普通限流、5xx、网络错误有限重试；连续失败断路。内容过滤或语义复核失败进入 quarantine，不替换事实，不自动绕过。

若只是额度恢复，原命令即可续跑。预算上限需要提高时编辑配置里的 `max_api_calls` / `max_total_tokens`；累计预算不会重置。模型、思考策略、输出上限或抽检策略变化需要新输出目录，避免混用缓存。

`attempts` 是单阶段本次运行的有限重试次数，可以在续跑时调整，不改变教师提示词、标签缓存或已接受样本。格式错误保留原始回执及 `validation_error`；如模型额外包了一层 `answer`，不会自动拆壳接受，而是用同一份 prompt 重新请求。累计请求和 token 上限始终包括失败与重试。

端点变化也需要新输出目录，不能覆盖原批次的设置或 SQLite 账本。复制不可变的 `requests.jsonl` 和准备信息即可复用同一题集；新端点的缓存、回执和预算单独记录。端点连通不证明套餐允许自定义批量标注，也不证明实际计入套餐；需确认官方使用范围和账号权益。不要伪装客户端、绕过限流或自动改用普通余额。

只复用准备好的题集、另起一个批次（不调用 API，不复制旧标签/SQLite/预算）：

```bash
python scripts/glm_annotation_pipeline/pipeline.py fork-inputs \
  --source /mounted/disk/glm_full20_labels_original \
  --out /mounted/disk/glm_full20_labels_new_endpoint
python scripts/glm_annotation_pipeline/start_pipeline.py \
  --out /mounted/disk/glm_full20_labels_new_endpoint \
  --config scripts/glm_annotation_pipeline/config.coding_plan.json
```

源目录只读，新目录必须不存在；复制前后验证输入哈希和训练/验证隔离。扩大数据量仍需用 `prepare --train ... --val ...` 重新准备题集；`fork-inputs` 只复用同一题集，不生成新问题。

## 状态与产物

```bash
python scripts/glm_annotation_pipeline/pipeline.py status --out /mounted/disk/glm_full20_labels_new
python scripts/glm_annotation_pipeline/pipeline.py inspect --source /path/to/tasks.json
python scripts/glm_annotation_pipeline/pipeline.py export \
  --out /mounted/disk/glm_full20_labels_new --config scripts/glm_annotation_pipeline/config.json
```

- `requests.jsonl`、`manifest.json`：不可变问题/证据任务及来源哈希。
- `ledger.sqlite3`：调用前持久化预算、结果缓存、断点状态；运行时应连同 WAL 保留，不能只复制活动中的 SQLite 主文件当可靠备份。
- `status.json`、`launch.json`：阶段、调用数、账面 token、错误原因、真实后台 PID。
- 运行进度同时包含 `total_tasks`、`accepted`、`quarantined`、`remaining_tasks`、`unstarted_tasks`、更新时间和端点。剩余数不包含已隔离样本，不能把“全部处理”误认成“全部标签被接受”。
- `api_receipts.jsonl`：脱敏 API 回执。
- `train.jsonl`、`val.jsonl`：P-tuning 格式；`train.json`、`val.json`：ShareGPT 格式。
- `quarantine.json`：失败/隔离样本原记录，不混入训练。

导出命令不调用 API。单一数据目录有进程锁，避免并发写入。只读 status 不会把正在调用的请求误标成中断。

`complete` 表示全部计划样本被接受，不表示正确率达标。模型产出的标签仍需语义抽查；manifest 默认 `review_required_before_training=true`，不会自动触发训练或更换发布模型。部分成功集也可导出，但不能把它误报为整批完成。

## 验证

```bash
cd scripts/glm_annotation_pipeline
python -m unittest -v test_pipeline.py
```

当前 47 项测试，无真实 API 请求：完整原文、旧答案不入 prompt、轮次映射、全来源/问题隔离、缓存续跑、余额停机、预算跨重启、并发上限、重试、原始输出隔离、复核、脱敏、后台启动预检、官方端点白名单和缓存隔离，以及题集复制、实时计数、历史标签不能掩盖失败、已完成批次零调用和配置快照固定、额外输出包装的有限重试、调整重试次数不重复标注。测试通过不计入 GLM 真实标注数。
