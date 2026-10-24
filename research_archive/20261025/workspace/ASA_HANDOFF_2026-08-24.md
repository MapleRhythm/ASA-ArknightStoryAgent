# ASA 项目进度与后续任务交接

更新时间：2026-08-25 14:15 CST  
交接目的：让下一位 AI 不重做已完成实验，直接收口 reranker 评测、完成 4B embedding 消融，并推进 story-family 数据扩增。

## 1. 最重要的当前结论

ASA 的最终目标仍是降低弃答率和幻觉，并将 GPU 上单条完整回答控制在 20 秒以内。当前阶段主要在处理全链路前半段的两个瓶颈：候选召回和 reranker 排序。回答生成、证据充分性、引用—主张一致性仍需在检索定型后继续做端到端复评。

目前可以确认：

- `listwise v2` 相对线上 `v6` 的排序能力有真实提升。三折、story-family 零泄漏 LOFO（n=208）中，D_hit@1 提升 9.62pt、NDCG@1 提升 8.41pt、X-rate@1 降低 7.21pt，三个 fold 均未退化。
- 旧 blind67 的 top-1 退化不能直接作为否定结论，因为旧 blind67 是只认原始 gold 段的二值标签，而训练和 LOFO 使用 GLM query-only 五档标签。两者评测目标不一致。
- 已用与训练相同的 `GLM-5.3 + query-only v3.5` 协议完成 blind67 第一轮重标。当前有 64/67 个唯一成功结果，剩 3 个失败题需要断点续跑，尚未形成最终同协议结论。
- `Qwen3-Embedding-4B` 已完整下载到挂载盘，但 4B dense 索引尚未构建，也尚未证明召回优于 0.6B。
- 扩增到大多数 story-family 尚未开始。现训练 teacher 数据仅覆盖 14 个 family，val 覆盖另外 4 个 family，这确实是当前泛化实验的主要数据覆盖缺口。

因此，当前不能把 `listwise v2` 写成“已经正式上线”，也不能把 4B embedding 写成“已提升召回”。正确状态是：listwise v2 已通过同协议 val 和 LOFO，正在等待 blind67 同协议最终复核；4B embedding 等待严格单变量消融。

## 2. 机器、仓库和安全约束

主机与主仓库：

```text
${REMOTE_USER}@${REMOTE_HOST}
${REMOTE_HOME}/ASA-ArknightStoryAgent
branch: dev-training
HEAD: b18616e
```

241 是主仓库。248 只用于同步训练/实验计算。2026-08-25 重新检查时 A100
空闲，现正用于 4B sidecar；不能杀其他用户进程。

本机 SSH askpass：

```text
${ASA_ROOT}/.codex_tmp/ssh_askpass_asa.sh
```

连接模板：

```bash
env SSH_ASKPASS=${ASA_ROOT}/.codex_tmp/ssh_askpass_asa.sh \
  SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  setsid -w ssh ${REMOTE_USER}@${REMOTE_HOST} '<command>'
```

不要把密码或 API key 写入脚本、日志、交接文档或 Git。GLM key 已保存在 241 的环境文件中，使用：

```bash
source ~/.bigmodel_env
```

所有大型数据、模型、索引、训练结果和评测结果必须写到挂载盘。本文后续为便于阅读使用 `/mnt/...`，它始终表示下面这个完整根目录：

```text
${REMOTE_ASA_ROOT}/rank_v8_coverage
```

快照时挂载盘约 15T，总使用约 579G。不要把大产物写到系统根目录，不要覆盖线上索引和线上模型。

2026-08-25 12:55 快照时 241 GPU：

```text
GPU0 RTX 4090: 23113 MiB，被其他用户占用
GPU1 RTX 4090: 5325 MiB，被其他用户服务占用
GPU2 RTX 4090: 23135 MiB，被其他用户占用
```

状态会变化；启动任何 GPU 任务前重新执行 `nvidia-smi`。不得停止无法确认归属的进程。

## 3. 最新 Claude 记录

最新且与本任务直接相关的 Claude 会话：

```text
${LOCAL_HOME}/.claude/projects/-home-maplerhythm-ASA/93702bdd-4d04-4766-857f-ff5882339c03.jsonl
```

该会话最后完成了 listwise v2、正则化对照和三折 LOFO；随后启动 blind67 同协议重标与 4B 模型下载。不要只采信会话中较早的“blind67 退化、不能上线”段落；后续 LOFO 已修正了该判断，原因是旧 blind67 标注协议不同。权威汇总文档是：

```text
${REMOTE_ASA_ROOT}/rank_v8_coverage/docs/listwise_v2_lofo_conclusion_20260824.md
```

## 4. 已完成：新候选池和 query-only 标注

候选池由 `Qwen3-Embedding-0.6B + Sparse v2` 生成，649 题，每题固定 120 候选。标注协议只接收 query、query_type、source_name、story_family 和候选文本，不允许 answer、gold evidence、required_evidence、claim 或原始 rank/score 进入 teacher 管线。

候选输入：

```text
/mnt/.../data/query_answerability_v3/qwen3_sparse_v2_input_20260822/train_safe.jsonl
/mnt/.../data/query_answerability_v3/qwen3_sparse_v2_input_20260822/val_safe.jsonl
```

GLM query-only v3.5 标注与物化数据：

```text
/mnt/.../data/query_answerability_v3/qwen3_sparse_v2_glm_5_3_high_v3_5_20260822/
```

精确计数：

```text
train input:       533
train labels:      523 unique
train listwise:    514 unique，14 story-family
val input:         116
val labels:        116 unique
val listwise:      114 unique，4 story-family
```

teacher 配置：

```text
provider: bigmodel-coding-plan
endpoint: https://open.bigmodel.cn/api/coding/paas/v4/chat/completions
model: glm-5.3
reasoning_effort: high
temperature: 0
max_tokens: 32768
protocol: query_answerability_v3
prompt: query_answerability_v3.5_positive_caps
```

## 5. 已完成：listwise v2 训练和评测

正式模型：

```text
/mnt/.../runs/query_listwise_v2_20260823/full_train
```

初始化模型（当前线上 v6）：

```text
${REMOTE_HOME}/ASA-ArknightStoryAgent/model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch
```

训练配置：

```text
max_length=640
group_size=16
BF16 + gradient checkpointing
2 epochs
lr=1e-5
```

最佳 checkpoint 是 epoch 1.48 / checkpoint-96。`full_train` 根目录的 `model.safetensors` 已是最佳权重，不需要从最后一个 checkpoint 手工恢复。

同协议 val（116 题）结果：

| 指标 | v6 | listwise v2 | 变化 |
|---|---:|---:|---:|
| D_hit@1 | 0.579 | 0.675 | +0.097 |
| D_hit@5 | 0.851 | 0.895 | +0.044 |
| NDCG@1 | 0.673 | 0.781 | +0.108 |
| NDCG@10 | 0.602 | 0.663 | +0.061 |
| X-rate@1 | 0.097 | 0.079 | -0.018 |

三折 LOFO，同协议且 held-out story-family 在训练中完全不可见，合计 n=208：

| 指标 | v6 | listwise v2 | 变化 |
|---|---:|---:|---:|
| D_hit@1 | 0.4808 | 0.5769 | +0.0962 |
| D_hit@5 | 0.8221 | 0.8654 | +0.0433 |
| NDCG@1 | 0.6170 | 0.7011 | +0.0841 |
| NDCG@10 | 0.5486 | 0.5880 | +0.0394 |
| X-rate@1 | 0.2019 | 0.1298 | -0.0721 |

三个 fold 都没有退化。逐 fold 结果位于：

```text
/mnt/.../runs/query_listwise_v2_20260823/lofo_act25side_eval/
/mnt/.../runs/query_listwise_v2_20260823/lofo_act17side_eval/
/mnt/.../runs/query_listwise_v2_20260823/lofo_act14side_eval/
```

`full_train` 下保留了 checkpoint-64/96/128/130；每个 checkpoint 还含约 4.5G optimizer。它们是后续可清理对象，但本次没有删除。确认根目录最佳权重、复现需求和 checksum 后，优先保留 checkpoint-96，其他 checkpoint 再做安全清理。

## 6. 正在收口：blind67 同协议重标

冻结且无泄漏的输入：

```text
/mnt/.../data/query_answerability_v3/blind67_relabel_input.jsonl
```

输入审计：67 题 × 120 候选，67 个唯一 query，无监督字段泄漏，候选平均约 426 字符。

输出目录：

```text
/mnt/.../data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824
```

2026-08-24 16:38 CST 快照：

```text
第一轮进程:     已自然退出
成功结果:       64 rows / 64 unique
失败日志:       3 rows / 3 unique
冻结输入未成功: 3 queries
第一轮 token:   3,578,801 total
```

三个未成功 query：

```text
黍和沉默的樵夫之间是什么关系？为什么黍称他为'神农'？
斯第奇·画布为什么要离开际崖城去地上？
夕为什么要帮助左乐对付绩？
```

前三个已有失败原因：

```text
黍...神农：finish_reason=length
斯第奇...地上：support quote 校验失败
夕...左乐：GLM HTTP 429 / code 1305，模型访问量过大
```

标注器的断点逻辑：成功输出中的 query 才算 completed；再次用同一命令运行时会自动跳过 64 个成功结果，只重跑失败题。failure 文件会追加历史失败记录，不影响成功结果；不要删除或覆盖它。

当前运行命令等价于：

```bash
source ~/.bigmodel_env
BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
${REMOTE_HOME}/miniconda3/envs/train/bin/python \
  "$BASE/scripts/label_query_answerability_candidates.py" \
  --input "$BASE/data/query_answerability_v3/blind67_relabel_input.jsonl" \
  --output "$BASE/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824/blind67_labels.jsonl" \
  --failures-output "$BASE/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824/blind67_labels.failures.jsonl" \
  --model glm-5.3 \
  --endpoint https://open.bigmodel.cn/api/coding/paas/v4/chat/completions \
  --api-key-env BIGMODEL_API_KEY \
  --provider bigmodel-coding-plan \
  --temperature 0 \
  --reasoning-effort high \
  --max-tokens 32768 \
  --max-workers 3
```

### blind67 接手后的第一优先级操作

1. 按 `(source_name, query)` 统计成功唯一数，不能只看 `wc -l`。
2. 原命令再运行一次；建议先降为 1 worker，避免 429。保持 model、协议、reasoning effort、temperature 和候选池不变。
3. 如果 `length` 或 quote 校验仍反复失败，先检查 API 原响应和协议修复逻辑；不要手工伪造标签。只有经过相同确定性校验的成功结果才能进入评测。
4. 达到 67 个唯一成功结果后，物化一个新的、冻结顺序、无重复文件。不要覆盖旧二值 blind67 结果。

物化模板：

```bash
BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
IN="$BASE/data/query_answerability_v3/blind67_relabel_input.jsonl"
SRC="$BASE/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824/blind67_labels.jsonl"
OUTDIR="$BASE/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824/materialized"
mkdir -p "$OUTDIR"
${REMOTE_HOME}/miniconda3/envs/train/bin/python \
  "$BASE/scripts/materialize_priority_teacher_shards.py" \
  --input "$IN" \
  --teacher "$SRC" \
  --output "$OUTDIR/blind67_same_protocol.jsonl" \
  --manifest "$OUTDIR/blind67_same_protocol.manifest.json" \
  --expected-candidates 120
```

物化器会检查 67 题覆盖、候选数量、候选池指纹和重复 key。标注脚本已经对每条成功结果执行协议/引文确定性校验。仍建议增加一个单 split 审计报告，检查：67 unique、每题 120 candidates、teacher ranking 恰好 120 个唯一 ID、标签值合法、每个 D/P/S quote 是候选文本的逐字子串。

### blind67 同协议最终评测

输出到全新目录：

```text
/mnt/.../runs/query_listwise_v2_20260823/blind67_same_protocol_20260824/
```

命令模板：

```bash
BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
INPUT="$BASE/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824/materialized/blind67_same_protocol.jsonl"
OUT="$BASE/runs/query_listwise_v2_20260823/blind67_same_protocol_20260824"
EVAL="$BASE/scripts/evaluate_query_answerability_reranker.py"
PY=${REMOTE_HOME}/miniconda3/envs/train/bin/python
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 "$PY" "$EVAL" \
  --input "$INPUT" \
  --model ${REMOTE_HOME}/ASA-ArknightStoryAgent/model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch \
  --output "$OUT/v6_result.json" \
  --device cuda --max-length 1024 --batch-size 16 \
  --cutoff 1 --cutoff 5 --cutoff 10 --cutoff 32

CUDA_VISIBLE_DEVICES=0 "$PY" "$EVAL" \
  --input "$INPUT" \
  --model "$BASE/runs/query_listwise_v2_20260823/full_train" \
  --output "$OUT/listwise_v2_result.json" \
  --device cuda --max-length 1024 --batch-size 16 \
  --cutoff 1 --cutoff 5 --cutoff 10 --cutoff 32
```

核心判据是 D_hit@1、D_hit@5、NDCG@1/10、X-rate@1，并结合 LOFO 判断。不要把旧二值 blind67 指标与新五档结果直接横比。如果同协议 blind67 也改善，listwise v2 才具备进入线上 A/B 的完整离线证据；仍需测实际候选分布上的延迟和端到端回答质量。

## 7. 待做：Qwen3-Embedding-4B 单变量消融

4B 模型已完整下载：

```text
/mnt/.../models/Qwen3-Embedding-4B
总大小约 7.6G
model-00001-of-00002.safetensors 约 4.97G
model-00002-of-00002.safetensors 约 3.08G
```

现有 0.6B 基线索引：

```text
/mnt/.../indexes/qwen3_embedding_0.6b_1024_sparse_v2
documents: 54,252
embedding dim: 1024
```

4B 第一版必须使用 Matryoshka 1024 维截断，以保持索引维度和体积可比；Sparse v2、documents、query prompt、query max length、融合参数全部冻结，只替换 dense document/query encoder。

本机已准备 dense-only sidecar 脚本，且 `python3 -m py_compile` 已通过：

```text
${ASA_ROOT}/.codex_tmp/build_dense_sidecar_index.py
```

脚本尚未复制到 241、尚未运行、尚未提交。应复制到：

```text
${REMOTE_HOME}/ASA-ArknightStoryAgent/scripts/build_dense_sidecar_index.py
```

目标索引目录：

```text
/mnt/.../indexes/qwen3_embedding_4b_1024_sparse_v2
```

该脚本只重建 `faiss.index`，并软链接冻结的 `documents.jsonl`、`bm25_tokens.pkl`、`sparse_index.pkl`、`operator_aliases.json`；默认拒绝覆盖已有 dense 索引。它会把 query prompt、1024 截断和最大长度写入新 `index_meta.json`。当前 retriever 已确认会读取这些 metadata，并在 query encode 时使用相同 prompt 和 `truncate_dim`。

先做 model load/encode smoke，再正式构建：

```bash
BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
cd ${REMOTE_HOME}/ASA-ArknightStoryAgent
CUDA_VISIBLE_DEVICES=0 GOLDENGLOW_USE_TRAIN_OVERRIDE=1 \
  ${REMOTE_HOME}/miniconda3/envs/train/bin/python \
  scripts/build_dense_sidecar_index.py \
  --source-index "$BASE/indexes/qwen3_embedding_0.6b_1024_sparse_v2" \
  --output-dir "$BASE/indexes/qwen3_embedding_4b_1024_sparse_v2" \
  --embedding-model "$BASE/models/Qwen3-Embedding-4B" \
  --device cuda --batch-size 8 \
  --embedding-dim 1024 --max-seq-length 1024
```

所有 stdout/stderr 应重定向到 `$BASE/logs/`，不要放仓库或根盘。若 OOM，先降 batch size，不要改 embedding dim、语料或其他变量。4B 权重本身约占 8G，能在空闲 4090 上构建，但是否适合在线服务还取决于与生成模型共存时的显存和延迟。

### 0.6B 与 4B 的公平召回评测

两个数据集都跑：

```text
hard59: data/processed/evidence_chain_reranker/failure_hard_pool_v1/failure_hard_pool_listwise.jsonl
easy200: data/processed/evidence_chain_reranker/failure_hard_pool_v1/sample200_listwise.jsonl
```

两边固定参数：

```text
dense_top_k=120
sparse_top_k=120
dense_weight=0.5
sparse_weight=1.0
dense_min_quota=20
sparse_min_quota=100
fusion_top_k=120
reranker_candidate_top_k=120
minirag_top_k=0
minirag_weight=0
neighbor expansion=false
skip_rerank=true
oracle_sources=true
hit_mode=segment
segment_coverage_threshold=0.9
```

必须用 `segment` 口径同时重跑 0.6B 与 4B，不能把新结果与旧 whole-chain trigram 结果直接比较。输出中保留 per-query records，用于查 dense 独占命中、sparse 独占命中和融合丢失。

命令模板（分别替换 INDEX、EMBED、SET、TAG）：

```bash
CUDA_VISIBLE_DEVICES=0 GOLDENGLOW_USE_TRAIN_OVERRIDE=1 \
${REMOTE_HOME}/miniconda3/envs/train/bin/python scripts/evaluate_retrieval_recall.py \
  --listwise "$SET" --output "$OUTPUT" --device cuda \
  --index-dir "$INDEX" --embedding-model "$EMBED" \
  --reranker-model "$BASE/models/__disabled_reranker__" \
  --top-ks 1,5,10,20,50,120 \
  --dense-top-k 120 --sparse-top-k 120 \
  --dense-weight 0.5 --sparse-weight 1.0 \
  --dense-min-quota 20 --sparse-min-quota 100 \
  --fusion-top-k 120 --reranker-candidate-top-k 120 \
  --minirag-top-k 0 --minirag-weight 0 \
  --skip-rerank --oracle-sources --include-records \
  --hit-mode segment --segment-coverage-threshold 0.9 \
  --tag "$TAG"
```

`__disabled_reranker__` 应为不存在路径，避免 `--skip-rerank` 时仍加载 reranker 占显存。

除 Recall@K/MRR 外，必须记录：模型冷启动时间、query encode 的稳态 p50/p95、总 retrieval p50/p95、峰值显存。线上完整回答目标小于 20 秒，因此即使 4B recall 略高，若其额外显存或延迟阻碍与生成模型共存，也不应直接替换 0.6B。可以将 4B 作为 teacher/离线挖掘模型，线上保留 0.6B。

旧 0.6B whole-chain hard59 仅作历史参考，不能与新 segment 指标横比：

```text
fusion R@120:       0.6271
dense R@120:        0.6271
sparse R@120:       0.6780
source union R@120: 0.6949
```

它提示的关键问题是：当前固定 120 的 fusion pool 会丢掉 dense/sparse 联集中的约 4–7pt，未必只是 embedding 不够强。完成 4B 单变量消融后，下一组应做：

1. fusion pool 120 → 160，再由 reranker 压回线上 top-k，评估 recall/延迟。
2. 自适应 dense/sparse 配额，而不是所有 query 固定 20/100。
3. 按 query type 或 sparse 置信度门控。
4. 强制保留 source union 中的独占候选。
5. MiniRAG 继续禁用；此前图扩展收益差且增加耗时，不要在当前单变量实验中重新加入。

## 8. 待做：扩增到大多数 story-family

该任务尚未实施。不要直接从正文随机截一句或用单一模板批量造问题，这会造成弱问题、答案自证和虚假的训练提升。

建议流程：

1. 先建立语料 story-family inventory，并统计当前 649 题对 family、活动、query type 的覆盖。当前已知训练 listwise 只覆盖 14 family，val 4 family，但全语料 family 总数尚未严格审计。
2. 固定 family-level split，预留一批完全不进入训练的 holdout families；所有生成、标注和评测都按 family 隔离，防止同故事泄漏。
3. 对未覆盖/低覆盖 family 分层采样。第一轮每个 family 生成 2–4 个高质量、自然用户会问的问题，平衡 fact/relation/causality/reasoning/reveal 等类型。
4. 问题生成阶段可以阅读故事以产出问题，但进入 reranker teacher 的冻结输入必须再次剥离 answer、reference answer、gold evidence、required_evidence、claim、原始 rank/score 等字段。
5. 使用与线上一致的 `BM25/Sparse v2 + dense` 召回生成每题完整 120 候选，再通过现有 query-only audit。
6. 用同一 query-only v3.5 协议做 GLM 标注；先 smoke、审计标签分布和失败率，再全量。API 调用成本很高，先算 family 数、题数和 token 预算。
7. 训练集、验证集、blind/LOFO 必须按 story-family 完全隔离。对新增模型至少报告 seen-family、held-out-family 和 hard retrieval 三类结果。

合理的首轮目标不是“每个故事大量造题”，而是覆盖大多数有效 family，同时保持独立 holdout。先做覆盖广度，再对 hard family 定向加密。

## 9. 论文方向：从全链路提炼研究问题

挂载盘已有三份论文方向材料和一份充分性先导实验：

```text
/mnt/.../docs/asa_paper_survey_20260822.md
/mnt/.../docs/asa_proposal_direction_A_20260823.md
/mnt/.../docs/asa_proposal_v2_DEF_20260823.md
/mnt/.../docs/sufficiency_probe_results_20260823.md
```

这些材料的观点有先后修正关系。较早材料曾优先推荐“回答/续检索/拒答三择决策”，但后续文献复查发现 adaptive retrieval、search/stop policy、充分性 critic 和 RL 搜索已经非常拥挤；ASA 当前只有 38 条 KTO 对，也不足以单独支撑该方向。下一位 AI 不应只读最早的 proposal 后继续把三择决策当唯一主线。

### 9.1 推荐主线：证据链充分性是集合性质，并用缺口驱动定向续检索

暂定题目：

> **From Relevance to Sufficiency: Structured Evidence-Set Diagnosis for Multi-hop Narrative RAG**  
> 从候选相关性到证据充分性：面向叙事多跳 RAG 的集合级诊断与缺口驱动检索

核心问题不是“某一条候选是否相关”，而是：给定 query 和 top-K 证据集合，回答所需的多段证据链是否已经完整；若不完整，具体缺哪一环，下一轮应该检索什么。

这是最贴合 ASA 当前真实瓶颈、也最有方法论文潜力的方向。ASA 已有的先导结果为：

```text
sample200，prompt top-12：
full     27 / 200 = 13.5%
partial  99 / 200 = 49.5%
none     74 / 200 = 37.0%

top-120 全池：
full     62 / 200 = 31.0%
```

即使证据进入 120 候选池，经过融合、重排和 prompt 截断后仍会丢失大量完整链。更关键的是，常见单候选分数池化几乎不能预测“不充分”：

| 特征 | AUROC |
|---|---:|
| fusion max | 0.465 |
| fusion mean | 0.467 |
| top1-top2 gap | 0.455 |
| fusion std | 0.547 |
| 高分候选数 | 0.549 |

这已经形成一个很好的 motivation：**相关性分数高不等于证据集合足够回答问题，充分性不能由单候选分数简单池化得到。**

要把该方向做成论文，贡献不能停留在 full/partial/none 三分类。建议形成完整闭环：

1. **集合级充分性预测**：输入 query + top-K 候选集合，输出 `full / partial / none`，并做概率校准。
2. **结构化缺口定位**：对 partial 集合输出缺失的实体、事件、原因、结果、关系或时间环，而非只给低置信度。
3. **缺口驱动续检索**：把缺失环转成 follow-up query，进行一轮受预算约束的定向检索。
4. **安全停止决策**：第二轮后根据集合充分性选择 answer 或 abstain，评估能否同时降低幻觉与弃答。
5. **延迟约束**：将 20 秒端到端目标作为方法约束，报告质量—检索轮数—延迟 Pareto 曲线。

建议的模型层次：

- 无训练基线：max/mean/count/gap、reranker 分数池化、证据数量阈值。
- 强基线：逐候选 NLI 后池化、LLM-as-a-judge、query + 拼接 top-K 的普通分类器。
- 方法组：candidate encoder + permutation-invariant set encoder，联合预测集合充分性与候选贡献；或者让 4B 模型 LoRA 输出充分性和缺口结构。
- 闭环组：集合判官 + 缺口 query generator + 第二轮检索。

关键指标：

```text
充分性：macro-F1、full-vs-not AUROC、ECE/Brier
缺口定位：missing-hop recall/precision、缺失实体/关系覆盖
检索闭环：第二轮后的 chain full-rate、evidence recall@K
回答效果：supported answer accuracy、claim support、hallucination、abstain
系统成本：额外检索次数、p50/p95 延迟、峰值显存
```

关键消融：

- 去掉集合交互，仅对单候选打分再池化。
- 去掉缺口定位，只用通用 query rewrite。
- 固定二次检索 vs 按充分性自适应触发。
- 去掉 reranker、改变 prompt top-K、改变候选覆盖率。
- seen-family、held-out-family、hard retrieval 三个分桶。

外部验证是必要条件。仅用明日方舟剧情容易被认为域太窄，建议至少补 HotpotQA、2WikiMultiHopQA 或 MuSiQue 中一个有 gold evidence chain 的公开数据集。方法可以先在 ASA 上开发，再用公开集验证迁移。

一个非常重要的数据边界：现有 query-only GLM teacher 的 D/P/S/X/I 标签适合训练 reranker，但它没有看到 gold answer/evidence，不能凭空证明整个证据链是否完整。充分性研究的 `full/partial/none` 标签应由隐藏 gold evidence chain 离线计算并由人工抽查；gold 只用于构造监督和评测，绝不能进入模型输入或 reranker teacher 输入。这两条管线必须物理隔离。

资源可行性：该方向不要求从头训练大模型。A100 可训练 set encoder 或 4B LoRA，三张 4090 可并行跑召回、基线和端到端评测。首篇论文无需 RL，先用监督学习把“诊断—补检索—安全回答”闭环做扎实。

### 9.2 第二方向：分级、替代证据感知的叙事检索基准

暂定题目：

> **Beyond Binary Gold: Leakage-Resistant Graded Evidence Ranking for Narrative QA**

当前 blind67 的评测口径冲突本身揭示了一个有价值的问题：二值 gold 只认一条原始证据链，但叙事文本中经常存在多段不同却同样能回答问题的候选。如果模型把替代有效证据排到 top-1，传统二值评测会判错，甚至可能反转模型优劣结论。

ASA 已有可形成基准的资产：

```text
649 个 query
每题 120 个线上分布候选
约 77,880 个候选判断
D/P/S/X/I 五档相关性
逐字 quote 佐证
query-only teacher，避免答案/gold/rank 泄漏
story-family 级零泄漏拆分
```

建议将贡献定义为三个部分：

1. **替代证据感知的分级标签**：区分 direct、partial、supporting、hard context、irrelevant，而不是只认单一 gold。
2. **防泄漏标注协议**：teacher 只看 query + 完整候选正文；候选匿名、移除原始 rank/score/gold/answer；输出 quote 经过逐字校验。
3. **story-family 泛化评测**：随机 query split 与 family holdout/LOFO 同时报告，量化常规随机拆分对叙事检索泛化的高估。

论文中的核心对照应包括：

- 二值原始 gold vs 五档 query-only 标签，是否会改变模型排序结论。
- pairwise vs listwise、binary vs graded、random split vs family split。
- teacher 看 gold/答案的泄漏条件 vs query-only 条件。
- BGE 基线、v6、listwise v2 和通用 LLM reranker。
- 自动 teacher 与人类复核的一致性、quote 有效率和各 label 的混淆矩阵。

当前 listwise v2 的同协议 val 与三折 LOFO 已经是这篇论文的关键方法验证：三个 held-out family 都改善，合计 D_hit@1 +9.62pt。但 blind67 同协议复评尚未收口，必须完成后再写最终结论。

该方向最大的风险是数据发布与版权。明日方舟原始剧情文本不应默认可以公开；如果不能发布候选全文，单纯发布 query 和候选 ID 会严重削弱可复现性。建议至少采取一种补救：

- 增加公版中文/英文小说数据，例如 Project Gutenberg 或合法开放语料。
- 在公开多跳/叙事数据集上复现同一标注与 family split 方法。
- 只发布允许发布的派生标注、构造/审计代码和可复现协议，并在投稿前确认数据许可。

还需做人工质量审计。不能只以 GLM 自己生成的标签作为 benchmark gold。建议分层抽取完整 query pool 和候选对，至少双人标注，报告 weighted kappa/Krippendorff alpha、teacher-human 一致性，以及替代有效证据的占比。若无法解决版权和人工标注问题，该方向更适合作为方法论文中的数据章节，而不是独立 Dataset & Benchmark 投稿。

### 9.3 可合并进第二方向的分析：证据粒度与评测判据

已有 hard59 结果显示，整条 gold chain 的字符重合判据会系统性低估 chunk-level 召回：

| 判据 | R@20 | R@32 | R@120 |
|---|---:|---:|---:|
| whole-chain trigram | 35.6% | 47.5% | 62.7% |
| E-segment coverage ≥ 0.9 | 55.9% | 59.3% | 76.3% |

这个方向可以形成一篇短论文或并入 9.2：研究 gold evidence 粒度、chunk 边界、标注改写和替代证据如何共同影响 retrieval evaluation。真正的贡献应是一个经过人工验证的、对 chunking 稳健的 evidence coverage 指标及误差分析，而不只是“换了一个阈值后分数更高”。

至少需要比较：exact span、whole-chain trigram、segment subsequence coverage、semantic entailment/LLM judge，并人工核查各判据的 false positive/false negative。所有模型必须在同一判据下重跑，禁止跨口径比较。

### 9.4 系统型方向：可验证证据承诺下的选择性 RAG

暂定题目：

> **Commit, Retrieve, or Abstain: Verifiable Evidence Commitments for Selective RAG**

ASA 的特色不是简单输出 `[E3]`，而是模型提交原子 claim、E-ID 和逐字 quote，服务端检查并重建答案，验证失败时禁止 raw fallback。这可用于研究 hallucination—abstention 权衡。

但“回答/检索/拒答”动作学习本身已有大量工作，不能再把三择动作当作主要创新。若继续此方向，差异必须集中在：

- action 同时附带可机械验证的证据承诺。
- verifier 对模型决策有服务端否决权。
- retrieve_more 必须输出结构化缺口，而不是泛化的“证据不足”。
- 在固定延迟/检索预算下优化 supported answer coverage。

实验主图应是 risk-coverage / hallucination-abstention 曲线，而不是只报 JSON 合法率。基线包括规则门限、生成后 verifier、无硬否决、无 quote、无缺口检索。当前 2,430 条 SFT 数据可作为起点，但只有 38 条 KTO 对，远不足以支持强偏好学习结论；需要扩充 action gold 和 preference 数据，并补公开多跳集验证。

还要明确：逐字 quote 校验只能证明“引文来自候选”，不能自动证明 quote 支持 claim。要避免把格式验证误写成语义 grounding，必须加入 claim-evidence entailment 的人工/模型复核，尤其覆盖否定、主体互换、因果方向、身份关系和时间顺序。

该方向适合在主线 9.1 完成后作为系统扩展，因为集合充分性与缺口定位可以为 retrieve/answer/abstain 提供更可靠的依据。

### 9.5 高风险后续：抗 reward hacking 的可验证 grounding RL

可以研究用硬 E-ID/quote/claim 验证器构造 RLVR 奖励，并与 lexical、NLI、LLM judge 奖励比较 reward hacking。但不建议把它作为当前第一篇论文：

- 逐字 quote 验证本身仍可能被“复制无关原文 + 错误 claim”攻击。
- 需要同时设计 claim entailment、答案效用和拒答校准奖励，否则模型可能通过全部拒答刷安全分。
- 要建立攻击集，覆盖错 E-ID、无关正确 quote、主体互换、否定翻转、因果倒置、拆分/合并 claim 和关键词堆砌。
- RL 对照多、方差大，虽然 3×4090 + 1×A100 可做 4B LoRA/GRPO 小规模实验，但复现实验成本明显高于监督主线。

只有在硬 verifier 的语义可靠性和反作弊评测集完成后，才建议进行 SFT vs preference optimization vs RLVR 的系统比较。

### 9.6 只能作为消融、不能单独包装成论文的工作

以下工作对 ASA 工程很重要，但单独不构成足够研究贡献：

- 将 Qwen3-Embedding-0.6B 换成 4B。
- 修 BM25 分词边界、增加 char/领域词/字段 lane。
- fusion top-120 改 160 或调整 dense/sparse 配额。
- 禁用低收益 MiniRAG。
- 单纯用 GLM 标签微调 BGE reranker。
- 将输出证据全文缩成 `[Exx]`。

它们应作为主论文的检索强基线、延迟消融或系统组件。若要把“融合”升级为方法方向，必须提出可泛化的、受候选预算和延迟约束的 source-aware coverage optimization，并在多个公开数据集上证明优于固定 RRF/配额，而不是只在 ASA 上调参。

### 9.7 推荐论文组合与执行顺序

最稳妥的组合是“一篇方法主线 + 一篇数据/分析副线”：

1. **主论文优先做 9.1**：集合级充分性 + 缺口定位 + 一轮定向续检索。现有先导结果已经提供强动机，且直接命中弃答率高和证据不完整的系统瓶颈。
2. **副线做 9.2 + 9.3**：query-only 分级证据排名、family holdout、替代证据与粒度稳健评测。先解决版权和人工复核，再决定独立投稿还是并入主论文。
3. **9.4 作为系统扩展**：把集合充分性接到 answer/retrieve/abstain 和服务端证据承诺上，形成端到端 risk-coverage 结果。
4. **9.5 最后再做**：只有监督闭环和 verifier 审计稳定后才上 RL。

建议下一位 AI 在工程实验同时维护一张论文证据矩阵，每个 claim 都绑定可复现实验：

| 论文 claim | 当前证据 | 还缺什么 |
|---|---|---|
| 充分性是集合性质 | sample200 池化 AUROC ≤ 0.549 | 人工审计、公开集复现 |
| prompt 截断造成链覆盖损失 | top-120 full 31% → top-12 13.5% | reranker v2 后重跑、多 K 曲线 |
| query-only graded labels 提升泛化 | val + LOFO 三折一致改善 | blind67 同协议收口、人类标签对照 |
| 二值 gold 会误判替代证据 | 旧/new blind67 口径冲突 | 逐题人工 adjudication、统计显著性 |
| 缺口检索降低弃答且不增幻觉 | 尚无闭环结果 | 训练 gap locator、端到端实验 |
| 硬证据承诺抑制幻觉 | 已有服务端验证实现 | claim entailment、人类误杀/漏检率 |

资源分配建议：A100 优先训练 set-level sufficiency model 或 4B LoRA；三张 4090 分别承担召回索引/评测、reranker/充分性基线、端到端生成与 verifier 消融。避免同时维护多条没有共同主表的训练线。

## 10. Git 和脏工作区

241 仓库快照有 46 条 dirty/untracked 状态，其中大量是用户或先前实验改动，不能整体提交。不要执行 `git add .`，不要 reset/checkout 覆盖用户改动。

本轮直接相关但尚未提交的候选文件包括：

```text
scripts/audit_query_answerability_candidates.py
scripts/build_query_answerability_candidates.py
scripts/train_query_listwise_reranker.py
tests/test_train_query_listwise_reranker.py
scripts/evaluate_retrieval_recall.py          # 含后续 segment hit 等改动，提交前逐段审查 diff
tests/test_evaluate_retrieval_recall.py
scripts/build_dense_sidecar_index.py          # 尚在本机 .codex_tmp，需复制
```

还有其他未跟踪 reranker v7 文件和大量 runtime/inference 改动，归属不清，不应混入本任务提交。

已提交的前一阶段召回改造：

```text
b18616e retrieval: add Qwen3 dense and multi-lane BM25 sidecar
```

最终提交原则：

- 先逐文件 `git diff` / `git diff --no-index` 核对。
- 只显式 stage 本任务确认过的脚本、测试和复现实验文档。
- 大模型、索引、JSONL、checkpoint、日志只留挂载盘，绝不进 Git。
- 先跑相关单元测试和 `py_compile`，再提交。

## 11. 推荐接手顺序

1. **收口 blind67**：等当前进程结束，补齐到 67 unique，物化、单 split 审计、同协议评测 v6 vs listwise v2。
2. **并行构建 4B sidecar**：确认 GPU0 仍空闲，复制并 smoke dense-only 脚本，构建 1024 维索引。
3. **0.6B/4B 单变量消融**：hard59 + easy200，统一 segment 口径，同时测稳态延迟和显存。
4. **再优化融合**：先测 source-union/fusion 120→160 和自适应配额，不要立即重新启用 MiniRAG。
5. **做 family inventory 与扩增设计**：固定 holdout family 后再生成问题和调用 GLM，避免数据泄漏。
6. **端到端复评**：用仓库外自拟问题跑完整回答链，分解召回、rerank、证据充分性、生成、引用校验和弃答门控；同时验证完整回答 <20s。
7. **谨慎提交**：仅提交本轮明确文件，附实验复现文档，不混入既有脏改动。

## 12. 最终上线判据建议

reranker 上线至少满足：

- 同协议 val 改善；已满足。
- story-family 零泄漏 LOFO 多 fold 一致改善；已满足。
- blind67 同协议不出现实质 top-1 退化；等待完成。
- 在实际线上候选分布上，Recall@K、D_hit/NDCG、X-rate 和弃答率共同改善。
- reranker 延迟与端到端完整回答仍满足 20 秒目标。

embedding 替换至少满足：

- hard59 和 easy200 的 segment Recall@K 有稳定、可解释提升。
- 增益来自 dense lane，而不是评测口径或候选池变化。
- fusion 后仍保留提升，不只是在 source oracle 上变好。
- 显存与 p95 延迟允许与生成模型共存；否则仅作为离线 teacher/挖掘模型。

系统最终目标不是单独刷高 reranker 指标。完成上述离线实验后，必须回到全链路：低弃答不能靠放松门限换取更多幻觉；正确方向是提高可回答证据覆盖、校准充分性判断，并对最终答案逐主张验证引用支持关系。

## 13. 2026-08-25 实时续跑补充

截至 13:52 CST，两个阻塞任务在不改变“241 为主仓”的前提下继续推进：

- 241 上旧 GLM 请求保持 HTTPS 连接超过 3000 秒且没有新增结果，已仅终止这组
  本任务进程；原 64 条成功标签逐字节保留。三条 pending 现已迁到 248，继续使用
  相同的 GLM-5.3 Coding Plan 端点、query-only v3.5 协议、完整 120 候选、
  temperature=0 和 high reasoning。自动 watcher 只有在校验到 67 unique 后才会回写
  241，并重新启动 materialize、single-split audit、v6/listwise-v2 finalizer。
- Qwen3-Embedding-4B、0.6B 和冻结 Sparse-v2 索引已由 241 直传 248。不要从
  Codex 工作站中转大文件：实测 VPN 中转约 0.6 MB/s，而 241→248 直传约
  111 MB/s。
- 4B sidecar 正在 248 空闲 A100 上构建。原 builder 会一次保留 54,252 条完整
  2560 维向量，再复制为 1024 维；现改为每 512 文档一块，并直接写 1024 维
  float32 memmap，以降低 host RAM 风险，同时保持相同的归一化 Matryoshka 向量。
  builder 现还会原子保存进度；14:07 从 10,240/54,252 恢复并把 batch 8 提至
  batch 16，显存约 25.1/40GB，运行稳定。
- 4B index 完成后，watcher 会把 `faiss.index`、`index_meta.json` 回写 241 挂载盘；
  hard59/easy200 × 0.6B/4B 将在 248 跑统一 segment 召回消融，四个 JSON 最终都
  回写 241 的 `runs/qwen3_embedding_4b_ablation_20260825/`。

当前工作站自动化文件：

```text
.codex_tmp/asa_blind67_sync_finalize_local.sh
.codex_tmp/asa_embedding_4b_sync_eval_local.sh
.codex_tmp/asa_embedding_4b_ablation_248.sh
.codex_tmp/asa_embedding_4b_build_248.sh
.codex_tmp/build_dense_sidecar_index.py
```

重要空间约束：248 `/mnt/store` 暂存模型后仅余约 2.2 GB。当前文件是运行中任务
必需输入，不能提前删除；确认 index、四份评测 JSON 均到达 241 后，应优先清理
248 上这次临时复制的 4B/0.6B 模型和冻结索引副本。

14:10 后 blind67 补标改为 3 worker 并行，仍是原协议/原候选/原模型，只将
`max_tokens` 从 32768 提至 65536、单请求 timeout 提至 900 秒。原因是串行第一题
两次 600 秒读超时会阻塞另外两题，且该题历史上曾 `finish_reason=length`。新的远端
watcher PID 记录在：

```text
${REMOTE_STORE}/rank_v8_coverage_241_mirror/logs/finalize_watcher_248.pid
```

248 为自动回写 241 创建了本任务专用临时 Ed25519 key，并只在 241
`authorized_keys` 添加了 comment 为 `asa-248-to-241-20260825` 的一行。所有结果
同步结束且确认到达 241 后，应按这个 comment 精确移除该 authorized-key 行，并删除
248 上 `.sync_248_to_241_ed25519*`；不要改动其他 SSH key。

## 14. 2026-08-25 17:48 续跑结果：4B 融合、reranker 与端到端

### 14.1 blind67 与 listwise v2 已收口

blind67 的 GLM-5.3 query-only v3.5 标注已完成 67/67 unique，输入候选池未变、无重复、
无错误。物化文件：

```text
/mnt/.../data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824/materialized/blind67_same_protocol.jsonl
```

同协议复评中，listwise v2 相比 v6：D_hit@1 +7.46pt、D_hit@5 +2.99pt、
NDCG@1 +3.98pt、NDCG@5 +4.55pt、X-rate@1 -1.49pt，p95 仅 +0.008s。
结合三折 story-family LOFO，listwise v2 已具备端到端灰度资格，但仍未替换线上 v6。

### 14.2 Qwen3-Embedding-4B 与正式融合

4B/1024 维索引已构建并同步到 241：

```text
/mnt/.../indexes/qwen3_embedding_4b_1024_sparse_v2/
```

4B dense 改善中前排召回，但不是所有 K 都优于 0.6B。更重要的是，dense+sparse 的
source-union 上界高于旧 RRF 融合。把 4B 配置从 `dense_weight=0.5, quota=20/100`
调到 `dense_weight=1.25, quota=40/80` 后，无 reranker 正式评测为：

| split | 指标 | baseline | tuned | 变化 |
|---|---|---:|---:|---:|
| hard59 | R@5 | 16.95% | 25.42% | +8.47pt |
| hard59 | R@20 | 54.24% | 57.63% | +3.39pt |
| hard59 | MRR | 0.1556 | 0.1838 | +0.0282 |
| easy200 | R@5 | 38.50% | 44.00% | +5.50pt |
| easy200 | R@20 | 67.50% | 71.00% | +3.50pt |
| easy200 | MRR | 0.2850 | 0.3178 | +0.0328 |

但是经过 listwise v2 的现有“文档打分 + 证据链二次打分 + 启发式加成”后，tuned
相对 baseline 的 top-1/MRR 轻微退化：

| split | baseline | tuned | 结论 |
|---|---|---|---|
| hard59 | R@1 0.2373 / MRR 0.2979 | R@1 0.2203 / MRR 0.2909 | R@5/10/20 各 +1.69pt，但 top-1 退化 |
| easy200 | R@1 0.4450 / MRR 0.5181 | R@1 0.4400 / MRR 0.5146 | R@5 +1.0pt、R@20 +0.5pt，MRR 退化 |

两组配对都新增 3 个覆盖、丢失 2 个覆盖；说明融合增益能进入 reranker，但当前后处理
未能稳定传递。因此不要直接上线 4B + `1.25/40/80`。完整 JSON/日志：

```text
/mnt/.../runs/qwen3_embedding_4b_ablation_20260825/rerank_fusion/
```

### 14.3 reranker 延迟与 640 消融

当前 search 并非只对 120 个文档打分。它还构造最多约 80 条证据链，再调用同一个
BGE 二次评分，一题实际可能接近 200 个 pair。1024/batch16 的稳态约 3.54--3.73s/query。

`max_length=640 + batch24` 仅提升约 7--12%：hard59 3.73→3.27s/query，easy200
3.54→3.29s/query；但两组 MRR、R@5/10/20 均小幅下降，并各丢失一个 top-20 命中。
不要直接改线上长度。更高优先级实验是：

1. 1024 长度下仅做 document-only rerank，关闭 chain model rescore/heuristics。
2. 再分别测 chain top-N=0/8/16/24，而不是当前最多 80。
3. 对 120/96/80/64 候选做质量—延迟曲线。

248 上已准备临时 document-only 脚本，但启动时 A100 被用户
`zhuhongwang` 的 Panacea 任务新占用约 27GB，加载 4B + reranker OOM。未停止该任务；
待 A100 空闲后重跑：

```text
${REMOTE_HOME}/asa_evaluate_retrieval_recall_doconly_248.py
${REMOTE_HOME}/asa_rerank_doconly_1024_b24_248.sh
```

### 14.4 端到端新问题评测正在运行

10 个仓库外问题与冻结运行配置：

```text
/mnt/.../runs/e2e_novel_questions_20260825/questions.txt
/mnt/.../runs/e2e_novel_questions_20260825/runtime.json
```

配置是 0.6B embedding + listwise v2 + 120 候选 + MiniRAG/邻居/章节扩展关闭，生成端
DeepSeek v4 Flash，严格 grounding。第一轮因 DeepSeek 余额不足统一 HTTP 402，日志
保留为 `inference_attempt1_ds402.log`；API 恢复后已续跑。

截至 17:48，前三题完成：39.88s、146.28s、26.58s；均远超 20s 目标。旧 trace 也显示
每轮将一个问题扩成约 15--16 条 query，全库 BM25 每条跑一次，是 30--45s retrieval
的主要来源。dense 已批量编码，sparse 仍逐 query 全库计分。必须做 query budget 2/4/6/8
和 BM25 batch/cache/倒排实现，而不是只压 reranker 0.5 秒。

前三题的质量归因：

- 血骑士题：召回到同章节但未保留村民凭红盔甲、健硕身姿认人的精确 chunk；“为何买田”
  原文本身也未明确说明。系统整题弃答，暴露多子问题无法 partial answer 的门控缺陷。
- 斯凯/雅斯彭题：正确 chunk 在 top evidence，但 quote 校验失败后最终输出了三个长而无关的
  “片段事实”，既不回答问题又占用长度；属于 verifier fallback 表达缺陷。
- 托希娅题：首轮正确回答，证据与结论基本匹配；但答案有重复句意和双句号，渲染需清理。

当前运行 PID：

```text
/mnt/.../runs/e2e_novel_questions_20260825/e2e.pid
```

### 14.5 当前最重要的全链路判断

1. **延迟首因是 query fan-out × 全库 sparse**，不是 embedding 或单次 reranker。
2. **召回错误仍存在，但 tuned 候选池增益被 chain rerank 抵消**；需先验证 document-only。
3. **高弃答的重要原因是集合级/子主张级充分性缺失**：一个子问题无证据会拖累整题。
4. **幻觉与弃答共因是 verifier 只有 quote/词法校验**：能证明引文存在，不能证明逐主张语义蕴含。
5. **`_has_answerable_evidence` 只按任一证据长度 >=80 判断**，不是真正的集合充分性。

论文主线仍建议：集合级充分性预测 + 缺失环定位 + 定向二次检索 + partial/selective answer
+ latency-aware routing。新的端到端案例可作为研究动机：从 query fan-out 预算、证据集合覆盖、
子主张可答性到 answer/retrieve_more/partial/abstain 的联合决策。

## 15. 2026-08-25 21:15 续跑结果：回答证据职责分离与子问题覆盖

本轮已在 241 主仓完成并提交：

```text
commit: 9e63000 inference: preserve direct evidence coverage
branch: dev-training
```

仅提交了以下四个文件，没有 stage 既有的配置、release、训练脚本或日志改动：

```text
api-mode/run_api_inference.py
src/goldenglow/inference/cpu_pipeline.py
tests/test_run_api_retrieval_config.py
tests/test_prompt_evidence_chain_boundary.py
```

### 15.1 已确认的根因与修复

1. ranking chain 只能给 reranker 排序，不能进入回答 prompt、citation、grounding 或 fallback。
   同一约 3,000 字 chain 曾复制到多个候选，既扩大 prompt，又把直接 chunk 截掉。
2. `_finalize_hits()` 已输出 anchor rescue → reranker 的目标顺序；prompt selector 不能再拿
   `fusion_score`、`rerank_score` 与 `evidence_chain_score` 跨尺度二次排序。
3. CRAG 不能只保留孤立高分句。本轮保留每个 seed 前后各一个 strip，并针对复合问题保留
   子问题对应的两个非相邻局部窗口，避免丢掉因果桥和枚举细节。
4. 新增无 API 延迟的 compound-subclaim pinning：只对显式多问句触发，从完整 finalized
   候选中为每个子问题预留直接 chunk，再按原 finalized 顺序替换 top-k 尾部。匹配仅检查
   direct chunk，绝不检查 ranking chain。
5. strict quote 硬校验通过后，不再让旧中文 token-overlap verifier 二次误杀答案；词法告警
   只留审计记录。speaker label/标点删减仍允许，但 quote 正文必须来自 prompt。
6. 增加泛称括号别名后校验：若证据没有明确写出 `巨物（X）/巨物是X/巨物即X`，则从答案
   和 supported fact 中删除模型自行补出的括号别名，保留原泛称并写 grounding warning。

### 15.2 真实案例验证

血骑士题的正确 chunk 原本已在最终候选第 14/15 名：

```text
activities/act13side/level_act13side_hidden_st03#chunk-0009
activities/act13side/level_act13side_hidden_st03#chunk-0012
```

新选择器将两段 pin 到 prompt 9/10；CRAG 继续保留：

```text
一身红盔甲、健硕身姿
矿石病不好治，所以才选离村庄最远的地
```

真实 DeepSeek Flash 输出已完整回答两问。凯尔希题也稳定保留同一 chunk 中相隔较远的：

```text
若 Mon3tr 攻击，尖刺会贯穿胸膛
仍是人类伟岸的壁垒
没人能剥夺生而为人的荣耀
```

乔迪题的模型曾把 `巨物` 扩成 `巨物（海嗣）`。仅加强 prompt 仍偶尔不服从；加入确定性
后校验后，真实输出恢复为 `巨物将触肢贴在舱壁上`，不再添加无逐句绑定的类别别名。

### 15.3 最终端到端结果

最终完整十题目录（别名后校验前；其他核心链路与已提交版本相同）：

```text
/mnt/.../runs/e2e_novel_questions_20260825/full10_subclaim_crag_final_20260825/
```

结果：

```text
平均 11.668s
中位 11.012s
最大 18.233s
超 20 秒 0/10
8 个可答题全部作答
反事实题给出证据边界内的不确定回答
虚构越界题正确 abstain
```

最后三项单题回归：

```text
血骑士完整两问: 14.213s
凯尔希完整两问: 12.249s
乔迪别名后校验: 9.560s
```

目标回归测试：

```text
21 passed
git diff --check passed
```

### 15.4 仍需继续处理的问题

1. 当前子问题 signal family 是轻量确定性路由，验证了机制价值，但还不是论文级开放域
   subclaim parser/coverage predictor。应把 `question_subclaims × evidence` 覆盖矩阵显式建模。
2. exact quote 仍只能证明引文存在；人物绑定、事件绑定、因果蕴含、指代展开需要独立的
   claim-quote entailment 校验，不能回退到旧 token overlap。
3. 第 4 题“没有彻底毁掉车尔尼”的参考口径需要重新人工定义：当前 evidence 同时支持
   “折服于音乐所以手段变温和”与“计划早已失败、只能赌一把”。应在评测集写清所问事件
   和允许答案，不要用模糊 reference 判断链路好坏。
4. 本次固定十题是定向回归，不足以宣称总体弃答/幻觉率。下一步应冻结 100--200 道仓库外、
   含复合题/反事实/指代/不可答题的 blind E2E 集，并报告 subclaim coverage、claim precision、
   selective risk、p50/p95 latency。
5. `full10_subclaim_crag_final_20260825` 后新增的括号别名后校验只单题验证过；扩大评测时需
   检查是否错误删除原文明确给出的合法同位语。

### 15.5 论文主线进一步收敛

最强全链路方向不是继续单独刷 BGE 指标，而是：

```text
Dual-view Evidence Routing
  ranking representation: 允许跨段 chain，优化相关性/排序
  answering evidence: 只允许 direct chunk，优化可引用性/可蕴含性

+ Set/Subclaim Sufficiency Matrix
  question subclaim -> supporting direct evidence
  covered / partial / missing / conflicting

+ Targeted Retrieval and Selective Generation
  answer / partial_answer / retrieve_missing_slot / abstain

+ Latency-aware Early Exit
  已覆盖时单轮结束，仅对明确缺失槽位定向二检索
```

本轮提供了直接的实证动机：chain 对 hard59 排序有益，但进回答上下文有害；正确证据已召回
仍会被 post-rerank selection 和 sentence refinement 再次丢失；修复职责分离与集合覆盖后，
十题从旧的高弃答/高延迟状态改善到平均约 11.7 秒且 0/10 超时。可将这个现象形式化为
`ranking utility != answering sufficiency`，围绕 coverage-risk-latency 三目标联合学习展开论文。
