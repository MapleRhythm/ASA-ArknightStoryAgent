# 任务：构建高质量 claim-citation hard-negative 数据集（证据绑定）

你是 gpt-5.6-sol 子 agent。目标：修好 hard-negative 选取,使其真正"相关但不支持",并用 GLM 验证质量。

## 背景与已知问题
证据绑定 = 每条主张(fact)↔它引用的证据编号(E-ID)。当前 claim-citation alignment 仅 0.209,是主瓶颈。
我已用 char-trigram 相似度挑 hard-negative,但**失败**：抽象的 fact 与原文对白几乎零词汇重叠(mean sim 0.008,max 0.337),所以选出的负例不"硬"(不是混淆项,近随机)。
根因：问题/主张用抽象词,证据用剧情内具体叙述,词汇错位。

## 数据
- 源 SFT：`${REMOTE_STORE}/exx_grounding_v1/data/exx_grounding_v3_sft_minimal559_glm53_20260903/train.json`
  - 每条 grounded_action_generation(answer_directly) 的 prompt 有 `evidence:` + [E1..En] 段,输出 supported_facts 有 fact + evidence_ids(gold 绑定)。
- 我已生成的初版(trigram,质量差,可覆盖)：`${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v1_20260904.jsonl`
- 检索/嵌入器可复用：`${REMOTE_STORE}/exx_grounding_v1/tools/retrieval_reward.py` 里的 RewardEngine 用 Qwen3-Embedding-0.6B(reasoning env,VLLM_USE_FLASHINFER_SAMPLER=0,CUDA_VISIBLE_DEVICES=0)。也可以直接用 sentence-transformers 加载 `${REMOTE_STORE}/rank_v8_coverage_241_mirror/Qwen3-Embedding-0.6B` 做句向量。
- GLM：`source ~/.bigmodel_env`,endpoint `https://open.bigmodel.cn/api/coding/paas/v4/chat/completions`,model glm-5.3。必须 `env -u https_proxy -u HTTPS_PROXY -u http_proxy -u HTTP_PROXY -u ALL_PROXY -u all_proxy`(代理端口没监听会 connection refused)。reasoning_effort 用 medium(够用且快),max_tokens 4096,timeout 120。

## 你要做的
对每条 (claim, gold_eids, 可见证据池)：
1. 用 **dense 句向量余弦相似度**(不是 trigram)对每个非 gold 的可见 chunk 打分(embed claim vs embed chunk)。
2. 额外把**与 gold chunk 同 source_name/同场景**的非 gold chunk 标为高混淆候选。
3. 取 top-4 hard negative(优先 dense 相似度高 + 同场景)。
4. **GLM 验证**：对每个候选 hard negative,让 GLM 判 (claim, chunk) 是否支持该主张 → 只保留 GLM 判"不支持"的,作为干净 hard negative(GLM 判"其实也支持"的要剔除,因为那不是负例,是漏标的正例——这本身也是有价值的发现,单独记录)。为控成本,GLM 验证可只跑前 ~300 条 record 的候选。
5. 输出 `${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v2_20260904.jsonl`,每条含 claim / positive_eids+texts / hard_negatives(eid,text,cos_sim,glm_verdict) / 以及标记出的"疑似漏标正例"。

## 质量报告(写到 stdout 和一个 .report.json)
- dense hard-neg 的 cos 相似度分布(证明比 trigram 的 0.008 高得多)
- GLM 验证的 hard-neg 中真"不支持"的比例(越高越干净)
- 发现的"疑似漏标正例"数量(gold 之外其实也支持的 chunk)——这个数字很重要,它说明 exact_evidence_set 只有 29.85% 有多少是标注多解性造成的假阴性。

先跑一个 20 条的 smoke 验证管线通,再全量(或前 300 条 GLM 验证)。完成后把结论写到 `${ASA_ROOT}/.codex_tmp/analysis/HARDNEG_FINDINGS.md`。
