# 任务：4B 本地验证器 vs GLM 一致性校验（决定本地验证器能否当 RLVR 奖励）

你是 gpt-5.6-sol 子 agent。目标：量化"4B 本地 claim-citation 验证器"与"GLM 裁判"在同一批 (claim, chunk) pair 上的判决一致性。若一致率高，本地验证器可当 RLVR 奖励（快、离线、确定）；若某类系统性不一致，指出是哪类，供后续补标。

## 背景（务必理解，避免测出假高一致）
- 证据绑定 = 每条主张(fact)↔它引用的证据段(E-ID)。目标是判 "chunk 是否支持 claim"。
- 已有实测：4B **zero-shot** pairwise 判别 acc=0.886, prec=0.950, rec=0.815, spec=0.958（vs GLM 验证的硬负例）。但那批负例本身就是 GLM 标的，正例是"gold 假定正确"未经 GLM 独立判。所以那个 0.886 **不是纯净的一致率**——正例侧 GLM 没独立表过态。
- 因此本任务要**让 GLM 对正例也独立判一次**，才能得到无偏的 4B-vs-GLM 混淆矩阵。

## 数据
- v2 数据集：`${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v2_20260904.jsonl`
  - 每 fact-row 有：claim, positive_texts(gold 引用段), hard_negatives[{eid,text,glm_verdict}]（glm_verdict=unsupported 的是 GLM 已判负例）。
- 抽样：取 ~250 个 fact-row，每行取 1 个 positive chunk + 1 个 glm_verdict=unsupported 的 hard-neg → ~500 pair，正负均衡。

## 你要做的
### 1) 三方判决，同一批 pair
对每个 (claim, chunk) pair 收集三个标签：
- **gold 先验**：positive=支持，hard-neg=不支持（数据集自带，但正例未经 GLM 独立判，仅作参考）。
- **GLM 独立判**：对**每个 pair 都新调 GLM**（包括正例！），问"这段证据是否支持这条主张，只输出 支持/不支持 + 一句理由"。这是本任务的 ground truth。
  - glm-5.3, reasoning_effort=medium, max_tokens=4096, timeout 120。
  - 必须 `env -u https_proxy -u HTTPS_PROXY -u http_proxy -u HTTP_PROXY -u ALL_PROXY -u all_proxy`（代理没监听会 connection refused）。`source ~/.bigmodel_env` 拿 key。
  - 并发≤3，遇 429/1301 退避重试；1301(内容过滤) 记为 skip 不算错。
- **4B 本地判**：用 vLLM 跑 base 4B（zero-shot）判同样的 pair。
  - 复用已跑通的脚本模式：`${REMOTE_STORE}/exx_grounding_v1/tools/probe_4b_binding.py`（reasoning env, VLLM_USE_FLASHINFER_SAMPLER=0, CUDA_VISIBLE_DEVICES=0, enforce_eager, **必须有 `if __name__=="__main__"` guard 否则 vLLM spawn 会递归崩**）。
  - 也测 4B 输出一个可用作分数的信号（例如对"支持"token 的 logprob，或多次采样的支持比例），用于算 AUC 而不只是二值一致率。

### 2) 指标（写全）
- 4B vs GLM 的**一致率**（agreement rate）、Cohen's kappa。
- 4B vs GLM 的混淆矩阵（以 GLM 为 ground truth）：acc/prec/rec/spec。
- **AUC**：4B 支持分数 对 GLM 二值标签。
- **分层**：按 query_type 和按 正例/负例侧分别看一致率——重点找"4B 在哪一类系统性偏离 GLM"。
- **GLM vs gold 先验的一致率**：顺带量 GLM 判正例时有多少判成"不支持"（= gold 正例里的噪声/多解），以及判 hard-neg 时确认率（应≈之前的 88.9%）。

### 3) 判据与结论
- 若 4B-vs-GLM agreement ≥ ~85% 且 kappa 合理、AUC ≥ ~0.9 → 本地 4B 验证器**够格当 RLVR 奖励**（或微调后更好），无需 GLM 在环。
- 若某 query_type / 某侧系统性低 → 指出该类，建议对该类补 GLM 标注再微调验证器。
- 明确写出：本地验证器当奖励的**推荐形态**（zero-shot 4B / 微调 4B / 微调 reranker）与理由，以及**是否需要先微调**才能达标。

先跑 ~40 pair 的 smoke 打通三方管线，再全量 ~500。
产物：结论写 `${ASA_ROOT}/.codex_tmp/analysis/AGREEMENT_FINDINGS.md`；原始三方判决存 248 `${REMOTE_STORE}/exx_grounding_v1/eval/verifier_glm_agreement_20260904/`。
