# 分析任务简报：ASA Exx grounding 模型的格式正确率、判断率、多轮召回增益

你是 gpt-5.6-sol，作为分析子 agent。目标：基于以下已收集的实证数据，深入分析三个问题，给出**机制层面**的因果判断和可执行建议。不要复述数据，要给出因果链和反事实推理。你可以 SSH 到 241/248 复核或补充证据。

## 服务器与环境
- 241（4090×3，检索/标注）：`ssh -F none ${REMOTE_USER}@${REMOTE_HOST}`，用 `env SSH_ASKPASS=${ASA_ROOT}/.codex_tmp/ssh_askpass_asa.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid -w ssh ${REMOTE_USER}@${REMOTE_HOST} '<cmd>'`
- 248（A100，训练/eval）：同样方式 `${REMOTE_USER}@${REMOTE_HOST}`
- 主仓库 241: `${REMOTE_HOME}/ASA-ArknightStoryAgent`；Exx 产物 248: `${REMOTE_STORE}/exx_grounding_v1`
- GLM key 在两台的 `~/.bigmodel_env`

## 系统背景
ASA = 明日方舟剧情多跳问答 RAG。生成器是 Qwen3.5-4B + LoRA，输出结构化决策 JSON（grounded_action_exx_v1 协议）：
`{"next_action": "answer_directly|retrieve_more|abstain", "supported_facts":[{"fact":"","evidence_ids":["E3"]}]}`
- answer_directly：每条原子事实绑定 1-2 个证据编号 E-ID；服务端从通过校验的 facts 重建答案，禁止 raw fallback。
- retrieve_more：给 follow_up_hypothesis 触发第二轮检索。
- abstain：证据不足固定弃答。
链路：query → dense+BM25+MiniRAG 混合召回 → RRF 融合 → cross-encoder rerank → 证据进 prompt → 4B 生成决策 → 最多 2-3 轮。

## 数据一：格式正确率（strict JSON / schema 合规 / 无效 E-ID）

val79（固定证据 prompt，79 题 grounded_action，A100 vLLM 推理）：

| 模型 | strict JSON | schema 合规 | 无效 E-ID | invalid action(总) |
|---|---:|---:|---:|---:|
| quote 时代 KTO 混合（在 Exx prompt 上评） | 68.4% | 6.3% | 0.0% | 31.6% |
| Exx SFT (success559, lr3e6, 1ep) | 87.3% | 84.8% | 2.5% | 12.7% |
| Exx SFT+KTO v7 | 86.1% | 81.0% | 2.5% | 13.9% |

- SFT 把格式从 ~68% 拉到 87%，几乎全部提升来自格式（schema 6.3%→84.8%），不是判断变准。
- 旧 quote 模型 31.6% 无效是分布外产物（它训练学的是复述 quote 格式，被要求输出 E-ID 格式）。
- KTO 在 SFT 之上没有再提升格式，反而略降。

## 数据二：判断率（action 准确率）

- Exx SFT 账面 action 准确率 73.4%；剔除无效 JSON 后条件准确率 84.1%（三个模型都卡在 83-84%）。
- 按 gold action 分层（train783 rollout，n=783）：
  - answer_directly：555 题，答对 458，**82.5%**
  - abstain：99 题，答对 78，**78.8%**
  - retrieve_more：129 题，答对 37，**28.7%** ← 唯一重灾区
  - 混淆矩阵关键项：retrieve_more→answer_directly 64；answer_directly→invalid 39。

## 数据三：retrieve_more 失败的标签审计（GLM-5.3 盲审 50 例）

对 retrieve_more→answer_directly 的 50 个分歧案例，GLM 只看 问题+证据+模型回答（不看教师标签）判定：
- supported（模型对、教师保守）：**1 例 (2%)**
- partially_supported（有支持但缺关键面，抢答）：17 例 (34%)
- unsupported（模型实错）：**32 例 (64%)**
  - 引用幻觉（编造/错引 E-ID）14 例
  - 误读证据 13 例
  - 其他 5 例
- 结论：26.8% 失败率基本真实，不是标签噪声；错误集中在"证据不足时抢答 + 引用幻觉 + 误读"。

## 数据四：证据充分性探针（主实验线，theme E）

200 道叙事多跳题，真实召回 top-120，用 E-段级子序列覆盖判据自动标注 prompt(top-12) 内 gold 证据链完整性：
- full（证据链完整）13.5% / partial 49.5% / none 37.0% → **86.5% 的 prompt 证据链不完整**
- 用候选分数的池化特征（fusion max/mean/std/gap/高分计数）预测"证据不充分"的 AUROC 全部在 **0.45-0.55（随机水平）**。
- 含义：证据充分性是候选集合的性质，单候选分数看不见"缺哪一环"；4B 模型在 prompt 里同样判断不好——这正是 retrieve_more 判断率只有 28.7% 的上游原因。

## 数据五：多轮召回增益（关键，本次新查）

真实多轮检索 recall（v2 LoRA 模型，sample50，planner=conclusion，max_rounds=3）：

累积 recall（cumulative_recall_by_round）:
| cutoff | round1 | round2 | round3 |
|---|---:|---:|---:|
| @5 | 0.62 | 0.70 | 0.72 |
| @10 | 0.68 | 0.74 | 0.76 |
| @50 | 0.74 | 0.78 | 0.80 |

每轮"新鲜"recall（round_recall，仅该轮检索结果）:
| cutoff | round1 | round2 | round3 |
|---|---:|---:|---:|
| @10 | 0.68 | 0.68 | **0.52** |

首次命中轮次分布（40 个命中里）：round1=37，round2=2，round3=1；未命中 10。
rounds_run 分布：跑满 3 轮 35 题，2 轮 11 题，1 轮 4 题；stopped_reason=answer_directly 15。

关键事实：
- 累积增益：round2 比 round1 只 +6pt(@10)，round3 再 +2pt。
- 每轮新鲜 recall 不升反降（round3@10 从 0.68 掉到 0.52）——后续轮的 follow-up hypothesis 检索质量在退化。
- 命中几乎全在 round1（37/40）。gold 要么第一轮就召回，要么后面也召不到。

## 你要回答的三个问题

1. **是什么影响了格式正确率？** 为什么 SFT 能把 schema 从 6.3% 拉到 84.8%，但还剩 12.7% 无效？剩余无效的成因是什么（长度截断？特定 query 类型？证据条数？decode 长度？）——去 248 抽查 val79 里 invalid 的那些预测，找共性。
2. **是什么影响了判断率？** 为什么 answer/abstain 都 ~80% 但 retrieve_more 只有 28.7%？结合数据四，判断这是"信息不可得"（充分性对模型不可见）还是"能力不足"（模型看得见但学不会）。设计一个能区分这两者的判别实验。
3. **为什么多轮召回增益不大？** 从数据五的三组数字（累积增益小、新鲜 recall 退化、命中集中在 round1）推断根因：是 follow-up hypothesis 多样性不足（重复检索同一批文档）、还是 hypothesis 质量退化、还是 gold 本就不在库里？去 241 抽 multiround records 里 round2/3 的检索结果与 round1 的重叠度、以及 follow-up hypothesis 文本，验证你的假设。

## 输出
写到 `${ASA_ROOT}/.codex_tmp/analysis/CODEX_FINDINGS.md`，包含：每个问题的根因判断（带你补查的证据）、反事实（如果改 X 会怎样）、以及按 ROI 排序的 3-5 条可执行改动。
