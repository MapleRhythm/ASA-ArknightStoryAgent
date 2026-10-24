# ASA 编号引用审查、真实后半链路回放与去重候选否决 v21

日期：2026-09-09。241 是唯一规范代码/数据源，248 是训练镜像，本地仅作传输缓存。

## 结论

本检查点没有新的可认证整体回答正确率，也没有将实验候选合入运行时。v20 的规范源码和 362 项规范回归保持不变。

1. 编号引用 8B 的 58 状态生成与盲审全部完成。它没有优于原 4B 对照的可靠证据，不晋级。
2. 同一开发题的真实稀疏检索后半流程，确实会在提示排序/去重阶段丢失关键证据，但此前不能把候选池排名直接当成最终模型可见性。
3. 四份唯一提示的真实 4B 生成全部完成：唯一被放行的答案混淆了事件；另外三条被拒答。没有新增合格正确展示答案。
4. “先按分数排序，再近似去重”虽修复了低分抢占高分的不一致，却在真实回放中丢失了原本可见的关键段。隔离测试 384 项全通过仍不足以放行，已否决该候选。
5. 仅完全相同去重、仅字面包含去重也未恢复目标段的模型可见性：在 seed-retained 条件下仍落在 top10 之外。没有通过提高 k、放宽引文门槛或删失败记录来制造提升。

## 数据与版本边界

- 仓库：`${REMOTE_HOME}/ASA-ArknightStoryAgent`，HEAD `0d594a3c9fd641fd8d7f798c8ca6b07db6c7020b`。
- 实验目录：`${REMOTE_ASA_ROOT}/rank_v8_coverage/experiments/canonical_quote_eval_20260908.irPugk/fact_coverage_v1.leU0LZ`。
- 规范源码冻结沿用 `checkpoint_v20_state.json` 的 `runtime_source_sha256`。
- fixed79 是 58 个不同问题的 79 个历史状态，不是 79 道独立全链路题。历史 39 direct / 31 retrieve / 9 abstain 不变，不等于正确率。
- 本文覆盖的 58 题是开发检查，不是未见过的 holdout。单题的多种条件也不能当作多道独立测试题。
- 本轮没有改默认模型、默认提示、重排窗口、引用长度门槛；没有训练、提交/push、重启线上服务或覆盖旧标签。没有证实线上进程加载了工作区修复。

## 一、编号引用 8B：58 状态全部完成

生成后验：`quote_pointer8b58_generation_postflight_v1.json`。
机械诊断：`quote_pointer8b58_failure_diagnosis_v1.json`。

- 58/58 严格 JSON，58/58 stop；实际输入没有截断，输出预算没有缩减。
- 模型原始 58 条均选择 `answer_directly`。
- 24 条在绑定阶段失败，全部涉及跨 E 的 pointer；不是 JSON 失败。
- 其余 34 条：14 answer_directly / 13 retrieve_more / 7 abstain。
- 535 个引用中 75 次跨 E 错配、460 次合法 pointer，无未知 QID。另有一个空引用 fact，被更早的绑定异常遮住，独立诊断保留。
- 20 条非回答状态中，18 条触发总引文超过 400 的警告；警告可能重叠，不能累加为互斥状态数。
- 没有修错 E、删除坏 fact、放宽 80/160/400 门槛或手改模型输出。

### 固定协议盲审

`quote_pointer8b58_blind_audits_v1.json` 与对应 postflight 已完成。
三组共 174 状态，110 个去重请求；82 个已有对照结果连同错误原样复用，新增 28 次请求，24 成功、4 失败。协议仍为 coverage-v3/content-v2，glm-5.3，max_tokens=16384，并发 2，无失败重试。

| 语义盲审状态（每组分母均为 58） | 原 4B | native 8B | pointer 8B |
|---|---:|---:|---:|
| supported_complete 暂判 | 12 | 3 | 5 |
| supported_partial | 8 | 1 | 1 |
| unsupported | 9 | 4 | 4 |
| audit_error | 3 | 1 | 4 |
| no_answer_action | 25 | 49 | 20 |
| pipeline_error | 1 | 0 | 24 |

这些是自动审查暂判，不是人工 gold 正确率。pointer 的 coverage 为 13 complete / 1 partial，也不能代替语义正确性。

新增失败全部在语义审查：威灵顿旗帜为支持片段契约失败；玛露西尔与老鲤为 length；巫女林将摘句中的“与”改为“和”，违反逐字要求。原始失败完整保留，不能据此认定模型路由变更。

### 源文独立复核的范围

`quote_pointer8b58_selected_source_review_v1.json` 记录 9 项选定复核，不是全 58 源文认证，也不是人工 gold。它没有发送给生成器或裁判，没有改旧标签。

已发现的风险包括：把个人流亡中的盗窃泛化为整个组织的共同特征；把同事的苦难陈述误绑定为莫斯提马真实身份；把血衣推理疑问等同于质疑定罪权力；把提高寻人效率的生火吃饭行为自动说成“不着急”的原因；把麟青砚的引文错绑给解真。角色、时间、因果和作用范围仍是系统性问题。

## 二、真实后半链路：排名不等于证据可见

源：`susie_seed_retention_v1.json`、`susie_prompt_handoff_v1.json`、`susie_prompt_dedup_trace_v1.json`。

问题：“夏栎最终是如何救出苏茜的？”
诊断段：`activities/act10mini/level_act10mini_st05#chunk-0025`。

seed-retained 实验将两条去重链扩展为 17 条，复用两条字节完全相同的旧完整输入分数，新算 15 条完整 3072 上限的 CPU 分数，没有输入截断。该构建器未合入规范源码。

实际调用规范 `rerank_and_rescue_hits`、`prepare_prompt_evidence` 和 `build_conclusion_prompt`，输入仍为冻结的稀疏 120 池与已实测 logits。此处没有新跑 dense/fusion、前端 scope/hypothesis 或在线服务。

固定规范设置：final32、prompt top10、完整块 12000 字符预算；MMR/pinning/CRAG/pyramid/web 均关闭。

| 条件 | 基础 tokenizer 诊断 tokens | 关键救援段可见 |
|---|---:|---|
| 原构建器，旧 prefix1024 分数 | 5172 | 是，E10 |
| 上一行加 order-only reserve6 | 5172 | 是，提示完全相同 |
| 原构建器，complete3072 | 7005 | 否 |
| 上一行加 order-only reserve6 | 7005 | 否，提示完全相同 |
| seed-retained，complete3072 | 7963 | 否 |
| 上一行加 order-only reserve6 | 7967 | 是，E3 |

所有条件的目标段都进了 final32，所有提示都能容纳配置的输出预算。可见性差异来自下一阶段的排序/去重/选取，不是这里的 token 裁切。

原 prefix 条件在本次后半流程中能看到目标段。因此不能把此前第 17 名的候选池诊断说成“线上最终必然召回不到”；旧冻结 E1–E10 与本次稀疏后半流程是不同输入状态。

软人物保留实验 `explicit_speaker_reserve_v1.json` 的词典来自规范 54,252 文档的 speaker 字段，共 6521 标签。它只匹配原问题、不猜别名。但“感染者”“孩子”等也是 speaker 标签，不能把这套词典当成可靠人名识别或硬实体约束。该实验也未晋级。

## 三、四次真实 4B 生成

源：`susie_handoff4_ctx12288_v1/`、`susie_handoff4_validation_v1.json`、`susie_handoff4_postflight_v1.json`。

只运行上述四份唯一提示，完全重复的两份未重跑。同一 4B、实际 LoRA、原问题 seed；temperature=0.1，top_p=0.8，repeat=1.05，输出 2048/context 12288。连续至少 60 秒 GPU 空闲后启动。实际 seed、LoRA、tokenizer IDs、模型前后 SHA、完整输入及未缩输出预算均核验。

| 唯一提示 | 原始回答/校验现象 | 最终动作 |
|---|---|---|
| 原 prefix | 找到藤蔓、救援和安抚，但 refs 写成 `[E7]`/`[E10]`，另有一段引文超过 80 | abstain |
| 原 complete | 混合藤蔓事件与米克尔伪装救逃兵 | answer_directly，但错目标/混事件 |
| seed complete | 回答救逃兵，藤蔓引文又错绑 E1 | abstain |
| seed complete + reserve6 | 找到正确事件，但 E3 的救援引文全部绑到 E2 | abstain |

四次均 stop，并通过严格 JSON 规范化；没有新的合格正确展示答案。未修编号、删 fact 或人工改输出。

## 四、候选修复被真实回放否决

候选文件：`prompt_priority_candidate.py`；验证：`prompt_priority_candidate_verified_v1.json`。

原去重是按输入先后贪心选择代表，再按 prompt 分数排序。实际存在 1.875159 的旧链提前排除 2.644684 的新链（Jaccard 0.827759），且新链独有关键救援句。因此尝试先按现有分数稳定排序，再使用原去重。

- 新增 22 项定向测试，原代码实测 12 失败 / 10 通过。
- 最新 v20 隔离源码上，原 362 + 新 22 = 384 项全通过。
- 先逐条重现全部六份旧提示、选择和 packing，再比较候选；没有重算 logits 或调用模型。
- 真实回放中，原构建器两种分数下的提示文本不变（部分重复链代表 ID 改变）。
- seed-retained 两种顺序都变为 7964 tokens，关键段均不可见；原 reserve6 条件原本可见，现在反而丢失。

故该候选被否决：384 项通过是隔离测试结果，不是“规范代码已通过 384 项”或“答案已提升”。没有执行 apply，没有镜像到 248。

`prompt_dedup_policies_v1.json` 进一步对全部六种状态比较三种固定策略：score-priority near082 / exact / containment，共 18 个 CPU 回放状态。对 seed-retained：

- near082 去重后 10 项，但仍丢关键段。
- exact/containment 去重后均 21 项，目标仍未入 top10；提示 7184 tokens，仍不可见。

这些策略也未晋级。仅改近似阈值或排序并不足以解决错误事件高分、重叠长链挤占和独有事实保留问题。

## 后续

下一步应分别处理：可追溯的链成员去重/完整块保留；以及减少生成输出中可避免的冗余引用绑定，同时保留严格事实校验。任何新协议须新生成、新保存，并用反例核验；不得拿旧错误输出删字段后当新模型成功。

原编号引用的 QID 已全局唯一，却还要求模型同时生成对应 EID，存在可消除的冗余。可研究“模型只输出全局 QID，由冻结目录唯一还原 EID/原文”的新协议。它不能自动修复错事实、错事件或引用不足，也不能把旧 24 条错误直接转成成功；本检查点尚未验证该新协议。

目标继续 active。没有宣称完成全 RAG 召回评估或新完整多轮 E2E 正确率。
