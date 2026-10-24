# ASA 回答正确率：系统诊断、修复与失败实验

日期：2026-09-08。基于本地源码、冻结 strict9 输入、实际模型推理及逐题原文复核。

## 后续更新：按用户要求取消 quote 输入证据的固定截断

用户明确要求“不要截断”后，已修改本地回答提示的 minimal、full、direct
三条路径：`quote / strict / grounded` 保留整块正文（包括链式上下文），
不再应用每块1000字符的固定前缀截断。未采用 direct-only 展示，不改变
检索器、top-k、模型权重或模板选择；E-ID 和旧短摘要调用保持原行为。

总证据预算仍保留，当前 GPU quote 配置为12000字符。整块放不下时记录
`budget_skip` 并继续尝试后续块；块间换行计入预算，首块也不能超额。
这不是无限输入：被跳过的整块仍不可见；12000是证据字符预算，不是整条
模型上下文的 token 预算。引用编号按实际可见块连续分配，MiniRAG hints
不从被跳过的块生成，grounding 仍只对实际展示的正文核验。

同一冻结 strict9、同一选段器及12000总预算的离线对比：

| quote 输入展示 | 必要最小集全文可见题数 | 最小段全文可见数 | 平均证据字符 |
| --- | ---: | ---: | ---: |
| 旧：每块最多1000字符 | 5/9 | 9/13 | 5673.33 |
| 完整块 + 遇超预算即停止（离线反事实） | 8/9 | 12/13 | 9179.44 |
| **本地新实现：完整块 + 跳过超预算块** | **9/9** | **13/13** | **10275.00** |

新实现4题触发整块跳过，最大证据长度11993字符；E-ID 原基线仍为9/9题、
13/13段。审计在 runtime policy 下将未预打包的选段池直接交给真实提示
构建器，并逐题对照独立 packing trace，quote 和 E-ID 各9题均一致。
其他 packing policy 仅为离线反事实。

**这是必要原文的展示覆盖改善，不是9/9答案正确，更不是线上正确率。**
本次未重新生成答案、未重跑79条全量回归，未部署或提交。9题沿用开发集
Codex 复核的最小集，不是独立人审 gold。此前 direct-only 等36条生成实验
无稳定收益的结论保留，不能与这次完整链式正文改动混为一谈。

新归档：`data/eval/strict9_quote_complete_evidence_visibility_20260908.json`
（`asa_rendered_evidence_visibility_v2`），SHA256：
`7065880ef5d7ee97c378d1a92da67d3760fd3c1e22699ac6f454d9b6cb1c8fb7`。
旧报告 `data/eval/strict9_rendered_evidence_visibility_20260908.json` 原样保留。
报告绑定输入快照、复核标注、配置及相关源码 SHA，不用新 renderer 伪装复现
旧截断结果。

当前完整测试 **382 passed**，包括长链尾部保留、超预算首块与中间块、
全部不可装入、换行计数、引用编号重排、隐藏段不可借引、hints 无泄漏及
真实 runtime packing 的回归。以下“不修改默认 renderer”等表述记录的是
用户此次要求之前的阶段性决定，以本更新为准。

## 前序研究与阶段性结论

同日早期离线研究见 `ACCURACY_IMPROVEMENT_RESEARCH_20260908.md`：当时
E-ID 的复核最小集13/13段已真实可见；简单改去重反而退化。quote只改变
展示正文的候选有可见性收益，尚未运行新的生成或验证生产正确率。
本文后面的96条scope重判已完成，以“训练入口续审”一节为准。

上述renderer候选后续已在空闲4090完成36条基座quote生成；没有稳定的
最终正确率净收益，不推广。真实运行合同、逐题复核及校验改写影响见
`QUOTE_RENDER_ABLATION_FINDINGS_20260908.md`，不将基座smoke称为生产结果。

## 本轮结论

最主要的已知问题不是“模型不肯答”或单纯 top-k 不够，而是：

1. reader 在充分证据下仍错绑说话人、主体、关系方向和否定；
2. 多子问覆盖与是否继续检索的决策不可靠；
3. 格式、词法校验和单一裁判分数会掩盖这些错误；
4. 历史训练标签受截断污染，已复判的标签仍未完成动作级和问题族隔离；
5. 严格引用路径存在确定性绕过，本轮已修复。

没有得到新的生产端到端正确率，也没有验证新的 checkpoint 优于当前部署。
本轮仅将可复现的安全修复留在源码中；**不推广换行/主体提示，不修改生产
检索配置、checkpoint、默认 renderer 或生成提示，不启动带泄漏风险的重训。**

## 已修复的工程问题

| 问题 | 后果 | 修复与边界 |
| --- | --- | --- |
| `parsed_evidence or full_evidence` | 显式空文本、空 E-ID 块或解析失败时，引用可回退到模型没看到的完整文档 | quote 与 evidence-ID 区分 `None` 和空字符串；只有未提供 prompt 的旧离线调用允许 fallback |
| quote 用 `eid_map and unknown_id` 检查编号 | 空编号表时不报告不存在的 E-ID | 即使映射为空也必须拒绝未知编号 |
| generation 用 `evidence_prompt_text or prompt` | 空证据被替换为整个任务/规则提示 | 普通生成和截断恢复路径均保留显式空值 |
| quote/strict/grounded 先返回无引用专用答案 | `abstain` 或原生成答案可被旧角色/事件模板覆盖，绕过引用核验 | 与 evidence-ID 一样，不在结构化 grounding 模式调用无引用模板；legacy weak 路径保留 |
| quote 拒绝后将候选原文拼成“片段事实” | 即使动作是弃答，用户仍会看到未经逐条验证的伪答案 | 预算耗尽时返回中性弃答，保留拒绝原因及 grounding diagnostics |
| `sample_grounded_conclusions` 异常分支引用未传入 `retrieval_trace` | 一条样本失败可能触发 NameError，后续自一致性采样与故障记录中断 | 显式传入可选 trace，覆盖有/无 trace、多样本续跑、单样本抛错、截断恢复 |

新增引用边界回归在旧实现上为 **9 failed / 5 passed**；修复后全部通过。
完整测试结果见本节末尾。安全修复可能增加以前被旁路掩盖的弃答；这是阻止
不合格答案被放行，不等于已经改善完整回答率。后续真实 quote E2E 必须同时
检查正确回答保留率、无依据回答与过度弃答。

仍未解决：词法核验不是语义蕴含。运行最小反例时，原文
“阿米娅没有同意申请。”与相反主张“阿米娅同意了申请。”在 quote 和
evidence-ID 检查中均没有 hard issue。quote 还可能只对观点错归或寻找方向
倒置给 warning。不能把这些漏洞解释成通过了语义验证，也不应临时堆叠
人物专用规则来声称已解决。

## 固定证据的 108 条真实对照

同一 9 题、两种证据状态（minimal / 加真实同场景干扰），每题三个版本：

- `flat`：冻结原提示，字节不变；
- `dialogue_lines`：仅恢复 snapshot 原文换行；
- `dialogue_lines_attribution`：再增加通用说话人、对象、否定、子问覆盖约束。

两个 reader：Qwen3.5-4B base 与历史 clean-SFT，共 108 条。
原文逐 E-ID 核对；所有非空白字符、E-ID 和顺序保持不变。没有增加说话人、
标准答案或 reviewer notes。guidance 由已知失败类型启发，**这是开发调优，
不是新盲测**。数据只作评测，不导出为训练样本。

每个 reader 均 54/54 完成，0 输出截断。36 条 flat 的 system/user 提示和
raw output 与 2026-09-07 冻结结果完全一致（每模型 18/18）。

### 统一复核下的结果

主指标是：必要要点完整、没有无依据实质断言、协议合格、未截断。
正确的额外背景不直接判错；重复及格式另外检查。所有题都留在分母中。

| Reader / 证据 | 原提示 | 仅换行 | 换行 + 主体约束 |
| --- | ---: | ---: | ---: |
| base / minimal | 3/9 | 2/9 | 3/9 |
| base / +干扰 | 1/9 | 1/9 | 2/9 |
| clean-SFT / minimal | 3/9 | 2/9 | 2/9 |
| clean-SFT / +干扰 | 1/9 | 2/9 | 3/9 |
| base / 两状态合计 | 4/18 | 3/18 | 5/18 |
| clean-SFT / 两状态合计 | 4/18 | 4/18 | 5/18 |

18 行仍是 **9 个问题族的重复条件**，不是 18 个独立问题。两种模型都仅有
一行净增益，且 minimal 存在退化；不能声称稳定泛化改善。base 的
minimal+guidance 有 4 条语义 complete，但其中一条重复事实导致 schema
失败，故主指标为 3/9。

本次是 Codex 对全部 108 条输出及原文的复核，不是外部独立人审，也不是
新的 GLM 自动 judge 分数。一条 clean-SFT 换行干扰输出的多余引用有争议，
标为 `uncertain`，不计成功；其重复事实本身也未通过 schema。

### 为什么与旧报告的基线数不同

旧报告用 GLM set audit 加少量 Codex overrides；本轮重新逐题审核，并
同时报告语义与协议。这些评分差异不是模型变化：

- base / 带剑女子 / minimal：旧判 complete，但它额外断言镇中力量
  导致女子与剑客分离。原文明确说不记得分离时间，没有确认分离原因。
  本轮不接受这个新增因果，因此从成功中移除。
- clean-SFT / 带剑女子 / minimal：只有自称渡魂剑、失散与寻找，没有
  解释器物化伥及原佩剑/主人的关系，本轮判 partial。
- base / 杰米 / 干扰：疾病、死后危险、污名和制度阻碍均已覆盖；额外
  “就算有存续空间也与你何干”确实出自 E1，不能仅因多一句原文话语
  把其完整原因回答清零，本轮判 complete。

旧报告和原始审计不覆盖删除。新逐题结论单独绑定 matrix/prediction SHA；
这些分歧进一步说明正式验收需要独立 reviewer，而不能将任何一次模型
复核直接升级为人类 gold。

### 具体发生了什么

- 芙蕾达：两种 reader 的三个版本都没有稳定区分汤姆的指责与威尔的
  辩护；clean-SFT 甚至生成“芙蕾达认为威尔是叛徒”。换行不是根治。
- 莫斯提马：base 将同事的“苦难陈述者”职务归给她；clean-SFT 的
  干扰+guidance 反而把甘比诺的“萨卡兹”猜测当成真实身份。
- 老天师：base 把织物害怕她改为她害怕织物，guidance 后还把绩的
  吟唱归给她；clean-SFT 只重复 E1，漏掉问题要求的 E2 态度对照。
- 杰克逊：base / minimal+guidance 从 `retrieve_more` 转为回答，但
  把梅兰德基金会条件句换成莱茵生命不可信，不能把 action 提升算成正确。
- 女妖：base / 干扰+guidance 正确答娜斯提，是局部有效样例；但 minimal
  和 clean-SFT 全部版本仍未答，改善不稳定。
- 雷吉斯：clean-SFT 换行后排除了干扰遗嘱，却在 minimal 漏掉团聚目的。
  同一改动既救题又伤题。

### 格式、成本与动作

| Reader / 证据 | 指标 | 原提示 | 仅换行 | 换行 + 主体约束 |
| --- | --- | ---: | ---: | ---: |
| base / minimal | schema valid / over-retrieve | 7 / 2 | 6 / 2 | 7 / 1 |
| base / +干扰 | schema valid / over-retrieve | 7 / 2 | 8 / 1 | 9 / 0 |
| clean-SFT / minimal | schema valid / over-retrieve | 8 / 2 | 9 / 2 | 9 / 2 |
| clean-SFT / +干扰 | schema valid / over-retrieve | 7 / 2 | 6 / 1 | 8 / 2 |

每格分母 9。base / 干扰+guidance 已 9/9 愿意回答、9/9 schema 合格，
但仅 2/9 通过完整受支持回答标准，是典型“流程成功不等于答案成功”。

平均输入 token：minimal 约 778 → 780 → 897；干扰 1008 → 1011 → 1128。
guidance 比仅换行增加约 117 input token。base 各格平均输出 202–232 token，
clean-SFT 93–111。两组完整运行计时分别为 462.71s / 404.48s，runner 的该
计时包括末尾文件写入和模型资产哈希，不含模型加载；不是纯服务时延，
也没有逐 variant latency，不能据此声称哪种提示更快。

## 整个系统的下一步优先级

| 层 | 已确认 | 本轮未证明 / 下一步 |
| --- | --- | --- |
| 评测可信度 | 旧文档把 Codex reviewer 写成人审；同一输出存在裁判分歧 | 冻结独立问题族与必要要点；对 complete、分歧和随机失败做人类/独立复核 |
| 训练标签 | 326 gold 与 480 candidate full-evidence 复判已完成；旧截断 verdict 不可直接训练；旧 some 混淆了支持范围与证据条数 | 先用 scope v2 盲重判 96 个 some，仅对真 partial 再原子化，补每个证据状态的 set-level 动作；不把 GLM 裁决当人工 gold |
| split | 当前 817 pairs 按 record 分，113 train / 10 eval；本轮确切原问题字符串交集为 0 | 没有证据证明这份内部切分有精确重复泄漏；但尚未完成改写/同事件/跨轮 family 与外部测试隔离，不能称盲测 |
| reader | 充分正文下仍主体、否定、关系错绑，通用提示只有一行净增益 | 用训练分区内的新原文构造关系/主体/否定配对 SFT，并保留 Full、Distractor、Missing 状态 |
| controller | 充分时 retrieve_more；鼓励回答会产生新增幻觉 | 单独监督可见证据支持的要点与缺口；不把检索到长文等同于已充分 |
| retrieval / packing | 历史多轮候选高度重复，集合覆盖有缺口；但不能解释所有 fixed-evidence 错误 | 固定 reader 测有效新证据、必要要点覆盖与最终回答传导，不盲增 top-k/轮数 |
| grounding / serving | 空证据回退、无引用旁路与故障续跑缺陷已修复 | 仍需校准语义 verifier；按实际 quote 默认协议做受控 E2E，区分拒答安全与正确率 |

不用 strict9 原题直接训练再用其验收。9B 参考模型未运行，不将资产缺失
写成已比较。下一轮训练优先于继续在这 9 题上叠加阅读指令，但必须先过
支持范围重判、必要的原子化、动作标签和 family 隔离门禁。

## 训练入口续审：支持范围、真实文件与内容绑定

以下是同日后续工程审计；不改变上面的 fixed-evidence 模型结果。

### 旧 `some` 不能直接解释成 96 条真正的部分支持

旧完整证据复判的 326 条标签为 union183 / some96 / ambiguous23 /
unsupported24。逐条阅读 96 个 some 的理由后，发现不少记录明确写着
“单段直接支持整条主张，无需多段合并”，却仍标 some。这把“整条是否
成立”与“需要几段证据”混为一谈。此前“全部 96 条必须先原子化”的结论
过强，应改为先重判支持范围，之后只对真正的 partial 拆解。

新工具 `rejudge_gold_claim_scope.py` 使用独立的 `asa_claim_support_scope_v2`
协议：`fully_supported / partially_supported / unsupported / ambiguous`。
只输入完整原主张与引用正文，不输入旧 label/reason 或标准答案；
partial 必须提供原 claim 逐字未支持片段。单段支持整条也是 full；
映射为 legacy `supported_by_union` 仅为兼容，不表示必须联合多段。

例如真理本名、玛恩纳对现代“贵族”的态度等旧 some 确实有直接完整支持；
但“打散伥气”被扩写为“击杀”、改革被断言为“直接导致”关系破裂等仍有
未支持成分。不能靠保留一个 E-ID 或简单删除几个字推定修复成功。

这 96 条复判使用 append-only journal 和完整 input_key；finish_reason
必须 stop，超时与 length 均不计成功。现已 **96/96 完成，pending=0**：
GLM 判 **fully_supported72 / partially_supported24**，没有新 unsupported
或 ambiguous。72个 full 中，70个只需单段、2个需要联合支持。预算4096、
8192、16384分别完成20、73、3条；最后一次只重试3个失败项，复用93个成功
判定。接口返回的 response_model 均为 glm-5.3，这只是服务端报告的名称。

**72 是 teacher 标签，不是72条已确证 gold。**重点复核最后的
`26dda657612e28a0d2a41212` 时，Codex发现“阿尔图罗因……勾结巫王残党而
被拉特兰通缉”缺少因果证明：E1说明她已被通缉，E4说明执行者此时发现她
与残党合作，不能将两项事实自动视作通缉理由。原始 GLM full 标签保留，
另用内容绑定的 quarantine 保存分歧。该源记录也因缺少故事来源未通过
分区，所以未进入此次候选；仍在 pair/SFT 入口增加显式否决，防止将来
来源恢复后通过旧正例标签绕开隔离。这是抽取式重点复核，不是96条的独立
人类语义审核。

`merge_gold_claim_scope.py` 在96条全部匹配原记录和 judge contract 后
创建新的326条兼容视图，原归档不覆盖。兼容标签为 union255 / some24 /
ambiguous23 / unsupported24，其中1条 union 带 `training_eligible=false`
的复核隔离标记。任何规模都不意味着可以跳过动作级和分区审核。

归档：

- `data/eval/gold_claim_scope_20260908.jsonl`
- `data/eval/gold_claim_scope_20260908.report.json`
- `data/eval/gold_claim_scope_codex_quarantine_20260908.json`
- `data/eval/gold_recalibrated_scope_reviewed_view_20260908.jsonl`
- `data/eval/gold_recalibrated_scope_reviewed_view_20260908.report.json`

scope SHA256：
`aa0f600099c79f26f71e5dcd3451b4a44971727a4c61b80eb8f5aa0a372ed829`。
带复核隔离的兼容视图 SHA256：
`63fccf883aef221a6614a8b28ac529a0d417b4a5239ba3fb4b77c716129065c5`。

### 真实 clean-SFT 的分区检查

从隔离归档取回实际训练文件，并核对历史 build report 的 SHA：

| 文件 | 行数 | SHA256 |
| --- | ---: | --- |
| clean train | 910 | `3d189c64c4183c375cd2de5ef3c4ed1bd1fa153e565b924b4e92bee2eaffe103` |
| clean val | 95 | `0b6a34530f447c48c203ee62247ebe28a41b0aa063cbf86f2934ee8d8cf9fc35` |

`audit_training_lineage.py` 将精确问题、来源 ID、跨轮来源和 question key
构建传递闭包；近似问句用 0.88 阈值作保守筛查。活动按故事活动、主线按
章节汇总训练 prompt 的正文曝光，并将来源缺失项隔离。

与 val95、development27（strict9 的三状态重复）比较：

- train 405 条存在 held-out 故事范围曝光重叠；
- 236 条缺少故事来源，保守隔离；
- 剩余 269 条是分区候选，属于 175 个来源/精确问题 component，
  **不是 269 条已经审定的训练样本**；
- 没有发现 train 与 val/development 的精确问题或 lineage component
  相交。报告的 9 个 cross-split components 全是 val 与 development
  之间；strict9 本就取自 val，不能把这个数字说成训练题泄漏。

故事曝光风险不等于证明答案泄漏；剧情语料仅存在于检索索引也不算监督泄漏。
该工具尚未独立标注语义改写的问题族，更不能让已经训练过的 checkpoint
“追溯性变成 held-out”。新训练仍需保留外部独立事件/问题族验收。

### 重建候选与训练出口的确定性修复

`stage_full_evidence_sft_candidates.py` 用真实 clean train、全量 fact manifest
和内容匹配的裁决恢复可支持事实。没有 scope v2 的冻结基线生成 **174**
条候选；269 分区候选中，179 个源记录有共 553 个 fact，另有50个
grounded记录无 fact manifest，40条是其他任务。179 个源记录中5条没有
任何可保留的完整支持事实，因此不生成答案，也不擅自改为 retrieve/abstain。

553 facts 的分类为 clean454 / union30 / legacy partial31 / some20 /
ambiguous6 / unsupported12。这里 legacy `partial` 表示引用集合中仅有
部分 E-ID 支持完整 claim，与 scope 的“主张本身只有部分支持”不是一回事。
保留完整事实并不证明整份回答完整：所有重建候选仍须 fresh set audit，
没有 registry，`training_ready=false`。

接入带复核隔离的 scope 兼容视图后，同一分区的20个旧some变成13个full、
7个partial，候选从 **174增至176**（142个来源/精确问题component）。新增
2条候选、修订6条原候选，恢复13个事实；另3个有fact的源记录仍无可保留
事实。新增两题是阿米娅同意娜塔莉娅申请的原因、学生被集中至凛冬学校的
经过。不能将“多了2条答案”解释成正确率提高。

after-scope 文件独立保存，未覆盖正在审核的174条 baseline：
`/path/to/private-archive`，
SHA256 `d6e8c2ed363c8960333fd9fadef4a6d55d1309a9749f0c04f44c4f8f6d83ebfb`。
重建前后报告在 `data/eval/staged_sft_before_scope_report_20260908.json` 和
`data/eval/staged_sft_after_scope_report_20260908.json`。抽审的12条文本未被
scope更改，但本轮10条SFT出口仍严格使用原冻结baseline及其审核；没有给
新版176条文件伪造全量审核通过标记。

训练出口现在同时检查：

1. 原始 source 的 row SHA、分区候选状态、quarantine reasons；
2. 重建候选报告对原始 lineage、完整候选文件和输出的绑定；
3. fresh set audit 对完整 system/prompt/completion 的内容 SHA；
4. 实际 protocol、endpoint、model、max_tokens 和 judge input_key，
   judge 必须完整结束；生成截断项不可放行；
5. lineage 报告与本次 held-out 文件的 SHA 必须一致；
6. 行内 evaluation-only、非 train split 或未显式晋级的 repair 标记，
   不能被 CLI 的 `--source-split train` 覆盖。

删除不安全 fact 后的 subset 只输出待重审候选，不直接成为 SFT/KTO。
原 `build_exx_clean_binding_sft.py` 还存在“验证失败仍写 train.json/registry，
最后才返回失败”的确定性问题；现改为先验证全部分区，失败仅写报告。
历史 materializer 成功也仅代表结构转换通过，不是训练语义门禁。

在冻结174候选中，按精确问题 family 和固定 seed 抽12条做全正文 set
审核，**12/12 完成，0 error，0 length**。GLM 判 38/38 facts entailed、
11/12 set complete，1/12 partial。所有可见证据均判 sufficient。
罗莎琳困境题虽然事实成立，但漏了成员失踪、外部危险等信息，再次说明
不能用逐条引用正确率代替问题回答完整率。

Codex 又对这12条的完整可见原文与输出逐条复核，同意隔离罗莎琳题，并额外
隔离潘乔题：GLM认为单句“希望龙舌兰关注玻利瓦尔未来”已完整，但该句漏了
原文的犹疑拖延、受坎黛拉影响等此次责备的直接依据。此分歧独立保存，未覆盖
GLM结果，也不把 Codex 复核叫作独立人审或新 gold。

新增 `--secondary-review` 可绑定冻结 predictions/audit SHA 作二次否决；
只可否决，不能把 GLM、协议或 lineage 已拒绝的项升级为通过。实际 CLI
出口为 **10 SFT / 0 KTO / 2复核隔离 / 162缺审核不放行**。这10条都是
answer_directly，只能验证审核与导出链路，不构成完整的训练配方；未启动
训练或修改 checkpoint。12题抽查也不是新的生产正确率或独立盲测。

归档：

- `data/eval/staged_sft_set_smoke12_20260908.json`
- `data/eval/staged_sft_set_smoke12_codex_review_20260908.json`
- `data/eval/staged_sft_set_smoke12_export_report_20260908.json`
- `data/eval/training_lineage_clean_sft_audit_20260908.json`
- `data/eval/training_lineage_v3_source_audit_20260908.json`

实际10条样本：
`/path/to/private-archive`，
SHA256 `b9d40fbff480a266cc6648eb0fb69885869df1c6811ea004b3d5df9a623257ff`。
新记录保留 source_question、来源 ID、故事/正文映射和原行 SHA，以免导出
后再次丢掉分区追溯信息。

```bash
PYTHONPATH=src python3 scripts/merge_gold_claim_scope.py \
  --facts /path/to/private-archive \
  --prior data/eval/gold_recalibrated_full_evidence_adjudications_20260907.jsonl \
  --scope data/eval/gold_claim_scope_20260908.jsonl \
  --quarantine data/eval/gold_claim_scope_codex_quarantine_20260908.json \
  --output /path/to/new_scope_view.jsonl

PYTHONPATH=src python3 scripts/stage_full_evidence_sft_candidates.py \
  --sources /path/to/private-archive \
  --facts /path/to/private-archive \
  --judgements /path/to/private-archive \
  --prior data/eval/gold_recalibrated_scope_reviewed_view_20260908.jsonl \
  --lineage data/eval/training_lineage_clean_sft_audit_20260908.json \
  --output /path/to/new_candidates.json

PYTHONPATH=src python3 scripts/build_relevance_clean_exx_data.py \
  --predictions /path/to/private-archive \
  --audit data/eval/staged_sft_set_smoke12_20260908.json \
  --source-split train \
  --heldout-json /path/to/private-archive \
  --heldout-json /path/to/private-archive \
  --lineage-audit data/eval/training_lineage_clean_sft_audit_20260908.json \
  --candidate-report /path/to/private-archive \
  --secondary-review data/eval/staged_sft_set_smoke12_codex_review_20260908.json \
  --out-dir /path/to/new_output
```

后续应先完成其余候选的完整性审核并修复真正的partial，随后在训练分区内
构造 Full/Distractor/Missing 状态、分别复核证据充分性与动作，再做受控
小规模训练对照。176条answer候选与10条通过样本都不足以替代这一配方；
独立问题族测试和实际 quote 端到端验收仍未完成。

## 产物、复现和验证

代码：

- `scripts/build_reader_attribution_ablation.py`
- `scripts/summarize_reader_attribution_review.py`
- `tests/test_build_reader_attribution_ablation.py`
- `tests/test_summarize_reader_attribution_review.py`
- `tests/test_grounding_visible_evidence.py`
- `tests/test_conclusion_sampling_failures.py`

审计记录：

- `data/eval/strict9_reader_attribution_review_20260908.json`
- `data/eval/strict9_reader_attribution_review_summary_20260908.json`

本地完整输入与输出：

`/path/to/private-archive`

远端隔离实验目录（未操作远端主仓库）：

`/path/to/private-archive`

| 资产 | SHA256 |
| --- | --- |
| 54 行 matrix | `a7e4ec23e7b446670234f279adb603b6a436b4c8026cc9310300f44c8501d40a` |
| base predictions | `4d560103dd0ee3297c5531382342ac25fa42bd78e4b76333490058f903fd79a0` |
| clean-SFT predictions | `bd75c358f775d5f632dc84b2d525398ac1e020012c5a2a540c0f8b6e1baf8f3c` |
| base 顶层资产聚合 | `4dcba47639cd5ccef2436373743b1f122d11d08c85aa63c34fc8ce37aba1ff2b` |
| adapter | `75c3f1e0c2946c77537d81266b408913ce51bd6a41563ee2ca884e3a76c3c41c` |

输入 snapshot、旧 matrix 与 annotation SHA 均记录在 matrix report 和旧
reader 报告。推理沿用 canonical non-thinking template、bf16、greedy、
`max_new_tokens=768`；使用隔离环境中的空闲 RTX4090。

```bash
PYTHONPATH=src python3 scripts/summarize_reader_attribution_review.py \
  --matrix /path/to/private-archive \
  --review data/eval/strict9_reader_attribution_review_20260908.json \
  --prediction base=/path/to/private-archive \
  --prediction clean_sft=/path/to/private-archive \
  --output /path/to/new_summary.json

PYTHONPATH=src python3 -m pytest -q
```

汇总工具拒绝 hash 不符、缺行、重复 review、prompt/diagnostic 漂移与动作
不一致的标签，不会把未审计行自动补成成功。

grounding/reader 修复阶段为256 tests passed；训练入口续审后为303 tests
passed（含真实协议训练出口、陈旧内容拒绝、scope投影、来源继承、复核否决
以及禁止通过旧正例标签绕过隔离）。
`compileall`、`git diff --check` 通过。
未提交或 push；既有未跟踪 `experiments/` 未纳入，也未删除。
