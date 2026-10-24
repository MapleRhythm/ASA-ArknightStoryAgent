# ASA Exx grounding 分析结论

## 结论先行

三个瓶颈不是同一类问题：

1. **格式**：SFT 已经学会了协议的外壳，但生成仍是自由解码。剩余的 `action:invalid` 主要来自“输出很长、重复事实/关键词、在数组结束后少一个外层 `}`”以及达到 `max_tokens`，不是输入 prompt 太长。KTO 没有改善这个瓶颈；把上限从 1024 降到 512 后，截断反而更多。
2. **判断**：`retrieve_more` 的低准确率首先是**证据充分性没有被表示成模型可见的变量**：模型看到若干支持片段，却没有“哪一个必需链环缺失”的显式状态。其次才是 4B 的能力/校准问题：在部分支持时抢答、误读和编造引用。因而不是“信息不可得”与“能力不足”二选一，而是“上游缺失性不可见，触发下游能力错误”。
3. **多轮召回**：后续轮次不是在搜索新的语义邻域，而是在同一问题上加少量泛化词并重排同一批文档。round3 的检索集合几乎收敛，故新鲜 recall 下降。10 个全程 miss 里至少若干原子片段在库中，不能把主要责任归咎于 corpus 缺失；更像是 follow-up hypothesis 没有针对缺失链环进行检索。

## 口径与补查范围

- 格式数据：248 上的
  - `fixed_evidence_success559_val79_full_20260828/success559.predictions.json`
  - `fixed_evidence_kto_v7_val79_20260828/kto_v7.predictions.json`
  - 对应 `*.metrics.json`
- 多轮数据：241 上的 `outputs/retrieval_eval/multiround_v2_lora_sample50_gpu.json`。
- 多轮 trace 的 `top_doc_ids` 每轮只记录 5 个文档，因此下面的重叠度是 **top-5 trace 重叠**；recall 数字仍采用原报告的 @10/@50。
- 另外对 241 的 `indexes/arknights_story/documents.jsonl` 做了归一化全文扫描。`gold_excerpt` 是多段证据拼接物，完整字符串不命中并不等价于“库中没有任何支持片段”。

## 1. 什么影响格式正确率？

### 根因判断

SFT 的提升来自**任务分布对齐**，不是模型突然具备了更强的 JSON 推理能力。旧 quote 模型学到的是“复述 quote”的输出分布，把它放到 Exx 的 E-ID 协议上，schema 只有 6.3%；SFT 直接把 `next_action`、`supported_facts`、`evidence_ids` 等目标结构写进训练目标，于是 strict JSON 从 68.4% 到 87.3%，schema 从 6.3% 到 84.8%。KTO 是在 SFT 之后做偏好优化，主要改变语义/动作偏好，没有增加语法约束，故格式不升反降（strict 86.1%，schema 81.0%）。

需要区分两个指标：

- `action:invalid` 是解析失败，SFT 为 10/79（12.7%），KTO 为 11/79（13.9%）。
- 另有少量“能解析但 schema/E-ID 不合规”的行；SFT 和 KTO 的 invalid E-ID 都是 2/79（2.5%）。因此不能把所有 schema 问题都解释成 decode 截断。

### 248 抽查得到的共性

| 现象 | SFT | KTO | 机制含义 |
|---|---:|---:|---|
| `finish_reason=length` | 3 行，均正好 1024 token | 4 行，均正好 512 token | 输出预算是实因；512 上限会把长输出截在 JSON 中间 |
| `finish_reason=stop` 但 JSON 解析失败 | 7 行 | 7 行 | 每一行都只差一个外层 `}`：输出已经结束 `supported_facts` 数组，却没有闭合最外层对象 |
| stop-malformed 行加一个 `}` 后可解析 | 7/7 | 7/7 | 这是终止/格式控制边界问题，不是 JSON 内容完全不可恢复 |
| 典型输出形态 | 反复复制同一事实，填到 8 条；或 follow-up 的关键词循环重复 | 同样的重复模式 | 模型倾向“多写一些支持事实”，没有最小充分输出的停止策略 |

SFT 的 3 个截断例子中，两个是大段重复关键词/事实直到 1024；KTO 在 512 上又多出两个同类长题（副总统杰克逊、塔露拉）。这解释了为什么降低 KTO 的 `max_tokens` 后，格式没有改善。

7 个 `stop` 失败行并非输入被截掉：它们的 prompt token 数大多只有 1750–1920，生成长度却达到 89–326 token，尾部规律都是 `...}]` 而不是 `...}]}`。把这 7 行各补一个 `}`：

- SFT 的 strict JSON 可由 69/79 变为 76/79；`action:invalid` 由 10 行降到 3 行。
- KTO 的 strict JSON 可由 68/79 变为 75/79；`action:invalid` 由 11 行降到 4 行。

这是“安全修复后”的反事实上限；补括号后仍必须重新跑 E-ID/cardinality 校验，不能把任意截断文本直接放行。例如 SFT 的美食节样本补括号后仍有每条事实绑定过多 E-ID 的问题。

### 不是主要原因的因素

- **输入 prompt 长度**：SFT action-invalid 行的 prompt 平均约 2965 token，action-valid 行约 3525 token；坏行反而更短，排除“上下文太长导致输入截断”作为主因。
- **query type**：SFT 的 10 个 action-invalid 为 causality 7、fact 2、reveal 1；KTO 为 causality 7、fact 1、reveal 3。causality/reveal 有轻微聚集，但样本很小，且同一批 `relabel-*` 行跨模型复现，说明内容复杂度/输出长度比 query_type 标签更接近真正原因。
- **证据条数**：SFT action-invalid 中 7/10 恰好是 6 个证据块；6-block 子集的坏率约 18.9%，高于 12-block 子集约 5%，但不是单调关系，且与特定 relabel 家族、事实条数和重复生成混杂。不能据此把“证据多”视为因果变量。

### 因果链与反事实

当前链路是：

`SFT 学到结构外壳` → `生成时仍自由展开事实/关键词` → `输出变长并重复` → `EOS 或 max_tokens 出现在外层对象闭合之前` → `整行被判 invalid`。

如果只把 `max_tokens` 从 512 调到 1024，KTO 的两个明显长题大概率可避免截断，但 SFT 已证明 1024 仍会有 3 行截断；所以“单纯加预算”不是完整解法。若加入 JSON grammar constrained decoding、最小事实条数策略和安全的单括号修复，格式损失应能从约 13% 降到几个百分点，且不需要更换基座模型。

## 2. 什么影响判断率？为什么 retrieve_more 只有 28.7%？

### 根因判断：上游是“充分性不可见”，下游是“部分证据下能力/校准不足”

`answer_directly`（82.5%）和 `abstain`（78.8%）主要是正例匹配或保守拒答；`retrieve_more` 要求模型做一个更难的集合判断：

> 当前证据是否已经覆盖问题所需的每一个原子链环？如果没有，缺的是哪一环，下一轮要查什么？

数据四说明这个变量在输入中没有被显式表示：

- top-12 中 evidence chain 完整仅 13.5%，partial 49.5%，none 37.0%；即 86.5% 不完整。
- 用 fusion max/mean/std/gap/高分计数等候选分数池化特征预测“不充分”，AUROC 全在 0.45–0.55。单个候选分数看不到“缺哪一环”，因此也不存在一个简单阈值让 4B 稳定做出 retrieve_more。

这使模型落入一个有偏的默认策略：看见一两个相关片段就先 `answer_directly`，把“部分支持”误当“完整支持”。数据三的盲审验证了这不是教师标签噪声：50 个 `retrieve_more → answer_directly` 分歧中，仅 1 个（2%）可认为模型其实完全正确；17 个（34%）是缺关键面仍抢答；32 个（64%）明确 unsupported，其中 14 个引用幻觉、13 个误读证据。

所以因果分解是：

`召回集合缺链环` → `缺失状态没有显式表示` → `模型无法可靠知道“还缺什么”` → `4B 以 answer 先验抢答` → `partial/unsupported + 错误 E-ID`。

其中最后一步也有独立的能力问题：即使给了部分证据，模型仍会误读或编造引用。因此结论不是“模型只要拿到信息就会做对”，而是“在信息不完整且缺口未结构化时，模型的能力错误被系统性放大”。

### 可区分两种原因的判别实验

做一个配对、随机化的 evidence-completeness factorial replay。每道题固定同一问题、同一批干扰证据、相同 token 预算和证据顺序随机种子，只改变 gold 链覆盖：

| 条件 | 输入构造 | 期望动作 |
|---|---|---|
| Full | 注入完整 gold evidence chain | answer_directly |
| One-edge-missing | 从 Full 只删除一个必需链环 | retrieve_more |
| None | 删除全部 gold 链，保留等量干扰证据 | retrieve_more/abstain |
| Decoy-full | 放入高分但不组成完整链的相关片段 | retrieve_more |

每个条件跑两种 prompt：

1. 当前的 raw evidence prompt；
2. 诊断用的“slot ledger” prompt：把问题拆成若干待验证槽位，并让模型逐槽标注 `supported / missing`（槽位标签由测试 harness 注入，不用于线上训练）。

记录四类指标：动作 confusion、显式 sufficiency/missing-slot 的 precision-recall、Full 条件下的答案/引用准确率、One-edge-missing 条件下的拒答/续检索率。

判别规则：

- **Full 仍低答对率**：即使信息齐全也不会做，说明能力/推理不足。
- **Full 高、One-edge-missing 仍大量 answer**：答案能力尚可，但缺失性不可见，是表示/coverage gate 问题。
- **slot ledger 一加就显著改善 One-edge-missing**：原始片段中信息并非不存在，而是没有被组织成模型可操作的缺口状态。
- **slot ledger 也救不回来，且引用仍错**：才应把主要预算投向更大模型、蒸馏或专门的 claim-citation 训练。

这个实验比单看现有 action accuracy 更有因果性，因为同一问题的语义难度和干扰证据被配对控制，唯一随机化因素是“证据链是否完整”。

### 反事实

若把完整 gold chain 作为 oracle 注入后，`answer_directly` 的条件准确率接近当前 answer 类的 80% 水平，而 One-edge-missing 仍抢答，则主要收益来自 coverage gate/slot 表示，不是换生成模型。反之，Full 也低，则 retrieve_more 低并不能归因于检索信息不足，必须改模型能力或任务分解。

## 3. 为什么多轮召回增益不大？

### overlap 审计（241，v2 LoRA sample50）

| 过渡 | 样本数 | top-5 Jaccard 均值/中位数 | 完全相同 | hypothesis question 完全相同 | keyword 集合 Jaccard |
|---|---:|---:|---:|---:|---:|
| round1 → round2 | 46 | 0.729 / 0.667 | 20/46 | 46/46 | 0.779 |
| round2 → round3 | 35 | 0.936 / 1.000 | 29/35 | 35/35 | 0.932 |

round1→round2 的 46 个过渡中，14 个保留 4/5 文档、8 个保留 3/5、3 个保留 2/5，仅 1 个完全不重叠。round2→round3 更明显：29/35 五个文档完全相同，另外 5 个只换一个、1 个只换两个。

follow-up hypothesis 的**问题文本在每一个过渡中都没有变化**。实体集合也高度稳定（Jaccard 均值 round1→2 为 0.928，round2→3 为 0.959）；变化主要是给 keywords 追加泛化词或别名（keyword Jaccard 从 0.779 收敛到 0.932），而不是明确指出“缺失的那条关系/事件/时间槽位”。

这与 recall 数字构成同一条因果证据：

- cumulative @10：0.68 → 0.74 → 0.76，仅 +6pt、再 +2pt；
- round-only @10：0.68 → 0.68 → 0.52，第三轮新鲜质量下降；
- 40 个命中中 round1=37、round2=2、round3=1，另有 10 个全程 miss。

### 代表性 trace

- **天火为何解救感染者**：round1/2 的 hypothesis 和关键词完全相同，top-5 仅保留/重排同一批 `story_skfire` 文档；round3 只加“解救、整合运动”，换进一个新文档后才在 rank5 命中。这个增益是一个新锚词加偶然排名变化，不是稳定的缺口推理。
- **卓娅的母亲遭遇**：round2 hypothesis 与 round1 完全相同，却把一个 `act10d5...st07` 文档挪进 top-5 并在 rank5 命中。这里的收益来自查询集合/融合重排变化，不是 follow-up 语义改进。
- **谁策划绑架**：round2/3 加入“主谋、幕后、恩希欧迪斯、凯特、卡尔、银灰”等候选名；这些是猜测性实体，不是由当前证据导出的缺失槽位。文档仍大部分来自同一 `story_svrash`，最终没有命中。
- **真理本名与学校关系**：只增加“真理真实姓名”，没有把“古米/凛冬各自学校关系”拆成独立检索槽位；top-5 仍几乎相同，最终 miss。

### corpus 覆盖补查与判断

对 10 个全程 miss 的 `gold_excerpt` 做归一化全文扫描，166MB 的 `documents.jsonl` 中没有一个完整拼接字符串直接命中。这一结果不能证明 corpus 缺失，因为 gold_excerpt 本身是多个 chunk 的拼接。

更有区分力的是原子片段：至少以下 miss 的关键片段在库中明确存在：

- 真理/古米/凛冬的学校关系与真理本名；
- 阿米娅感知娜塔莉娅强烈情感、因此改变调岗决定的对话；
- 罗莲娜放走卓娅及“离开校园后外面并不太平”的相关片段。

因此“gold 完全不在库里”不是主要解释。剩余 miss 仍可能包含 paraphrase、切片边界或标注摘要不与原文逐字相同的情况，需用 span-level coverage 而非完整字符串继续核验；但现有证据已经足以把主要责任放在**缺失槽位没有驱动 follow-up、后续检索缺少新颖性约束**上。

### 根因排序与反事实

1. **follow-up 多样性不足**：问题不变、实体基本不变、关键词只做同义扩展，导致同一语义邻域重复检索。
2. **hypothesis 质量在后轮退化**：round3 的追加词更泛化，甚至注入未经证据支持的候选实体；这与 round-only recall 从 0.68 降到 0.52 一致。
3. **融合策略带来的排名抖动**：有的 round2 命中只因同一 query 在不同 query list 下重新排序，收益不可复现，也不是 planner 真正找到新证据。
4. **corpus/切片覆盖是残余因素**：10 个全程 miss 不能排除部分覆盖缺口，但它解释不了“后轮文档几乎不变”和“少数后轮仍能偶然命中”。

反事实：

- 如果 round2 生成的是“每个 missing slot 一个独立 query”，并对已见文档施加 MMR/novelty penalty，top-5 overlap 应下降；若 overlap 降而 recall 不升，才说明 corpus coverage 是主因。
- 如果保留原始 query，同时追加 targeted query，而不是只用一个扩展 query，`卓娅母亲` 这类由融合重排造成的偶然收益会减少，但检索行为会更稳定、更可解释。
- 如果在 round2 直接 oracle 注入缺失链环仍不能命中，则应回查索引切片/chain graph，而不是继续堆多轮。

## 按 ROI 排序的可执行改动

### 1. 约束 JSON 解码，并做严格的安全修复

- 在生成端使用 JSON grammar/结构化 constrained decoding；
- 对“只缺一个外层 `}`、无尾部垃圾、finish=stop”的输出做白名单修复，再跑完整 schema/E-ID 校验；
- 任何未知 E-ID、过多 ID、非法 action 仍拒绝。

直接收益最大：仅括号修复就可恢复每个模型 7 个 strict 失败行；不需要训练新模型。

### 2. 把输出改成“最小充分事实”，并给长度自适应预算

- 训练/提示明确：只输出回答问题所需的 1–3 条原子事实，除非确有互不重复的独立链环；
- 服务端对 normalized fact 去重、限制关键词数量，检测重复循环后提前停止；
- answer 与 retrieve_more 使用不同的生成预算；长题至少允许 1024，但不要用无限预算掩盖重复。

这同时降低截断、过多 E-ID、事实条数溢出和少一个括号的概率。

### 3. 增加 set-level evidence coverage gate

- 在生成 action 前，把问题拆成 slots/edges，计算每个槽位是否被当前证据支持；
- 让 `retrieve_more` 绑定一个显式 `missing_slots`，而不是只输出泛化 hypothesis；
- 训练 hard negatives：partial evidence、看似相关但缺关键边、错误 E-ID。

这是解决 28.7% retrieve_more 的最高杠杆项；单调 reranker candidate score 不足以替代它。

### 4. 重写多轮 planner：针对缺口检索并强制新颖性

- 每个缺失 slot 生成独立 query，保留原 query 作为锚点；
- 对已见 doc/source 加 novelty penalty，加入 source/episode 多样性约束；
- 若 follow-up 与上一轮 question/keywords 基本相同且没有新文档，直接停止 round3，改为 abstain 或请求更强检索。

预期效果不是“再跑一轮”，而是让 round2 真正搜索不同证据邻域，并减少 29/35 的 round2→3 完全重复。

### 5. 建立可归因的离线评测面板

- 每轮记录 full/partial/none chain coverage、missing slots、top-k overlap、novel-doc rate；
- 对全程 miss 做 span-level corpus coverage/exhaustive lexical 检查；
- 同时报告 strict JSON、schema、invalid E-ID、action accuracy 和 conditional accuracy，避免重复事实等质量问题被当前 validator 隐藏。

这项改动成本低，却能把“索引没覆盖”“planner 没换 query”“模型不会判断”三类回归分开，避免以后只看一个总 recall 或总 action accuracy。
