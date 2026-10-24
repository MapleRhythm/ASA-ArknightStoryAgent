# quote 直接正文 A/B：覆盖增加没有转化为稳定正确率提升

日期：2026-09-08。承接 `ACCURACY_IMPROVEMENT_RESEARCH_20260908.md`。

## 决定

**不推广直接正文 renderer，不切换模板或权重。**本轮完成36条真实GPU
生成、36条机械校验和全部输出/原文的Codex语义复核，结果不足以证明这一
展示改动提升最终回答正确率。直接正文确实恢复了证据，但 reader仍错归
主体、添加关系、错选引用；校验重写又可能丢失必要信息。

这是已有strict9上的4B基座首轮quote实验，不是当前生产正确率。仓库默认
配置的发布LoRA在241对应路径不存在，故本轮明确使用无adapter基座，没有
拿历史Exx adapter替代发布模型。没有独立人类审核，也不是新的盲测。

## 资源与实验合同

开始检查时241的GPU2没有计算任务，随后出现临时进程，第一次启动被空闲
检查拒绝，未加载模型。该任务自然结束后再次检查通过，才使用GPU2。
其他共享 GPU 均有任务，本轮未使用、未停止它们。

- 运行卡：一张空闲 RTX4090；具体设备标识保留在私有运行归档。
- 模型：`/path/to/private-archive`，无LoRA。
- 顶层模型资产聚合SHA：
  `4dcba47639cd5ccef2436373743b1f122d11d08c85aa63c34fc8ce37aba1ff2b`，
  与此前reader实验一致。
- Transformers 5.6.2、PyTorch 2.10.0+cu128；bf16、greedy、
  repetition_penalty=1、max_new_tokens=1536、max_context=10000。
- 未改权重、生产配置、依赖环境或远端主仓库。仅在隔离目录运行。
- 任务结束后GPU2恢复18MiB，无本次计算进程。PyTorch记录的峰值分配
  为10,258,072,576字节，约9.55GiB；不等同于全部驱动/进程显存。

冻结9题 ×2正文展示 ×2模板，共36输出。同题两展示交替顺序；四个条件
都用同一模型、GPU及解码设置。参考答案、gold IDs、reviewer notes均不
进入模型消息。

| 因素 | 当前逻辑 | 对照 |
| --- | --- | --- |
| 正文展示 | 选择器不变；展示chain优先正文、每段截1000 | 同样selected IDs/E-ID顺序；仅改为direct clean_text，仍截1000 |
| 提示模板 | `runtime_raw`：逐字使用runtime prompt，保留assistant `{` 前缀 | `canonical_non_thinking`：同一system/user，使用checkpoint tokenizer的非thinking模板，不预填`{` |

top-k10、总证据预算12000、轮次1/2不变。模板是独立分层因素，不能把换
模板后的结果计为renderer收益。生成用Transformers确定性控制，而不是
实际部署vLLM的采样参数，所以即使runtime_raw也不称生产E2E。

最长实际输入4692 token，未触及上下文预算。36/36生成结束，0触及输出
token上限；但runtime_raw/chain有2条在JSON中途EOS（莫斯提马、芙蕾达），
**stop不等于结构完整**。保留原始输出，未通过增加token预算或截断恢复
生成来掩盖失败。

## 模型输出与校验结果

“校验后回答”是当前源码正常解析、quote校验、必要的事实重写及后续校验
实际返回的 `answer_directly`，不是只看quote有无hard issue。

| 模板 / 展示 | 补runtime前缀后完整JSON | 校验后回答 | 最终完整受支持回答 |
| --- | ---: | ---: | ---: |
| runtime_raw / chain | 7/9 | 3/9 | 0/9 |
| runtime_raw / direct | 9/9 | 4/9 | 0/9 |
| canonical / chain | 9/9 | 4/9 | 1/9 |
| canonical / direct | 9/9 | 5/9 | 0/9，另1条边界项 |

语义标准沿用此前冻结的必要要点：多子问完整、无新增无依据实质断言。
正确额外背景不直接判错。`raw_content`按该行全部可见正文判内容，
不代表所选引用有蕴含支持；`validated_content`另读校验实际返回的文本。
不能把原final_answer的评分复制到被改写后的答案上。

canonical/direct的迷迭香题，重写后已恢复主要对比，但把“有些感觉只有
战场上才能找到”简化为“只有战场上才能找到感觉”，有范围扩大争议。
保守标 `uncertain`，不计成功，待独立复核。**即使放宽这条，也只是
canonical的1/9→1/9，不构成净提升**；原来通过的杰米题则退化。
9题反复条件不能当36个独立问题，不计算生产准确率或声称统计显著。

### 具体失败如何传导

1. **莫斯提马：看到了关键句，仍认错人。**direct的两种模板都恢复
   “普通的信使”自称，却将同事的“苦难陈述者”职务归给她。
   两条都被校验放行。更多引用通过不等于更安全。
2. **老天师：错误关系能够通过词法核验。**direct将樵夫守护女儿写成
   守护诡异织物/视织物为家；源码校验保留此错，且没有回答她对织物
   毫不留情的态度对照。
3. **带剑女子：额外关系毁掉正确部分。**某些输出正确说出器物化伥，
   却加入恋人、恩师或把剑客写作麟青砚。恢复正文没有稳定消除这些错误。
4. **女妖：内容正确和引用正确是两件事。**direct/raw回答“娜斯提”
   正确，但引用的是遗愿/荒原回忆，不是已经可见的直接点名段，校验后
   没有答案。direct/canonical仍选择retrieve_more。chain/raw还捏造
   “已故的亲人娜斯提”。
5. **杰米：直接展示会引入新的观点错归。**canonical/chain唯一通过；
   direct/raw把另一感染者对陈说的“荒原/铁盒/没钱”归给杰米。
   direct/canonical把查丝汀娜个人经历并入解释，校验再缩成泛泛的
   绝症/排斥/无生存空间，漏掉必要原因。
6. **雷吉斯：修复步骤丢失正确要点。**direct/raw原答案有战争分歧、
   生前不归和团聚，但supported_facts没有战争分歧，warning触发重写
   后将其删除。direct/canonical反过来在supported_facts漏团聚，
   因此重写删掉团聚。不能通过简单保留原答案解决，因为其他题原答案
   确有无依据事实。
7. **芙蕾达：主体归因仍是核心瓶颈。**多个输出把“被骂叛徒”的对象
   从芙蕾达换成威尔，并反转谁认为谁在保护大家；恢复妥协原因并未
   修正主体。canonical/chain还把错误事实放行。

复核也必须防止自身误判：杰克逊的E10确实有“我们掌控”与能量井监控，
E8也明确把雅拉称作玛丽安娜·布莱克；不能因为陌生名字或只看E9就判
这些内容是幻觉。direct/canonical该题的科研/竞选代词归属及政治动机
概括留作边界项；其引用拼接问题已确定，校验后未产出答案。

## 成本：更短的输入没有带来更快的生成

| 模板 / 展示 | 平均输入token | 平均输出token | 平均生成秒数 |
| --- | ---: | ---: | ---: |
| runtime_raw / chain | 4094.56 | 375.11 | 14.63 |
| runtime_raw / direct | 3036.89 | 414.11 | 15.86 |
| canonical / chain | 4097.56 | 344.11 | 13.46 |
| canonical / direct | 3039.89 | 377.33 | 14.61 |

direct减少约1058输入token，但输出变长，本轮实际生成反而稍慢。
总逐行生成计时526.93秒；计时包含模型generate的prefill/decode及同步，
不含模型加载、资产hash、后处理、检索、联网和报告写入。不是服务E2E时延，
也没有重复运行来估计计时波动。

## 下一步应该解决什么

停止在strict9上继续叠加通用提示、改top-k或取消去重。此次已完成之前
计划的最小renderer验证，没有通过晋级门槛。

下一轮工作集中在两个可验证的训练目标，先补数据，不直接开训：

1. **完整答案与事实/引用一致。**从隔离训练分区的完整原文重建答案，
   让所有必要要点都进入supported_facts，每条quote指向真正证明该事实
   的可见段。逐事实蕴含、整题完整性和生成协议三层同时审核。
   176个候选尚待fresh set audit；10条answer-only样例不能成为训练配方。
2. **抗主体错绑与动作监督。**用真实说话人/对象/否定/因果对照及Full、
   Distractor、Missing状态。对每条反例明确错误主体、正确主体与来源跨度；
   不从被删除的gold正文向缺口生成器泄漏答案。同一事件/改写族跨条件一起
   隔离，strict9及本次36条输出只作评测回归，禁止回灌训练。

训练验证必须比较同base、相同token/步数下的清洁Full正例、关系/干扰
增强、再加动作监督三组，并用新独立问题族验收。语义verifier先做影子
评估，报告误放和误杀；不为提高通过率放宽quote gate，不靠直接返回
未经验证的final_answer掩盖重写缺口。

## 文件、哈希与复现

本地冻结输入、预测及运行metadata：

`/path/to/private-archive`

远端隔离目录：

`/path/to/private-archive`

| 文件 | SHA256 |
| --- | --- |
| matrix.json | `7ebd491e25a49b9299d27004c7fd4511ef5b9aa45cbf28ccb079cadc5a5b72e9` |
| predictions.json | `752c4fb83988f5585b35a13664f3afd5c92e17d12e7393c6c21c741fedc10b1a` |
| mechanical audit | `1e86497cca618d717aceb96aca7fbd48211d48e082ad93945f12d88cd266a3f6` |

仓库新增：

- `scripts/build_quote_render_ablation.py`：从冻结visibility审计恢复实际
  prompt，逐字核对哈希及非证据指令，生成无gold的36条输入。
- `scripts/run_quote_render_ablation.py`：检查目标GPU UUID/空闲、拒绝
  覆盖输出、加载前再次检查、拒绝输入截断、逐条fsync原始输出。
- `scripts/analyze_quote_render_ablation.py`：内容绑定后重放当前解析和
  校验，保留原动作、实际返回动作/答案、引用问题，不自动打语义通过。
- `scripts/summarize_quote_render_review.py`：要求完整review集合、SHA和
  动作一致，边界项不计成功。
- `tests/test_quote_render_ablation.py`：14项回归。
- `data/eval/strict9_quote_render_mechanical_audit_20260908.json`
- `data/eval/strict9_quote_render_semantic_review_20260908.json`
- `data/eval/strict9_quote_render_review_summary_20260908.json`

```bash
python3 scripts/analyze_quote_render_ablation.py \
  --matrix /path/to/private-archive \
  --predictions /path/to/private-archive \
  --snapshots /path/to/private-archive \
  --output /path/to/new_mechanical_audit.json

python3 scripts/summarize_quote_render_review.py \
  --audit data/eval/strict9_quote_render_mechanical_audit_20260908.json \
  --review data/eval/strict9_quote_render_semantic_review_20260908.json \
  --output /path/to/new_summary.json

PYTHONPATH=src python3 -m pytest -q
```

完整验证：331 tests passed，compileall及diff-check通过。
未提交、未推送、未正式训练；前轮dirty工作与`experiments/`均保留。
