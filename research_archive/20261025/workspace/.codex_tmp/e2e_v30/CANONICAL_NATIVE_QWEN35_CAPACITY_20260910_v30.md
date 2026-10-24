# ASA 原生 Qwen3.5-4B / 9B 容量与输出协议诊断 v30

241 为唯一规范源，248 为训练镜像。MiniRAG 禁用。
本轮首次补齐架构备忘录中未运行的同系列 Qwen3.5-9B；旧 v19 是 Qwen3-8B，不能混同。

## 结论

32 次生成及后验全部完成。两个原生非思考候选都不接入默认链路。
9B quote 机械放行由 4/8 增至 5/8，但额外放行包含错族群、错人物关系；
自然语言条件也没有消除主体错认。不能以模型变大或答案更长宣布准确率提升。

## 新资产和运行条件

Hugging Face 直连检查失败。之后从官方 `Qwen/Qwen3.5-9B` ModelScope 仓库读取完整文件清单，
按每文件 revision、大小和 SHA256 固定下载，13 个元数据/分词文件、4 个权重分片全部核验。
未使用未知量化副本、未运行下载仓库的 Python、未更新系统依赖或切默认模型。

```text
${REMOTE_ASA_ROOT}/rank_v8_coverage/models/Qwen3.5-9B_modelscope_20260910
```

系统盘当时仅剩约 23GB，因此全部资产放到另有约 14TB 可用的数据盘。
规范下载清单、各文件 commit 与 SHA 保存在实验目录的 `official_download_manifest.json`，
元数据和权重分别留存回执。不把可变的 `master` 当作复现用唯一标识。

两模型分别在同一 4090 上重新连续检查 60 秒空闲后顺序运行。
BF16、原生 chat template、`enable_thinking=False`、无 LoRA、text-only、eager、单请求；
context12288/output2048，temperature0.1/top_p0.8/repetition1.05。
4B GPU 上限0.52，9B为0.90；9B text-only 加载约16.8GiB，采样期间观察到总显存约22455MiB。
这些是本机实测和本次条件，不推断训练显存或其他环境容量。
采样沿用实验条件以比较，不宣称为官方推荐的最优设置。

## 对照设计

八个反复检查过的开发问题，每个模型分别运行：

1. quote：原 canonical system/user、hypothesis、完整证据和 schema，只用原生非思考包装，不强制左花括号。
2. plain：简洁自然语言任务，同问题和完整证据，不带 hypothesis、JSON schema 或引用要求。

plain 是组合协议干预和能力诊断，不是纯去掉 JSON；它没有通过生产 grounding，不允许直接替换正式展示。
同一协议内，两模型独立加载各自 tokenizer 后，全部 prompt 字节和 token IDs 一致；
每题 paired seed 一致，quote/plain 交替执行，模型间顺序执行而非交错。

| 模型 | quote stop | quote direct / abstain | plain stop | 总调用 |
|---|---:|---:|---:|---:|
| 原生 Qwen3.5-4B | 8 | 4 / 4 | 8 | 16 |
| 原生 Qwen3.5-9B | 8 | 5 / 3 | 8 | 16 |

两组全部逐请求核验 token IDs、seed、无 LoRA、完整输入、输出预算、模型/源码哈希，
quote 按实际原证据重放结果一致。以上是机械状态，不是 50%/62.5% 正确率。

## 内容结论

- 林地迁徙：9B依然把萨尔贡迁徙写成所问族群；plain甚至混称“阿卡胡拉丛林的萨米人”。
- 夜刀：两个quote都答天灾；原文E6明确指向矿石病。4B plain才提到矿石病，9B plain仍混同灾害阶段。
- 初雪：两模型plain均明确把初雪等同于雪绒，混入茱安娜父亲的车站事件；9B还出现父母双亡/母亲抚养自相矛盾。
- 小大厨：两个quote均有用心、进步的有据核心，9B展示保留“开窍”。
- 雅拉：9B quote仍套用缪尔赛思的迷茫经历；9B plain改用未出现的直接引语、把孩子贺卡推成与父亲密切交往。
- 拉普兰德：两quote均因引文/编号问题弃答；9B plain去掉4B的虚构追杀动机，但又将带队独断专行说成独自行动。
- 诺希斯：9B raw quote找到未来承诺，却仍因复制/绑定问题弃答；两plain有正确核心，9B额外推断首次相识、放弃安逸/复仇。
- 阿娜特：quote比此前部分LoRA输出更明确包含节奏中断，但仍有过强心理因果桥；plain也有额外无据解释。

复核是助手源文检查，不是独立人审。逐题证据和实际展示反例记录于 `source_review_v30.json`，
由脚本校验摘句存在、错误展示片段确实出现，再以 `checkpoint_v30.json` 哈希归档。
不汇总未经全面认证的完整正确数，也不把这些八题称为新 holdout。

别名观察也需精确：原规范表将初雪关联到恩雅；quote 的 hypothesis.keywords 已带“圣聆初雪、恩雅”，
但未显式表达别名关系。因此不能说 reader 完全没有这些名字。

## 后续

v31 单独比较同一9B开启/关闭思考，并给两组完全相同的4096总输出预算。
预算包括思考和最终答案，未完成思考或长度终止保留失败，不偷偷增加推理预算或修答案。
不把当前非思考诊断当作9B能力上限。

```text
${REMOTE_HOME}/ASA-ArknightStoryAgent/outputs/experiments/reader_native_qwen35_20260910.v30
```

没有训练、生产源码变更、默认模型切换、提交/push 或服务重启。
