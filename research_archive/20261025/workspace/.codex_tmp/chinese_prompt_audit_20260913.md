# 中文六题：证据进入 prompt 的只读诊断

日期：2026-09-13。未调用模型/API、未训练、未修改生产代码或配置。

## 结论与置信边界

不是完全没有传证据，而是存在关键证据在选择、句段裁剪中丢失的问题，同时混入了与剧情原文冲突的百科内容。

- DeepSeek：依据测试脚本逐题重建请求 body，6/6 与结果中的 `request_sha256` 一致。这能核验脚本发送的内容，不是服务端输入 token 的快照。
- 4B：依据当次结果中的运行配置、保存的最终检索证据，以及远端现有源码做静态核验。结果未保存逐轮完整 prompt、精炼后的证据和 token 输入，不能宣称已回放当时全部输入或唯一定位拒答原因。
- 本地模块化源码与远端 `goldenglow/inference/cpu_pipeline.py` 不是同一实现；以下 4B 诊断使用远端源码。

## 1. 4B 薇卡题：关键关系句被前 32 句段上限排除

远端源码：
`${REMOTE_HOME}/ASA-ArknightStoryAgent/src/goldenglow/inference/cpu_pipeline.py`

- 第 3462 行 `split_evidence_strips` 用换行、句号、感叹号、问号、分号分段，再直接 `return strips[:max_strips]`。
- 第 6183 行 `refine_evidence_strips` 先调用上面的分段截取，再交给 reranker 选择句段；增加相关性精排不能恢复此前已被截掉的后文。
- 当次配置：CRAG 开启，`max_sentences=32`、`top_sentences=5`、`require_full_documents=false`，每文档最终上限 1000 字，总上限 12000 字。
- 精炼后的副本删除 `evidence_chain_text`，因此不能假定证据链字段会把被截掉的关系句补回来。

对保存证据直接调用远端源码中的纯文本分段函数（未加载模型）：

| 最终检索排名 | 文档 | 总句段数 | “最好的朋友”句段位置（从 1 开始） |
| --- | --- | ---: | ---: |
| 1 | `activities/act10d5/level_act10d5_st02#chunk-0002` | 72 | 42 |
| 2 | `activities/act10d5/level_act10d5_st02#chunk-0001` | 69 | 43 |

32 条最终检索证据中，仅这两条 `clean_text` 含“最好的朋友”。对各文档取前 32 句段后，该字符串在候选句段中完全消失。
“我从来没有忘记过”在排名第 5 条，仍在其前 32 句段内；是否被后续精排保留，没有当时快照可证。

因此，按所查远端实现与记录配置，最直接的人物关系句无法从上述两个正文进入精炼结果。这是明确的证据准备缺陷，不等于已证明拒答只由它导致。

## 2. DeepSeek：前 10 条正文，不是 4B 的最终 prompt

脚本 `.codex_tmp/deepseek_chinese_pilot_20260913.py` 第 108 行附近：
只拼接 `row["evidence"][:10]` 的 `id` 和 `clean_text`，不取 `evidence_chain_text`，也不经过 4B 的去重、复合问题选证、CRAG 和渲染器。

- 真理姓名/学校题：全名“安娜·莫罗佐娃”出现在第 12 条百科正文，未进入 API 请求；前 10 条有“安娜”，没有完整姓名。
- 该题第 1 条百科正文把古米列为彼得海姆学生，并把凛冬、烈夏与早露、真理分为两校。API 的学校答案沿用了它。
- 与之冲突的游戏原文是：“我和古米，还有娜塔莉娅，是同一所高中的学生，而凛冬还有烈夏属于另一所学校。”原始文件 `data/ArknightsGameData/zh_CN/gamedata/story/activities/act10d5/level_act10d5_st02.txt` 第 72 行。该句不在这道题的最终 32 条 `clean_text` 中，也未出现在其证据链文本中。
- 薇卡题：API 的第 1/2 条正文包含“最好的朋友”，第 3 条包含娃娃，第 5 条包含“从来没有忘记”；核心证据确实进入了 API 请求。不能把 API 此题所有回答细节自动视为有依据，例如其额外“自治团成员之一”仍需单独审查。
- 古米训练题：第三下用尽全力、第四下挥不出在 API 第 8 条正文中，并非核心事实全部漏传。
- 凛冬领袖题：“荡平周边社团”有 API 第 3/4 条百科及第 9 条官方档案支持，不能仅因简短参考答案未写就判为凭空编造。

## 3. 需纠正的既有比较

本次中文 4B 的实际 `lora_path` 是 `teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2`，不是先前英文测试的 `asa_v55_a03_20260911/model`。
两侧输入格式、证据处理、检索轮次也不同。因此不能据此声称“同 prompt 下新训练 4B 与 API 的严格准确率差一题”，也不能仅凭这次拒答归因训练质量或模型能力。

## 可核验文件指纹

- 4B JSONL（本地与远端一致）：`4ea4f872ca1c5e6fd25f4d8320ea9c14b635cb8a5b282eb0dc418379f5ea9790`
- DeepSeek JSONL：`7fb8e03f75a59c1358d377c8a8a2ee57076bc257942111bd0f8dfc7a27fe54d1`
- DeepSeek 脚本：`54d4682e10399d2fe27a463c93c44286ae309aefc15f8d19fa7f521d12679b1d`
- 本次读取的远端 `cpu_pipeline.py`：`2eb4fea9d73730878aa3d1e854171e69ded8f5d6b6753399c63319671f29de95`

建议后续修复先覆盖证据候选全文，再精排/按预算裁剪；保存精炼后证据和实际 prompt，使用同一份可核验输入做少量配对测试。本次仅诊断，尚未实施。
