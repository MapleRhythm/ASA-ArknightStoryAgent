# ASA：fixed58 审核归档、错误引用修复撤回与后续优化 v26

241 为唯一规范源，248 为训练镜像。本地仅转运和分析。MiniRAG 保持禁用。
文件名延续 20260909 检查点系列；机器回执使用实际 UTC 时间。

## 结论

撤回此前刚应用的“唯一引文自动重绑 E 编号”和配套的末尾标点放宽。
原始输出反事实中唯一新增放行的是林地部族迁徙题，但它把萨米题目答成了萨尔贡另一族群的遭遇。
整体 GLM 审核把它判为 complete 是一个主体归属假阳性，不能报告“净增 1 条正确”。
此前该表述在此明确纠正。

保留两项可复现修复：批处理部分失败必须非零退出；训练 reward 脚本撤回到旧版的错误改动恢复到 241 HEAD。
没有切默认模型、训练、重启服务、提交/push 或覆盖历史标签。

## 1. 本轮 fixed58 的实际状态

基线为无 MiniRAG 的新检索/生成运行：

`${REMOTE_ASA_ROOT}/rank_v8_coverage/runs/e2e_no_minirag_fixed58_20260909.v23v3`

这是 58 个去重开发问题；历史 fixed79 是同一批 58 问的 79 个证据状态，不是 79 道独立全链路题。

| 分类 | 原自动审核 | 单独加入 031 助手复核的展示汇总 |
|---|---:|---:|
| 完整支持，暂判 | 13 | 13 |
| 部分支持，暂判 | 15 | 15 |
| 不支持/含错误，暂判 | 14 | 14 |
| 无实质回答 | 15 | 16 |
| 自动审核错误 | 1 | 仍保留原错误，另列复核来源 |

原始自动文件保留 57 成功、1 失败。混合来源汇总为 57 条模型判断加 1 条助手源文判断。
暂判完整 13/58＝22.4%；完整或部分 28/58＝48.3%。后者不是完整回答正确率。
证据充分暂判 51 条，其中完整13、部分14、不支持12、无回答12。
**该 51/58 不是独立测出的 RAG 召回率，也不是人工 gold；这些数字不认证线上正确率。**

031 实际只显示“现有检索证据不足以给出可校验的回答。”。
完整 E1–E7 已读；E4/E6/E7 明确支持恩希欧迪斯承诺的未来与研究舞台、共同建设谢拉格的理想和诺希斯的信任。
助手复核记为 no_answer / sufficient / inappropriate。自动 schema 失败不被手改成成功。

本检查点完整源文复核限于 011 和 031；未把其余自动 complete/unsupported 全部认证。
基线可见文本是按保存的 final evidence 与 packing 长度重建；旧 E2E 未记录逐请求完整 prompt token IDs，
因此这里不把“同来源且长度匹配”进一步夸大为原始输入逐 token 哈希证明。

## 2. 为什么撤回重绑定

011 原题：“林地部族决定迁徙的原因是什么？”

- 历史记录 `159596800f42e7e0-conclusion_generation-01` 的来源和回答落在萨米剧情。
- relabel `relabel-8b150fd76f7fbfa91bb916e4` 用林地萨满 E2 的不祥征兆和萨米意志解释。
- 本轮 E2 是 `activities/act15mini/level_act15mini_st03#chunk-0007`，仍有完整相关原文。
- 新放行答案使用 E1 的纳卢塔巴提河、风沙、王酋、采矿经历；对应
  `obt/sandboxperm/sandbox_1/sandbox_1_entry#chunk-0002`，没有证据证明是所问族群。

早期 teacher 的亡寒风雪解释与 relabel 的林地萨满解释也不应合成无争议人审标准；
短题干本身存在消歧缺口。但这不能授权生成器或裁判把主体改成萨尔贡。
候选盲读与候选答案审核应分开，引用字面存在也必须与“是谁的事实”分开。

撤回三份 release 源码，精确恢复到该实验应用前的 SHA，而非回退整个 dirty 工作树：

| 文件 | 撤回后 SHA256 |
|---|---|
| `quote_checks.py` | `2108ce590be98361c896e82a7e41f86fa007cef14f5521521be634f648ec07d2` |
| `quote_match_utils.py` | `d17cf94bc34089fd5fd49a1dde205e4779ade8476088f7ffa8366fb5507132a1` |
| `quote_validation.py` | `a1a8b719dc6a7892e6a060b87f8169291c576826831ba0404c03d61c4fcdbb7c` |

以上文件均位于 `release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/grounding/`。
被撤回候选及旧测试完整备份于：

`outputs/rollback_backups/20260909_quote_rebind_withdrawn_v26`

活动测试改为验证不自动重绑、不放宽标点；旧实验测试与结果继续留存。
撤回后重放 56 条可严格解析的末轮输出，0 条动作改变，回到基线；012、051 两条不强修解析，仍列 not_replayed。
这是 quote checker 回放，不是新生成，也不证明所有前轮完全不变。

## 3. 保留的修复与测试证据

`release/ASA-ArknightStoryAgent/scripts/run_cpu_inference.py` 保留错误计数与退出 1。
新增 6 项子进程测试运行真实参数解析和 `main()`，仅 mock 模型、检索器和索引依赖：
全成功退出0；混合/全失败退出1；失败后的成功题继续执行；两种展示模式的 JSONL 都不漏行。
这验证了 CLI 错误传播，不认证真实模型服务健康。

根 `scripts/glm_exx_semantic_reward.py` 此前被完整退回历史 v2，缺失 strict JSON、precision profile、
semantic gate 等依赖；它的旧字节精确等于历史提交，mtime 为 2026-09-06 02:09:12。
已单文件恢复 241 HEAD，SHA 为
`82e7d2931a958a252a085284032c172bcf6276c4a5b66d8f298c8523fa4d48bc`。
旧内容及 diff 保存在 `outputs/rollback_backups/20260909_glm_exx_semantic_reward_v2_reversion/`。
这证明版本退化和测试兼容性问题，不单凭 mtime 断言是哪一个模型修改，更不声称下游训练已改善。

本次正确配置的独立进程验证：

- release：286/286，0 失败/错误/跳过。
- 根仓库：248/248，0 失败/错误/跳过。
- `git diff --check` 通过。

计数不与历史重复测试累加。release 计数还受 MiniRAG retriever 参数化路径数影响。
首次测试遗漏 PYTHONPATH，随后一次缺 MiniRAG 测试所需环境变量，两次均在收集阶段失败；
未将其当作产品 bug 或抹去。`verify_canonical.py` 记录完整环境配置选择与分进程执行方式，
它检查 15 份运行配置及两份 retriever 实现；最终 XML、日志和 SHA 都已保存。

## 4. 下一项优化，不重复已否决方案

已读规范 v6、v12–v16、v19、v21，以及已保存的单 QID 四状态结果。
现有 quote-SFT、LoRA 直接开关、原生8B、删推理提示、直接放宽quote、去重优先级、
QID替代引文等均不能凭机械放行率晋级。

新实验限定为“先独立定位问题对应事件，再让原 reader 作答”：
定位阶段只接收问题和完整可见证据，不接收旧答案、旧审核或正确答案。
只将它选出的原 E 块原样交给 reader，不传播模型写的理由，不重排/重编号/截断块。
保留标准 reader 对照、相同 LoRA、成对 seed、全部失败和上下文溢出。
选择 8 个明确登记的开发诊断问题，检验主体误配、转折因果、弃答和正对照保留；不是新盲测集。
先做隔离实验，不改线上路由；若减少引用错误却仍答错人，仍不得晋级。

## 工件

规范归档目录：

`outputs/experiments/quote_rebind_withdrawal_20260909.v26/`

关键文件：`checkpoint_v26.json`、`source_review_v26.json`、
`cpu_verification_v2.json`、`replay_after_withdrawal_v1.json`、
`verify_canonical.py`、`rollback_rebinding.sh`。
自动审核原文件、候选正判和反例回放分别保存，不覆盖原标签、不生成训练正例。
