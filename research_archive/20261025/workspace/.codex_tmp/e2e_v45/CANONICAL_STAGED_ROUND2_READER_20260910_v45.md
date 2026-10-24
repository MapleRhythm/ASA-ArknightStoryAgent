# v45：完成既有首轮输出的第二轮 reader

规范目录：`outputs/experiments/staged_round2_reader_20260910.v45`。
文件日期按北京时间 2026-09-10；JSON 回执为 UTC 2026-09-09，vLLM 日志为本地时区。

## 已完成的范围

保留 v41 全部 24 个状态：10 个首轮终点原样带过，14 个 v44 第二轮检索状态
调用真实 canonical `generate_conclusion(round=2)`，包含其原有最终 direct fallback。
没有重新采样首轮、重做 planner/检索、自动修正 E-ID 或放宽校验。

原 4B SODA LoRA、legacy `qwen3_nothink` 模板；每次实际输出预算 2048 token，
temperature 0.1 / top_p 0.8 / repetition 1.05；证据 12000 字符，上下文 12288。
最长预登记输入 8460 token，无上下文溢出。14 次 CPU 合成弃答协议检查同时核对了
正常生成和最终回退路径（不计为模型调用）。

实际新增 23 次模型调用：14 次正常生成、9 次系统原有最终回退。全部 stop；
24 个终点完整保存。14 次 canonical CPU 重放逐字段一致，另一次观测式 CPU 重放
保留了被 fallback 替换之前的首个结论，终点仍逐字段一致。

preflight SHA：
`271a430b5ccfd0bf36fb9048f167b9a02b3a6ed05785f71a9bfb8ef1672305f7`。

## 结果不是正确率

| 条件 | 最终 direct | 最终 abstain | 本阶段新调用 |
| --- | ---: | ---: | ---: |
| control | 7 | 5 | 14 |
| candidate | 9 | 3 | 9 |

这是六道开发题、两个 seed、两个条件的分阶段续跑，不是 24 道独立题，也不是
重新抽样的独立 E2E benchmark。相同 seed 不保证相同生成。

- 007：四个首轮终点未动；control seed1 的“她母亲娜斯提”关系反转仍展示。
- 010：candidate 两个首轮“宗师是令小姐兄长”的窄答案原样保留。control 第二轮
  **已看见 E3 的正确原文**，但 seed0 绑成 E7、seed1 绑成不存在的 E8，最终仍弃答。
  seed1 fallback 还错写一处说话者。不是第二轮召回失败，不自动重绑。
- 017：seed0 两组输入完全相同。control 有列车事故核心，但一条 quote 漏掉
  “意外”而被拦，fallback 弃答。candidate 展示列车事故句后，追加了属于茱安娜
  父亲的治病打猎/车站经历，错误归到初雪父亲。seed1 两组继承的雪绒救援混淆未动。
  因此 candidate 多一个 direct 不能当改进。
- 023：四次第二轮均正确引用 E1，展示“花心思/师父心软、开窍”核心；两个配对
  seed 内 raw 一致，四个最终展示一致。附加“证明厨艺进步”仍是解释性表述，
  不认证所有额外原子断言；也不是 candidate 特有收益。
- 046：四个最终动作仍弃答。control 的 primary/fallback 都错指云青萍；
  candidate primary 指莫佚和看争斗的动机，但有“动机”等旧规则拒绝，
  seed1 primary 还出现 E2 引用错绑及“制造”问题。E4 的指使关系来自剧情摘要，
  不是对话。candidate seed0 fallback 又把“唤出幻象”的事件说成动机。
  **不放松 `动机` 校验。**
- 049：control seed0 是唯一从弃答转为 direct 的最终回退，但仍把“特雷弗制造了它”
  当成已证明关系。引用仅说“不是本人/被制造的工具”，没有给出制造者。E2 确有
  复制品、原始记忆及回答克丽斯腾问题的文本，不能把所有信息分享都判错。
  E1 上下文谈能源，“血/心脏”意象也不能直接认证成生理血液捐献。
  candidate seed0 fallback JSON 破损，进入 unparsed fallback 后被
  `missing_supported_facts` 拦住。两组 seed1 旧创造者断言原样带过。

9 次最终回退中，8 次仍弃答，1 次转为 direct；该唯一转变仍含未证明关系。
14 次第二轮正常生成中有 5 次 direct，无最终回退。以上都不是语义正确率。

## 验收与资源

prepare/run/check/primary-audit/close 均 exit 0。全部 23 个新 raw、24 个终点已读；
28 个可见源文绑定、24 个输出绑定已在规范仓库校验，`checkpoint_v45.json` 已归档。
十个旧终点以 v41 检查点和原样性校验绑定，不改历史标签。

GPU2 在新锁内完成新的超过 60 秒连续空闲核验；运行后释放，仅 GPU0 的无关进程
仍存在，未干预。本次日志保留 process-group shutdown warning，无 ERROR/Traceback 行。

结论：不推广 routing recovery，不声称正确率提升。继续只读诊断人物归属、
关系方向和回退诊断完整性；不恢复 E-ID 重绑，不更换默认模型，不训练或重启，
未提交/push。本阶段未增生产源码；更早 v33 排版改动仍保留。
