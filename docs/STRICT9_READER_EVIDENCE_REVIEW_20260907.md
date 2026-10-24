# ASA strict9 reader × evidence 原文复核

日期：2026-09-07

> 2026-09-08 更正：实际 reviewer 为
> `codex-primary-source-review-20260907`，即 Codex 对照原文复核。本文历史
> “人工标注/人工重校”措辞不代表独立人类审核。既有 JSONL 不改写，保留原始
> 内容指纹；此 rubric 仅用于开发诊断，正式测试仍需独立验收。

## 结论

本次逐题阅读
`family_holdout_strict9.snapshots_v3.jsonl` 中的原问题、gold facts、
gold 文本与实际 prompt 候选正文，并生成：

- `data/eval/strict9_reader_evidence_reviewed_20260907.jsonl`

9 道题均可批准用于 **P0 流程 smoke**，没有因问题歧义或证据不足而排除。
但 strict9 只有 9 个问题族，不能替代计划中的约 60 题正式 reader ×
evidence 诊断，也不能据此宣称新的系统回答正确率。

## 对旧判断的纠正

1. 迷迭香/阿米娅题不应排除。旧摘要只读到“相像之处”，但
   `obt/main/level_main_07-11_beg#chunk-0014` 后半段紧接着明确写出：
   阿米娅从不想与人为敌，迷迭香则从不放弃任务、作为歼灭战专员保护同伴。
   该单段已经完整回答“态度有何不同”。
2. 莫斯提马题的旧 gold 绑定不适合直接作为最小证据。旧
   `st03#chunk-0010` 交代守护铳事件，却没有命名真实身份；snapshot 实际
   prompt 中的 `st03#chunk-0009` 明确称她为“堕天使”。人工标注采用
   `07_end#chunk-0001 + st03#chunk-0009`。
3. 老天师题无需萌百背景段。正文 `07_end#chunk-0005` 已说明樵夫是徒辈、
   神志到极限及“不想再杀”，`chunk-0006` 则提供她无情消灭邪祟的对照。
4. 女妖题只需 `05_end#chunk-0006` 就能确认寻找对象是娜斯提且二人为母女；
   其余旧 gold 是人物状态、姓名和结局背景，不是回答该问题的最小必要项。

## 最小证据规模

| 问题 | 最小文档数 | 复核要点 |
| --- | ---: | --- |
| 雷吉斯/林吉斯 | 1 | 同段给出分歧、自我流放、死后返乡与团聚目的 |
| 莫斯提马 | 2 | 一段解释“普通信使”自称，一段明确“堕天使” |
| 带剑女子 | 1 | 同段明确她就是渡魂剑、由器物化伥 |
| 老天师 | 2 | 徒辈保护与邪祟处决形成题目要求的对照 |
| 迷迭香/阿米娅 | 1 | 同段直接比较二者战斗意愿 |
| 杰克逊 | 2 | 分别覆盖特里蒙与莱茵生命/科学政治姿态 |
| 杰米 | 1 | 同段完整给出疾病、污名和制度排斥 |
| 虚弱女妖 | 1 | 直接点名娜斯提并确认母女关系 |
| 芙蕾达 | 2 | 镇民/威尔分歧与本人妥协动机各一段 |

每题另选 1 个真实语料 hard distractor。干扰段优先来自同剧情、同角色或
高度相似主题，但不属于任何接受的支持组；具体理由记录在 JSONL 的
`notes` 字段。

## 使用边界

- 这份标注是人工重校后的 reader 诊断 rubric，不是对旧 GLM gold 的盲目复制。
- `minimal_sufficient` 只回答当前问题，不收录完整人物背景。
- `sufficient_with_distractors` 用于观察 reader 抗干扰退化。
- `actual_prompt` 保留冻结 snapshot 的 outer document 顺序；在 evidence-ID
  runtime 中，模型看到的是各 outer document 的直接正文，不把
  `evidence_chain_text` 偷渡为可见证据。
- 后续 semantic audit 未跑成功的行必须留在全行分母中，不能视为成功。
