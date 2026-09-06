# 多轮调度的 tool-call 化设计（2026-09-06）

## 结论

多轮调度值得改成“隐藏式、结构化的 tool call”，但不能把 BM25、Faiss、
图扩展和 reranker 分别暴露给模型，也不能让模型自行循环。tool call 主要
解决协议解析、预算控制、可观测性和重试问题；它本身不会提高证据召回或事实
蕴含能力。

当前实现已经是半结构化协议：模型输出 `next_action=retrieve_more` 和
`follow_up_hypothesis`，后端再执行下一轮。改造应保持这一协议兼容，先以
opt-in 方式验证，不能直接替换生产。

## 推荐控制面

模型每轮只选择一个动作：

```json
{"name":"search_more","arguments":{
  "missing_requirements":["问题仍缺少的实体/关系/时间/原因"],
  "queries":["候选查询"],
  "scope":{"entities":["已在问题或证据中出现的实体"],"storyline":""},
  "expected_answer_type":"fact"
}}
```

终止动作：

```json
{"name":"answer","arguments":{"supported_facts":[
  {"fact":"原子事实","evidence_ids":["E2","E7"]}
]}}
```

```json
{"name":"abstain","arguments":{"reason_code":"insufficient_evidence"}}
```

`name` 和 `arguments` 只存在于内部 trace；用户最终仍只看到自然语言答案及
`Exx` 标记。不要输出思维链。

## 后端必须拥有的权限

- 最大检索轮数固定为 2（特殊多跳问题可单独实验 3，但有硬超时）。
- 每轮查询去重、长度限制、问题/实体相关性检查由后端执行。
- `search_more` 只调用一个高层的 hybrid-retrieve 服务，内部并行执行
  sparse、dense、章节隔离、图扩展和 reranker。
- 新增证据数、证据集合覆盖和延迟由后端测量，不能由模型自报。
- 没有新证据、超时或查询全部重复时，控制器强制 `abstain`/最终受限回答。
- `answer` 经过 schema、E-ID、fact-level support 和 set-level coverage 校验。

## 保留 fact 间关系

tool call 不能只检查每条 fact。回答动作内部应保留可选关系：

```json
{"name":"answer","arguments":{
  "supported_facts":[
    {"fact":"事实A","evidence_ids":["E2"]},
    {"fact":"事实B","evidence_ids":["E7"]}
  ],
  "relations":[
    {"left_fact_index":0,"right_fact_index":1,
     "type":"causal","evidence_ids":["E2","E7"]}
  ]
}}
```

后端或离线 verifier 同时检查：

1. 每个 fact 是否被自己引用的证据完整支持；
2. 事实集合是否覆盖问题的核心信息需求；
3. 因果、时间、身份、指代关系是否有两端证据联合支持；
4. 是否夹带“真实但与问题无关”的事实。

## 训练和评测顺序

1. 先将现有 JSON 输出映射为虚拟 tool trace，验证控制逻辑，不增加模型调用。
2. 用 family-held-out 问题训练/评测 tool-call 格式；训练集和盲测题不得按
   原始 ID 或近重复问题重合。
3. SFT 只使用经过完整 set-level 审计的 unchanged answer，以及动作合适的
   `retrieve_more`/`abstain` 样本。删除坏 fact 后得到的半答案必须二次审计。
4. 绑定指标稳定后才做 RLVR；GLM 只离线标注，线上使用蒸馏后的本地 verifier。

验收指标必须同时满足：独立盲测中
`unsupported + contradicted <= 20%`、回答级关键主张风险不超过 20%、
不能靠大量弃答达标，且 warm p95 GPU 延迟不超过 20 秒。

