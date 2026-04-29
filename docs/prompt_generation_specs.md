# Prompt Generation Specs

本文档记录当前项目中用于数据生成与运行时推理对齐的三套 prompt 规格。以当前代码实现为准：

- 运行时实现：[src/goldenglow/inference/cpu_pipeline.py](/home/zhb/ASA-ArknightStoryAgent/src/goldenglow/inference/cpu_pipeline.py)
- 数据生成实现：[scripts/generate_prompt_supplement_from_teacher.py](/home/zhb/ASA-ArknightStoryAgent/scripts/generate_prompt_supplement_from_teacher.py)

当前流程只区分三种调用：

1. 初始假设文档生成
2. 多轮补充假设文档生成
3. 结论生成

不再通过在 prompt 中显式写 `task_type:` 字段来区分，而是通过三套不同 prompt 模板区分。

## 1. 初始假设文档生成

### 作用

把用户原问题改写成第一轮检索所需的结构化假设文档。

### 输入信息

- 用户问题
- 多轮上下文

### 字段含义

- `question`
  用户当前原问题，不要改写成别的问题。
- `intent`
  问题的语义类型，只表示问题类型，不表示流程状态。
  当前允许值：
  - `plot_fact`
  - `plot_reasoning`
  - `timeline`
  - `character_relation`
  - `event_summary`
  - `compare`
  - `persona_chat`
  - `out_of_scope`
- `entities`
  当前问题里最核心、最值得用于检索的实体，一般是角色、组织、地点、事件名。
- `keywords`
  比 `entities` 更宽的检索词，可以包含原词、同义改写、关系词、短语化检索扩展。
- `expected_answer_type`
  期望最终答案的形态，例如事实问答、身份关系、原因/动机、时间线、过程解释。
- `dialogue_context`
  多轮对话上下文，用于补指代和追问背景；如果没有多轮上下文，可以省略，系统会按空字符串处理。

### 返回格式

```json
{
  "question": "string",
  "intent": "plot_fact | plot_reasoning | timeline | character_relation | event_summary | compare | persona_chat | out_of_scope",
  "entities": ["string"],
  "keywords": ["string"],
  "expected_answer_type": "string",
  "dialogue_context": "string"
}
```

### 关键约束

- 只输出单个 JSON 对象
- 不输出解释，不输出思维过程
- 不允许额外字段
- 不要把最终答案写死
- `dialogue_context` 对初始 hypothesis 可省略

## 2. 多轮补充假设文档生成

### 作用

在当前轮检索证据不足时，生成下一轮检索要用的补充 hypothesis。

### 输入信息

- 用户原问题
- 当前检索轮次 / 最大轮次
- 当前假设文档
- 上一轮结论生成结果
- 历史生成结果
- 历史检索上下文
- 当前证据
- 当前未解点

### 字段含义

- `question`
  用户当前原问题，保持不变。
- `entities`
  在上一轮实体基础上，补充本轮证据里出现的关键桥接对象。
- `keywords`
  面向下一轮检索的缩小范围关键词，优先加入关系词、称谓、桥接短语。
- `expected_answer_type`
  继续沿用当前问题所需的答案形态，例如身份关系、原因/动机、时间线、过程解释。
- `dialogue_context`
  多轮对话上下文；通常沿用上一轮上下文，没有则可为空字符串。

### 不再输出的字段

- `intent`
  不在 assistant 输出中出现，默认继承上一轮 hypothesis 的 `intent`。

### 返回格式

```json
{
  "question": "string",
  "entities": ["string"],
  "keywords": ["string"],
  "expected_answer_type": "string",
  "dialogue_context": "string"
}
```

### 关键约束

- 只输出单个 JSON 对象
- 不输出解释，不输出思维过程
- 不允许额外字段
- 不直接回答用户问题
- 必须体现“缩小范围后的二次检索线索”

## 3. 结论生成

### 作用

基于当前证据生成当前阶段结论，并判断下一步动作：

- 直接回答
- 继续检索
- 澄清用户
- 放弃回答

### 输入信息

- 用户原问题
- 当前检索轮次 / 最大轮次
- 当前假设文档
- 历史生成结果
- 历史检索上下文
- 当前证据

### 字段含义

- `question`
  用户当前原问题。
- `next_action`
  当前证据下的下一步动作。
  只允许：
  - `answer_directly`
  - `retrieve_more`
  - `clarify_user`
  - `abstain`
- `answer`
  当前阶段结论文本。
  - `answer_directly` 或 `abstain` 时必须非空
  - `retrieve_more` 时必须为空字符串
  - `clarify_user` 时可为空
- `missing_slots`
  当前证据还缺哪些具体可检索的信息缺口，主要在 `retrieve_more` 时使用。
- `clarification_question`
  当问题本身有歧义时，向用户发出的澄清问题；仅 `clarify_user` 时必须非空。

### 返回格式

```json
{
  "question": "string",
  "next_action": "answer_directly | retrieve_more | clarify_user | abstain",
  "answer": "string",
  "missing_slots": ["string"],
  "clarification_question": "string"
}
```

### 关键约束

- 只输出单个 JSON 对象
- 不输出解释，不输出思维过程
- 不允许额外字段
- `retrieve_more` 时：
  - `answer` 必须为空字符串
  - `missing_slots` 必须非空且具体可检索
- `clarify_user` 时：
  - `clarification_question` 必须非空
- `answer_directly` / `abstain` 时：
  - `answer` 必须非空

## 4. 数据生成时的外层格式

teacher 数据生成脚本不会直接返回单条 assistant JSON，而是返回：

```json
{
  "samples": [
    {
      "id": "string",
      "task_type": "user_question_hypothesis_generation | follow_up_hypothesis_generation | conclusion_generation",
      "messages": [
        {
          "role": "system",
          "content": "string"
        },
        {
          "role": "user",
          "content": "string"
        },
        {
          "role": "assistant",
          "content": "stringified-json"
        }
      ],
      "meta": {
        "grounded": true,
        "difficulty": "easy|medium|hard",
        "notes": "string",
        "source_story_ids": ["string"],
        "source_stage_codes": ["string"],
        "source_activity_names": ["string"]
      }
    }
  ]
}
```

注意：

- `task_type` 只用于数据记录和清洗，不再作为 prompt 文本中的显式字段
- `assistant.content` 必须是单个 JSON 对象的字符串

## 5. 当前设计结论

- `intent` 只属于“初始假设文档”
- `follow_up hypothesis` 不再重新生成 `intent`
- `dialogue_context` 对初始 hypothesis 非强制
- 多轮历史通过 `retrieval_trace` 保存，供后续 prompt 使用
