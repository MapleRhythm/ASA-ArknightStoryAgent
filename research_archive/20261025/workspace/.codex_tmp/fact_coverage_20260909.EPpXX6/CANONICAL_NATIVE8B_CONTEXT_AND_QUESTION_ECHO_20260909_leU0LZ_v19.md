# ASA：原生8B、去推理元数据对照与题干复读修复（v19）

北京时间2026-09-09。241为唯一规范源，248为训练镜像，本地仅传输缓存。
接续v18，旧报告、生成、审核及失败记录不覆盖。真实回答正确率目标仍进行中。

## 结论

两组各58次新生成全部完成、全部正常stop，无输入裁切或输出预算缩减。
原生8B和删除hypothesis/minirag_hints两种候选都没有晋级；机械放行不是语义正确。
去提示有女妖找娜斯提、夜刀明确矿石病等选定样例改善，也有新增答非所问和格式退化。
只上线了可复现的窄修复：不再把完整显式问句当作事实展示。
本阶段未训练、切换默认模型/提示、重启服务、提交或推送，也未改旧数据/标签。

## 1. 两组真实生成和最终审核

所有统计分母各为同一开发集的58个冻结证据状态，不是新holdout或新RAG检索。
基线复用旧原4B结果，并非此次重新交错运行；不可将它们与fixed79混为一个分母。

| 指标 | 原4B提示 | 原生Qwen3-8B | 4B去hypothesis和minirag_hints（修复前） |
|---|---:|---:|---:|
| 状态 | 58 | 58 | 58 |
| 规范化后严格JSON | 57 | 58 | 49 |
| grounded direct | 32 | 9 | 31 |
| 语义supported_complete | 12 | 3 | 10 |
| 语义supported_partial | 8 | 1 | 8 |
| 语义unsupported | 9 | 4 | 9 |
| direct但无实质回答 | 0 | 0 | 2 |
| 语义审核失败 | 3 | 1 | 2 |
| 非回答动作 | 25 | 49 | 26 |
| pipeline error | 1 | 0 | 1 |

其中“规范化后严格JSON”和“pipeline error”不是同一检查；解析恢复可能处理部分非严格输出。
所有语义分类只是冻结协议的模型暂判，不是人工gold或认证正确率。旧裁判已有假阳性，不能据12、3、10算真实提升。

### 原生8B条件

本地已存在的`model/qwen3-8b`，原生chat template、enable_thinking=False、无LoRA、无强制左花括号前缀。
问题、system/user、完整证据、seed与原提示保持一致；模型、tokenizer和原生模板同时变化，是部署候选对照，不是纯权重消融。
context12288/output2048；最长输入8817；temperature0.1/top_p0.8/repeat1.05。
GPU利用上限0.90、bf16/eager/batch4096为实验配置；线上4B配置未改。
连续60秒重新确认4090空闲后启动。实际engine构造和每请求均核验无LoRA；静态describe_runtime中的旧LoRA描述不能冒充实际使用。

校验后9 direct /37 retrieve /12 abstain，无pipeline error。
盲审共82个去重输入：原4B的64个旧请求完整复用（包括失败），18新请求15成功3失败，无重试/手修。
3个新失败分别为无效JSON、语义摘句不逐字、覆盖摘句不逐字，均从原响应重现。
全部9条展示答案已对完整可见源文复核并保存。确认存在把银行经理指责归到里昂自责、
把随从艾露猫态度归到夜刀、把侦探故事血衣疑问连接到题干未出现的定罪权力讨论等错误。
其中有内容有据但短引用覆盖不全的情况；两者分开记录，未一概说成虚构。

### 去推理元数据条件

同原4B58题的完整证据、system/schema、真实LoRA、tokenizer、seed、采样和预算。
只删除生成user中的hypothesis整行和合成minirag_hints行；上下游hypothesis仍在，因此不是全链路污染清洗。
同时删两个字段，是组合干预，不能归因到其中某一个字段。
另一次新60秒空闲窗口后启动；最长输入7955；逐请求LoRA/echo均为1。
生成后资产、源码、token IDs及预算后验成功，之后才应用题干复读修复。

审核共124个去重输入：64旧请求原样复用，60新请求57成功3失败；无重试或手工修补。
两个新失败是把中文引号改成ASCII引号，违反逐字摘句要求。
第三个回传model仍为glm-5.3，finish_reason=length，completion16384、reasoning15997，
正文在JSON字符串中途截断；不是已证实模型路由变化，也不是4B生成被截断。
审核后验确认请求、rubric、原响应、来源和汇总一致。

## 2. 题干复读缺陷修复

拜松问题的新输出将整句问题写进supported_facts；旧renderer因此最终展示同一个问题，动作却是answer_directly。
这不是截断，也不能通过恢复未经核验的raw答案修复。

规范改动：

- `release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/grounding/quote_validation.py`
- 新测试`release/ASA-ArknightStoryAgent/tests/test_question_echo_grounding.py`

仅当原问题以英文/中文问号结束，且去空白和末尾问号/句点后与整个fact完全相同，才拒绝该fact。
supported和inferred两组都检查；不做一般语义匹配，不误杀嵌套引用问句、普通陈述或标题。
仅final_answer复读而facts有效时，走现有checked-fact renderer；不放宽quote门禁。

正确装配的旧代码基线23项为16失败/7通过；候选及241应用后348/348通过。
248隔离候选和应用后各192/192 CPU子集通过，不把它们累加到348。
248完整运行时Python及四个数据脚本与241核对一致，未新增依赖，也未认证GPU训练环境。
两边旧文件备份和逐文件SHA回执均保存，HEAD仍为`0d594a3c9fd641fd8d7f798c8ca6b07db6c7020b`。

重放fixed79、旧116、原生8B58均无grounded变化；去提示58仅拜松1条direct→retrieve_more：
修复后为30 direct /21 retrieve /6 abstain /1 pipeline error。
这是一条假放行被移除，不是新增一个正确答案。
原31direct审核保持原文件；另存修复后30direct投影，剩余裁判输入逐字相同，0新调用。
投影的语义分布仍为10 complete/8 partial/9 unsupported/2 audit_error，
另有1个direct但no_answer、27个非回答动作和1个pipeline error。

## 3. 独立源文复核边界

本阶段补读林地迁徙、娜仁图亚、恺撒、杰克逊、天火的完整可见源文，另存复核，不覆盖v18或旧标签。
林地迁徙不能把迁徙后侵略杀人当作先前迁徙原因；天火不能把主动随绑匪去工厂调查说成防止被抓去工厂。
娜仁图亚“赚钱”明确，但“欠债导致此次绑架”是额外解释桥；两版同类内容不能被裁判不一致地当作可靠优劣。

8B9条已全源文复核。去提示58对展示文本均已读，但并未完成58题全部源文认证：
女妖找娜斯提、夜刀点名矿石病、里昂删掉“不义之财”为选定的有据改善；
夏栎问题却答米克尔战友是明确目标不匹配，不能据此说另一救援事件全局不存在。
这些是独立助手复核，不是人工gold，未发送给生成器或审核器，不授予训练资格。

## 4. 本阶段工具错误也保留

若干自写脚本曾向Path-only SHA helper传入`__file__`字符串，预检/后验失败；已改为Path对象，
旧脚本备份及3项静态回归保留，与7项context测试共10项通过。未因此重跑生成。
一次启动因矩阵尚未构建成功而退出，没有GPU加载，之后构建完成才重新启动。

题干复读测试工厂第一版漏ConclusionResult三个必填字段，23项全部装配失败。
该XML和脚本备份保留，不能作为缺陷基线；修正工厂后v2才得到16失败/7通过，正式348通过。
未把过程改写成“从头到尾全成功”。

## 5. 当前测量边界与下一步

fixed79仍是58题的79个历史证据状态，39 direct/31 retrieve/9 abstain。
旧自动标签19 complete/11 partial/8 unsupported/40 no_answer/1 error已有假阳性，不是19/79真实正确率。
本阶段没有新RAG召回或完整多轮E2E正确率，也没有证明线上服务已加载工作区修复。

不继续把简单加提示、去提示、8B或放宽引用当作默认改进。
下一项离线实验把“模型抄引文”拆成“选择明确源文片段ID→程序原样取文→原quote门禁”，
保留完整可见源文和全部失败分母，检查能否减少抄写错误，同时独立检查主客体、因果、否定和覆盖。
该方案尚未验证，不先接入生产，更不把增加放行直接记作正确率提升。

规范工件目录：

```text
${REMOTE_ASA_ROOT}/rank_v8_coverage/experiments/canonical_quote_eval_20260908.irPugk/fact_coverage_v1.leU0LZ
```

关键文件：`qwen3_8b_generation_postflight_v1.json`、`qwen3_8b_blind_audits_postflight_v1.json`、
`without_reasoning_hints58_generation_postflight_v1.json`、`without_reasoning_hints58_blind_audits_postflight_v1.json`、
`without_reasoning_hints58_after_echo_audit_projection_v1.json`、`question_echo_applied_verified_v1.json`、
`mirror248_question_echo_delta/mirror_applied_verified_v1.json`、`checkpoint_v19_state.json`。
