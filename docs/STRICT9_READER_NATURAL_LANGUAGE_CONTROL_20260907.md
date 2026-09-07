# ASA strict9 reader 自然语言短答对照

日期：2026-09-07

## 目的

上一轮 strict9 reader × evidence smoke 在人工最小充分证据上，Qwen3.5-4B
base 与 clean-SFT 的 Exx 完整回答率均为 4/9。为了判断失败主要来自基础
阅读能力还是 Exx JSON/action/引用协议，本轮保持同一问题、同一证据正文和
顺序不变，只去掉 Exx schema、动作选择和 E-ID 绑定要求，要求模型直接输出
简洁自然语言答案。

这是协议诊断，不是新的模型排行榜，也不是线上系统正确率。

## 可追溯性

- 来源矩阵 SHA：
  `1886f257e4e289b8bd71a7282eceb89190bff8be745d3319b5253e07051580f0`
- 人工 annotation SHA：
  `c41f91dd098ae9cd344b78327d14f53cfefb0939051d23569fa7e7e001e13af4`
- 自然语言 control SHA：
  `585db8a51a84e31c8c4d2f88bdac45ec1685664e0fcbaf347b5a13278f0d60a5`
- base prediction SHA：
  `15f82122b235486be98a571993e2ddeadee109ac76af1cde6d09862381082b4b`
- clean-SFT prediction SHA：
  `43c832faaa0742552278b2684256eef31ccb97616fd96a0b992b9e3edaa76da7`
- base 模型资产聚合 SHA：
  `4dcba47639cd5ccef2436373743b1f122d11d08c85aa63c34fc8ce37aba1ff2b`
- clean-SFT adapter SHA：
  `75c3f1e0c2946c77537d81266b408913ce51bd6a41563ee2ca884e3a76c3c41c`

远端目录：

`/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/experiments/reader_natural_language_control_strict9_20260907_v1`

两种 reader 都在 241 RTX 4090、Transformers/PEFT、temperature 0、
non-thinking canonical template、`max_new_tokens=768` 下运行；均为 9/9
成功、0 generation truncation。人工答案要点只保存在 diagnostic 中用于
事后评分，没有进入 system/user prompt。

最初两次启动分别因 `train` 环境不认识 Qwen3.5、`reasoning` 环境缺少
`accelerate` 而在加载前退出，没有产生或覆盖 prediction。最终使用上一轮
已验证的 `rlvr_20260828` 环境。

## 人工结果

| Reader / 协议 | complete | partial | incorrect |
| --- | ---: | ---: | ---: |
| 4B base / Exx minimal | 4/9 | 3/9 | 2/9 |
| 4B base / 自然语言 minimal | 3/9 | 4/9 | 2/9 |
| clean-SFT / Exx minimal | 4/9 | 2/9 | 3/9 |
| clean-SFT / 自然语言 minimal | 5/9 | 2/9 | 2/9 |

Exx 的 partial/incorrect 行按上一轮人工修正后的 set support 归并：
`complete` 保持 complete，`partial` 为 partial，`none/contradicted` 为
incorrect；协议不合格且语义未审计的 base 行根据原始输出人工归类。自然语言
18 条则全部逐题对照原文和人工 answer facts，记录在
`data/eval/strict9_reader_natural_language_review_20260907.jsonl`。

## 逐题配对结论

1. **杰克逊是明确的协议/controller 问题。** 两个 reader 在 Exx minimal
   都错误输出 `retrieve_more`，但自然语言都能完整回答公开支持、私下厌恶
   以及政治表演动机。说明至少此题不是基础阅读能力不足，而是充分性/action
   决策把可答题误判为需继续检索。
2. **女妖不是单纯协议问题。** 两个 reader 在 Exx minimal 都
   `retrieve_more`；改成自然语言后又完全相同地把虚弱女妖错认成娜斯提、
   把寻找对象答成费尔南。直接证据仍未被正确绑定，属于主体/关系阅读错误。
3. **莫斯提马和老天师没有被自然语言救回。** 两个 reader仍混淆
   萨科塔/萨卡兹/堕天使，或遗漏老天师对樵夫与织物的态度对照及“再”的
   历史含义。
4. **芙蕾达题暴露稳定的说话人污染。** 两个 reader 都把威尔写成指责
   芙蕾达低头的人，而原文中威尔是在反驳汤姆并为芙蕾达辩护。去掉 Exx
   约束后错误仍完全同向。
5. **协议也可能帮助约束答案。** base 在雷吉斯和迷迭香/阿米娅的 Exx
   输出完整，但自然语言答案分别漏掉团聚目的、夹带“阿米娅以毁灭敌人为
   最终目的”的反向推断。

## 判断

自然语言只让 clean-SFT 从 4/9 增至 5/9，base 反而从 4/9 降至 3/9。
在 9 题的小样本上，不能把这一题差异解释成稳定提升；更重要的是，核心实体、
说话人和关系方向错误没有随协议移除而消失。因此：

- Exx 协议负担存在，但目前证据只支持它是局部瓶颈；
- sufficiency/action controller 需要单独校准，杰克逊是直接训练样本；
- 主要训练优先级仍应是 reader 的主体、说话人、关系方向、时间与对照阅读；
- 不应通过改成自由自然语言输出来绕开 grounding，也不应据此启动 RLVR。

由于同日回查发现历史 clean-SFT 的 strict-v1 judge 只看每个 E-ID 前 600 字，
下一步要先闭环 326 个截断/冲突 fact 的 full-evidence 复判，再构造同问题的
`Full / Full + hard distractor / One-edge-missing / Decoy-full` 配对 SFT，
并把女妖、芙蕾达、莫斯提马、老天师变成最小可解释 hard-negative 模板。
详见 `docs/GOLD_RECALIBRATION_SAFETY_AUDIT_20260907.md`。
