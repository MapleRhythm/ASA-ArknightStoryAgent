# ASA strict9 reader × evidence 诊断结论

日期：2026-09-07

## 结论

这是固定证据条件下的 **P0 流程 smoke**，不是生产系统或线上问答正确率。
样本只有 9 个问题族；每道题的 11.1 个百分点就是一道题，不能据此宣称模型
存在统计显著的提升或退化。

人工复核后的完整回答率为：

| Reader | 最小充分证据 | + hard distractor | 实际 prompt |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B base | 4/9 = 44.4% | 0/9 | 2/9 = 22.2% |
| clean-SFT | 4/9 = 44.4% | 1/9 = 11.1% | 1/9 = 11.1% |

在这批题上，clean-SFT 改善了 JSON/schema 可审计性，但没有改善实际 prompt
的回答语义；真实同场景干扰使两个 reader 都显著退化。27 个 evidence
condition 均经人工确认充分，因此 actual prompt 的失败不能简单归因为
strict9 召回缺失。

同日后续审计发现，该 clean-SFT adapter 的 strict-v1 标签由只看每个 E-ID
前 600 字的 judge 生成；326/1929 个 fact 因截断或内部冲突被隔离。因此本
smoke 的生成结果仍然有效，但它只描述历史 adapter，不能代表 full-evidence
干净 SFT 的上限。

当前不应启动正式 RLVR，也不应为了这 9 题修改生产 MiniRAG、top-k、MMR、
chain packing 或自动二次检索。先完成
`docs/GOLD_RECALIBRATION_SAFETY_AUDIT_20260907.md` 定义的 326 项完整证据复判，
再构造同题配对的 RAFT 风格 SFT 数据，补充主体、说话人、关系方向、时间和
否定 hard negatives，最后完成约 60 题正式 P0。

## 实验身份与可追溯性

可信运行使用 241 的三张空闲 RTX 4090 和 Transformers/PEFT：

- base：`/home/zhb/ASA-ArknightStoryAgent/model/qwen3.5-4b`
- clean-SFT：
  `/mnt/wwn-0x5000cca295f3594f/zhb_asa/relevance_data_20260906/models/exx_binding_clean_sft_v2_a100_r16_lr2e6_e1_retry8192_20260904`
- 远端实验目录：
  `/mnt/wwn-0x5000cca295f3594f/zhb_asa/rank_v8_coverage/experiments/reader_evidence_p0_strict9_20260907_primaryreview_v2`
- 本地只读汇总目录：
  `/home/maplerhythm/ASA/.codex_tmp/reader_evidence_strict9_results_final_20260907`

关键哈希：

| 资产 | SHA256 |
| --- | --- |
| 原始 strict9 snapshot | `5fc3420576571d2a05b24263c8ced33af24b77ccaa1709001ebcc7ba1a857777` |
| 人工 annotation | `c41f91dd098ae9cd344b78327d14f53cfefb0939051d23569fa7e7e001e13af4` |
| 27 行诊断矩阵 | `1886f257e4e289b8bd71a7282eceb89190bff8be745d3319b5253e07051580f0` |
| 4B base 顶层资产聚合 | `4dcba47639cd5ccef2436373743b1f122d11d08c85aa63c34fc8ce37aba1ff2b` |
| clean-SFT adapter | `75c3f1e0c2946c77537d81266b408913ce51bd6a41563ee2ca884e3a76c3c41c` |
| base predictions | `f6f744b3f88f8c0cdbe797efa570816cfbc0cad8ad9807d450f39f09b839eeef` |
| clean-SFT predictions | `7e04baf528c00a2d7fbb048523307e38a9366daf7c0473ceb0052f9e8ab8135e` |
| base final GLM audit | `31fe4e3ddd402ad486a20176cd5fd1608989b3840e30332379177c64447aceb6` |
| clean-SFT final GLM audit | `33674b1a8d70587a21404b4cd6de624de6d7116af0664f455f4e913a08273619` |
| 自动汇总 | `3854e3c799ffbdf795532e705f86121df7e594f21c20b4a29848b27b149f8a66` |

两种 reader 均运行 27/27 行，temperature 0、non-thinking canonical Exx
prompt、`max_new_tokens=768`，且 0 条 generation truncation。为修改中性
reviewer provenance 所做的复跑，其 prompt + raw output 投影与首次运行
完全一致。

9B 没有运行：241、248 和本地都没有找到可核验的 Qwen3.5-9B 资产，不能
伪造第三个 reader。

## 自动审计与人工修正

GLM evidence-only audit 的原始 `complete` 读数：

| Reader | 最小充分证据 | + hard distractor | 实际 prompt |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B base | 4/9 = 44.4% | 0/9 | 2/9 = 22.2% |
| clean-SFT | 5/9 = 55.6% | 1/9 = 11.1% | 2/9 = 22.2% |

base 有 21/27 行同时通过协议并获得有效语义审计；clean-SFT 为 27/27。
没有成功获得 audit 的行仍保留在 9 题分母中，不当作成功。

对 clean-SFT 自动判为 `complete` 的输出做人审后，记录两条假阳性：

1. `3c4ca0be...::actual_prompt`：模型只答“企鹅物流的信使”，漏掉 E8
   明确写出的真实身份“堕天使”，也没有完成“普通信使”为何是敷衍自称的
   解释。应由 `complete` 改为 `partial`。
2. `60cb1081...::minimal_sufficient`：模型只复述保护樵夫及其女儿，没有
   使用 E2 完成对诡异织物毫不留情的态度对照，也没有解释“再”表示不愿
   重演此前处决失控徒弟。应由 `complete` 改为 `partial`。

因此 clean-SFT 的自动 8 条 `complete` 中至少 2 条是假阳性；人工修正后的
三格结果为 4/9、1/9、1/9。机器审计原始结果没有被覆写，人工修正以
`data/eval/strict9_reader_semantic_overrides_20260907.jsonl` 绑定 prediction、
audit 和 matrix SHA 单独保存。

协议层汇总：

| Reader | strict JSON | schema valid | action correct | over-retrieve |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B base | 26/27 | 21/27 | 22/27 | 4/27 |
| clean-SFT | 27/27 | 23/27 | 23/27 | 4/27 |

这解释了 clean-SFT 的真实收益边界：输出更稳定、更容易审计，但 schema/action
不能替代回答正确率。

## 主要失败模式

1. **抗干扰能力很弱。** base 从 minimal 的 4/9 降至 hard distractor 的
   0/9；clean-SFT 从人工修正后的 4/9 降至 1/9。加入一条真实同角色或
   同场景干扰，常导致人物、职务、态度和事件关系串线。
2. **实际 prompt 的证据足够，reader 仍没有正确使用。** actual prompt
   9/9 均满足人工 rubric，却只有 base 2/9、clean-SFT 1/9 完整回答。
   因此 retrieval coverage 不是这 9 个固定输入上的唯一瓶颈。
3. **证据充分时仍过度检索。** 两个 reader 都在杰克逊和虚弱女妖的
   minimal/distractor 条件输出 `retrieve_more`；这不是安全拒答，而是
   sufficiency/controller 误判。
4. **主体和关系方向不稳。** 莫斯提马真实身份、老天师的态度对照、
   芙蕾达/威尔/镇民的“谁认为谁是叛徒”等题，暴露出职业与种族、说话人与
   观点持有者、表象与动机之间的错误绑定。
5. **自动裁判会高估完整性。** 本次两条 override 说明，单一 GLM judge
   不能同时承担标签生成、训练奖励和最终验收；关键 complete、分歧案例和
   随机样本仍需人工或独立裁判复核。

## 被排除的并发实验

提交 `56e8858` 曾混入
`experiments/reader_evidence_p0_strict9_20260907_gpt6audit_v2`。该目录来自
248 A100/vLLM，使用的是较旧 adapter
`exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827`，不是本轮指定
的 clean-SFT；其 base 27/27 输出 `<think>` 并在 768 token 截断，说明
template/停止条件失配，不能把“base 0%”解释成阅读能力为零。随后直接检查
本轮 4B checkpoint 的 `chat_template.jinja`，确认
`enable_thinking=false` 时 canonical assistant prefix 会包含空
`<think></think>` 块；诊断用 vLLM runner 已改为调用 tokenizer template，
不再手写缺失该块的 prefix。该修复只影响旁路诊断工具，不改生产 runtime。

它的矩阵 SHA 为
`735fc63c3bf754e9cccfcc2c6a2d89a657f85c2723d7a997e0163d13278c06f2`，
clean prediction SHA 为
`ad13a205bbcfb6be5222685c78e7ec4f7650538860a1f01012b18d875e08cf9b`，
均与上面的可信 v2 不同。该实验已从当前工作树移除；Git 历史仍保留原始
提交供审计，但其数字不能覆盖或混入本报告。

## 下一步

1. 先完成 GOLD_RECAL 326 个隔离 fact 的 full-evidence 复判，重建带内容
   指纹的候选正例与动作标签。
2. 从 strict9 的失败样式扩成同题配对的
   `Full / Full + hard distractor / One-edge-missing / Decoy-full /
   None-or-conflicting` 数据；每个状态保持长度与格式尽量匹配。
3. 第一轮训练采用 RAFT 风格 paired SFT，重点加入说话人、主体、对象、
   关系方向、时间、否定和职业/种族混淆 hard negatives；保留 clean-SFT
   只作为历史格式基线。
4. 在 minimal 失败题上补自然语言短答对照，区分 Exx 协议负担与基础阅读
   能力；不要把自然语言分数直接混入 Exx 模型排名。
5. 以约 60 个 family-held-out 开发问题完成正式 3-reader × 3-evidence
   P0；只有取得可核验 9B 资产时才加入 9B。
6. 正式 P0 与独立人工验收完成前，不启动 RLVR，不改生产检索默认。

## 后续自然语言对照

上述第 3 项已在同一 strict9 minimal evidence 上完成。人工结果为：

| Reader | Exx minimal complete | 自然语言 minimal complete |
| --- | ---: | ---: |
| Qwen3.5-4B base | 4/9 | 3/9 |
| clean-SFT | 4/9 | 5/9 |

只有杰克逊题被两个 reader 的自然语言输出明确救回；莫斯提马、老天师、
女妖和芙蕾达等核心实体/关系错误仍存在。因而协议负担是局部问题而非当前
主瓶颈，训练优先级仍是主体/说话人/关系方向 hard negatives 与配对证据状态
SFT。完整逐题分析见
`docs/STRICT9_READER_NATURAL_LANGUAGE_CONTROL_20260907.md`。
