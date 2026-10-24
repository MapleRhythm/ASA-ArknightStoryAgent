# 原生协议扩大配对评测接手记录

## 最新：生成及内容复核已完成

2026-10-14 16:47全部92条生成完成。用户随后要求“复核回答正确率，
格式问题后续调”，因此本次新增独立的**不扣格式/引用绑定**内容复核，
没有修改原冻结的生成脚本、任务、输出或严格可交付评分协议。

- `NATIVE46_CONTENT_REVIEW_20261014.json`：46行、两臂92项证据判断。
- `NATIVE46_CONTENT_REPORT_20261014.md`：结果、边界敏感性、逐题错因示例。
- `audit_native46_content.py`：原始产物哈希/顺序完整性+复核覆盖+可复现汇总。
- 主结果：4B完整内容正确20/46（43.5%）；27B17/46（37.0%）。
- 部分正确：4B13，27B22；错误：4B6，27B7；仅补检索：4B7，27B0。
- 4B1条合理补检索单列，不冒充实际回答正确；另4条无必要、2条2/2越限。
- 不因JSON错误扣分：4B第31行、27B六行仍计正确。
- 43组等权：4B41.9%，27B38.4%；新增38行：18/38 vs 14/38。
- 主干诊断22/46 vs36/46不是完整正确率；27B主要是正确主干后增错。
- 三题差距不是总体优劣的统计证明；不能宣称部署或独立人类金标。

**下次查内容复核请运行 `python3 audit_native46_content.py`。**
生成worker的`semantic_adjudication_complete=false`是原始不可变状态；
旧`audit_native46.py`不加载新增content-only复核时仍显示null，
这不能解释为本次内容复核未做。严格可交付/引用全覆盖分未在本次补评。

用户明确要求双方都用原生协议，扩大题量；不是重训、不是全链路检索。
已在 2026-10-14 15:34 北京时间实际启动，不要重复启动。

远端主目录：
`${REMOTE_ASA_ROOT}/bonsai2_ptuning_20261012`

本轮目录：
`native46_4b_sftkto_vs_bonsai_pt_20261014`

## 固定设计

- 全部审核留出 46 行，共 43 个去标点不同问题文本。
- 包含之前 8 行，新增 38 行；全部重新生成，不能拼历史挑优。
- 4B：旧 teacher_scored_kto_mix_v1_from_soda 权重，SHA 与历史一致；
  只用原生 minimal quote prompt。
- 27B：已训练好外置编码器 final_adapter；只用原生 E-ID prompt。
- 同问题、同轮次、同顺序完整 20 条正文，不输入 target。
- 无输入截断，无答案规则校验，无收费 API，无训练和发布变更。
- greedy，1024 新 token，记录输出上限触发。
- 92 份原始输出；必须另外离线核对实际证据才能报告正确率。
- 冻结协议 `NATIVE46_PROTOCOL_20261014.md`；
  数据哈希、适配器指纹和脚本哈希在 manifest.json。

## 真实进程

- GPU1 lane PID 4077342；4B child PID 4077344。
- GPU2 lane PID 4077343；Bonsai shard0 child PID 4077345。
- GPU2 跑原 val 偶数下标 23 行。
- GPU1 先跑 46 行 4B，完成后自动启动 Bonsai shard1 的另 23 行。
- 不停止其他任务；GPU0 不用。
- 2026-10-14 15:36 左右核实：4B 已完成 5/46，Bonsai 1/46，
  两进程正常生成。该数字是快照，不是最新进度。

## 读取状态

远端使用 `${REMOTE_PYTHON}`：

```
python native46_eval.py status --out native46_4b_sftkto_vs_bonsai_pt_20261014
python audit_native46.py --out native46_4b_sftkto_vs_bonsai_pt_20261014 --old-selection heldout8_selection_20261014.json
```

- `lane1.json`、`lane2.json`：真实 child PID、stage、exit code。
- `fourb.status.json`、`bonsai_shard0.status.json`、`bonsai_shard1.status.json`：
  完成数量、阶段、产物 SHA256。尚未到 shard1 时文件不存在正常。
- 对应 `.log` 和 `.results.jsonl`；原文在不可变 tasks.jsonl。
- 使用既有 SSH askpass，不能输出凭证。

## 验证

- `test_native46_eval.py` 本地/远端 8 项测试通过。
- remote prepare 重新通过 reviewed dataset lineage audit。
- 首批 6 份输出经 `audit_native46.py` 全证据、顺序和 prompt 指纹检查通过。
- 不要更改运行中的 native46_eval.py：worker 会核对脚本 SHA。
- 可添加独立离线报告、审核文件，不能改写原始输出。

## 后续

1. 检查状态和失败日志；已在运行的 worker 不重复启动。
2. 92 份完成后，先执行 audit 加 `--require-complete`。
3. 按全部证据逐题复核两个模型的内容、动作、引用和 JSON。
   GLM target 不是 gold；对原文歧义保留理由与敏感性，不删题。
4. review JSON 格式：
   `{"reviews":[{"arm":"fourb|bonsai","id":"...","semantic_complete":true,
   "strict_deliverable":false,"note":"具体证据理由"}]}`。
5. `audit_native46.py --review REVIEW.json --old-selection heldout8_selection_20261014.json`
   汇总全部46、旧8、新38和43个问题组等权结果。
6. 不把格式率、teacher agreement、NLL 或流程完成率当 QA 正确率。
7. 不声称可部署；原生 GGUF soft embedding 注入仍未验证。
