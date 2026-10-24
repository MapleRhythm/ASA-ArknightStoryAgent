# v49：正式补齐最终回退的诊断记录

规范仓库：`${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_HOME}/ASA-ArknightStoryAgent`。
实验目录：`outputs/experiments/fallback_trace_retention_integration_20260910.v49`。
北京时间 2026-09-10 文件系列；JSON 回执为 UTC 2026-09-09。

## 已安装的修复

仅修改一处已有 release 源码：
`src/asa_arknight_story_agent/inference/generation/answer_conclusion_generation.py`；
新增 `tests/test_final_direct_trace_retention.py`。

在原最终 direct fallback 成功/失败后，向末轮 trace 追加
`final_direct_fallback_attempts`。保留 primary 结论、已有 primary 生成异常、
fallback 结论或异常。已有尝试不覆盖；嵌套记录保存独立快照。

未改变回退条件、调用顺序、prompt、采样、上下文预算、原结论、答案、validator、
E-ID 绑定或任何旧 trace 字段。原有 stdout 异常提示也保留。无回退或空 trace
时，不凭空新建记录。相对安装前原文件为 33 行新增、2 行替换；
不能把整个 dirty worktree 的 Git diff 当作本次改动。

安装前逐文件哈希守卫确认其他既有源码、模型、配置、runner、retriever 和测试
没有变化；新测试路径不存在。候选修改函数和记录函数与 v48 已测实现的 AST
完全相同，其他函数 AST 与原版完全相同。原文件留存在：
`outputs/experiments/fallback_trace_retention_candidate_20260910.v48/answer_conclusion_generation_original.py`。

## 已验证

- 已安装源码的完整 release 测试：**326/326 通过**，无失败、错误或跳过。
- 14 个 v45 实际第二轮状态通过真实已安装 canonical 方法重放。
  仅删去新增侧记录后，旧 trace、状态 digest、结论、最终答案、prompt、kwargs、
  调用次数全部逐字段一致。
- 9 个实际回退状态完整保留 primary；另外 5 个无回退状态不增加此字段。
- 10 个首轮终点原样保留，不重试其中的已知错误。
- 两个合成异常路径保留旧结果与 stdout，并新增回退异常记录；
  这不是新的模型调用或模型失败。

prepare、guard/安装、fresh-process check 均 exit 0。
`installed_verification_v1.json` 保存完整回执；`checkpoint_v49.json` 为最终归档。

preflight：
`3098d76e80da7547f1b6007ba01d01b22e26693337fb597169cba1c05d8d35f3`。

源码安装前：
`43b3611bad56687a112258d0bbf2504126207f63fe5592e3136e87b4be63259a`。

源码安装后：
`a69d1a78258cf4f8c2e2cdc10f424752707481099a7ac23264a9c3bebefd550d`。

新增测试：
`02fb292bf7448685a4a6c9feb33b3db4f41df5118bfa6bbbeec4f0413faa8d43`。

## 边界

这是审计可观测性修复，不是答案语义正确率提升。v45 的人物归属错误、未证明的
制造者关系、引用错绑、过度解读及误拒绝问题仍保留，未通过此改动被修复或掩盖。
未推广 v39 routing recovery 或 v42 姓名门槛，未放松 `动机` 校验，
未恢复自动 E-ID 重绑。

无新 LLM、检索、GPU、训练调用；未更改默认模型或配置，未重启服务，未提交/push。
因此不能声称现有常驻服务已热加载。整个恢复任务现在除更早 v33 外，确有本次
v49 生产源码修改；不要继续声称“所有后续阶段都未改生产”。

历史 v45/v48 等 manifest 保留旧源码哈希不变。它们的“核验当前源码”操作现在会
因这一处已记录变更而有意失败；不得为了重跑而改写历史 manifest。后续核验应使用
v49 明确的安装前/后哈希映射或归档原源码，不能忽略全部旧文件校验。
