# v48：只补齐最终回退记录的隔离候选

规范目录：`outputs/experiments/fallback_trace_retention_candidate_20260910.v48`。
文件日期沿用北京时间 2026-09-10，JSON 时间为 UTC 2026-09-09。

候选只在最终 direct fallback 成功/失败后，向末轮 trace 追加
`final_direct_fallback_attempts`，保存 primary 结论/异常和 fallback 结论/异常。
原采样、解析、校验、提示、回退条件、输入输出均引用未改变的原函数。
已有侧记录不覆盖，嵌套数据保存快照，不受之后的对象修改影响。

## 验证

- 20 项隔离候选协议测试全部通过。
- 全部 14 个 v45 真实缓存状态，与原实现分别回放；只移除新侧记录后，
  原状态、trace、结论、答案、prompt、kwargs、调用次数全部相同。
- 9 个实际 fallback 状态都保留完整 primary 记录；5 个不需要 fallback 的状态
  不添加新字段。10 个首轮终点原样带过。
- 2 个明确标注为合成的异常路径，原答案/trace/stdout 不变，新增回退异常记录。
- 当前未修改的 release 基线 306 项通过；这是基线测试，不是 306 项候选测试。

广测第一次因为缺少 `ASA_NO_MINIRAG_RETRIEVERS` 在收集阶段 exit 2；
未运行测试。原 XML/日志完整保留。读过原 no-MiniRAG 测试及 v33 的测试环境配置后，
第二次按原约定设置完整环境变量，禁用 GPU，306 项通过，无跳过、源码或配置修改。

prepare/unit/replay/baseline-attempt02/close 均成功；`checkpoint_v48.json` 已归档。
preflight：
`ad886e7c16404a56e95d8b346550668e1d3ea5b2a27d54a7ed8b9603a5545bec`。
原源码 SHA：
`43b3611bad56687a112258d0bbf2504126207f63fe5592e3136e87b4be63259a`。

本阶段未改生产，无新模型、检索、GPU 或训练调用。隔离候选通过意味着可以进入
带原文件守卫的仅诊断集成；不能据此宣称答案正确率上升或服务已热加载。
