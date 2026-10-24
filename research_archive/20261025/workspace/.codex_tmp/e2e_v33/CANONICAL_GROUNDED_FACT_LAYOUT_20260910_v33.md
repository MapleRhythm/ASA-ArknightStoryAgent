# ASA：事实展示拼接的窄修复 v33

241 为唯一规范源。文件名日期使用北京时间，回执保留实际 UTC 时间。

## 已保留的修改

修复事实重建展示中的 `。；…。。`：将三处事实间的分号拼接改为换行，
不再额外追加末尾句号。原事实文字、标点、引号、顺序、去重规则与
“推断（非直接事实）”标识全部保留。不修改引文校验、事实选择或回答动作。

只改动一个已有运行源码文件：

`release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/grounding/grounded_fact_answers.py`

新增 `release/ASA-ArknightStoryAgent/tests/test_grounded_fact_layout_regression.py`。
已有 dirty 改动不被回退，HEAD diff 中此前的 fact-scoped quote 修复不属于本次新增。

| 文件状态 | SHA256 |
|---|---|
| 修改前源码 | `52bf832e64093da50de4b974c235d4c7a661e7f810c96a0f07113c395fde9592` |
| 修改后源码 | `b1cdd803884b23eb0f7ecd370bc440ad5512e3996d4196e0e4d41332089370f9` |
| 新增测试 | `8ae6e37efe75da1dce4371a1e7dcc5b9de6dfa5c86c42f9bf8e2dd6d87671189` |

## 验证

先完成隔离验证，确认当前源码与备份一致，才按单文件原子替换应用。

- AST 比对证明只改变三处拼接字符串及一处额外句号；其余代码结构相同。
- 隔离候选 20 项测试通过，覆盖句末标点、引号、已有双句号、内部换行、
  去重、空事实、推断分组与禁止借用兄弟事实引文。
- v29/v30/v31 共 58 条已完成的引用输出重放，与归档基线逐字段一致；
  新展示有 23 条改变，其余不变。所有动作、事实、引用、警告、诊断不变。
- 6 条原预算失败单列保留，没有从中间草稿提取答案。
- 安装后独立 pytest 进程：release **306** 项、root **248** 项全部通过，
  无失败、错误或跳过。release 已包含新增 20 项，不再与隔离测试累加。
- 既有 release 源码哈希只允许这一个文件变化；测试结束后再次核对。
  `git diff --check` 对本次目标通过。

58 条回放是同一批 8 个开发问题的不同条件，不是 58 个独立题目。
无新 GPU 生成，无训练，无模型指针切换，无服务重启，无提交/push。

## 不能冒称正确率提升

此修复只改变排版。v29 的拉普兰德反例仍会将“摇尾乞怜”的错误除名原因
拼进展示，检查点刻意保留该反例。语义蕴含、说话人归属、错误 fact 重建问题
仍须另行解决，不能用更好看的展示掩盖。

此修改只影响 `answer_from_grounded_facts` 分支，不全局重排未进入该分支的 raw answer；
原句中本来就有的双句号也不擅自删改，避免触碰引文和事实文字。

## 工件与恢复

规范目录：`outputs/experiments/grounded_fact_layout_20260910.v33/`。

修改前完整文件保存在 `baseline_grounded_fact_answers.py`，既有相对 HEAD 的
差异另存 `preexisting_source_diff.patch`。恢复时须先确认当前文件仍为本次安装哈希，
再只恢复此文件；如有后续改动，应人工合并，不能直接覆盖。

`isolated_replay_v1.json`、`installed_verification_v1.json`、
两套 XML/日志以及 `checkpoint_v33.json` 绑定完整机器证据。
