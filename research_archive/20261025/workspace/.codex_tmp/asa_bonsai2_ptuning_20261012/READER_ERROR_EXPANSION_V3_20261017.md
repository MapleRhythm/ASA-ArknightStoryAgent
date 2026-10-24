# 根据750条扩增版遗留问题继续补数据

## 依据与范围

依据 `NATIVE46_V2_CONTENT_REPORT_20261017.md` 的逐题内容复核：
750条数据训练的**旧输入层提示编码器**为25/46完整正确，尚非正在训练的逐层
P-tuning v2的新结果。两者不能都简称“v2”后混为同一个实验。

只从已审计750条训练数据的265个不同20证据包中取200个，优先问题较少的包、
按训练来源family轮转采样。新增的是问题覆盖，不是假称新增故事语料。
只把以下通用错因给题目设计者，不传开发集的题目、答案、人物名、证据或逐题笔记。

| 通用问题 | 计划问题数 | 目标 |
|---|---:|---|
| speaker_direction | 100 | 说话者/受话者、施受方向、人物/称号/器物分离 |
| event_state_scope | 100 | 否定、发生/计划/演示/猜测、前后阶段和派别范围 |
| minimal_complete | 100 | 答全核心后停止，不增加无关背景和因果 |
| partial_support | 100 | 双问仅部分有证据，准确保留已知并限定未知 |
| evidence_boundary | 200 | 同实体、邻近相似信息不能替代所问具体关系的证据 |

每包设计3题，共计划600题；1/2与2/2按确定性计划均衡设置。
不同设计类型不强迫标签动作，教师须重新独立判断是否可回答。
新题必须与原750训练题、46验证题和本批已接受新题规范化去重。
不把原题改几个字或复制原答案作为新标签。

## 标注与审核

- 沿用 `annotation_protocol.py` 的SQLite、预算预留、缓存、断点恢复和熔断。
- 仅GLM Coding Plan端点，模型glm-5.3；不回退普通计费端点。
- 题目生成不请求答案；新题用全部20条全文重新标注。
- 每条标签另请求一次语义复审，支持/动作均通过且issues为空才保留。
- 加强主体/对象、否定和状态、无据扩写、部分可答与缺证动作的复核。
- 教师附加质量指令不进入学生原生prompt；实际教师请求cache key绑定这些指令。
- 同一个GLM标注和复审不是独立人工金标，不能宣称全部语义已获证明。
- 不修补失败标签，语义不通过、输出不完整、格式或引用结构错误均隔离。
- 规则仅用于离线数据完整性与来源审计，不增加学生运行时答案校验器。

正式阶段预算上限：问题生成480次请求尝试/500万accounted tokens；
标注及复审1800次请求尝试/1500万accounted tokens；合计2280次/2000万。
这是保守预算上界，不是账单估计；达到上限即停止，不自动加额度。
小批预检独立目录，最多12+64次请求尝试、20万+80万accounted tokens。

## 数据隔离

- 当前深前缀训练使用的750条数据、代码、checkpoint不修改，不中途混入新数据。
- 新数据生成只使用CPU和网络，`CUDA_VISIBLE_DEVICES=""`。
- 46条验证集逐字节保留；禁止与训练在family、document、规范化问题上相交。
- 继续保护原验证来源与act10d5、act10mini。
- 烟测数据不合入正式数据。
- 全批处理完且至少保留350条、五类各至少40条，才进行候选合并。
- 候选集为原750条加新审核通过样本，最多1350条，实际数量由审核决定。
- 候选集manifest明确 `ready_for_training=false`，
  `review_required_before_training=true`，待助手分层抽检后再决定是否放行。
- 自动输出每类3条、合计15条的完整证据抽检队列。
- 本次workflow止于候选数据准备，不自动另开训练，不改当前750条训练。

## 文件与运行方式

脚本：`expand_reader_errors_v3.py`；离线测试：`test_expand_reader_errors_v3.py`。
本地28项测试通过（新增8项加旧标注/审计回归20项），远端新增8项测试通过。

远端根：`${REMOTE_ASA_ROOT}/bonsai2_ptuning_20261012`

预检目录：`reader_errors_v3_20261017_smoke01`，4个包、计划12题。
正式目录：`reader_errors_v3_20261017_formal01`，200个包、计划600题，已后台启动。
是否真正启动和完成须读取各目录的launch/workflow/status/ledger，不凭路径判断。

```sh
python expand_reader_errors_v3.py prepare --out reader_errors_v3_20261017_formal01 --packs 200
python expand_reader_errors_v3.py launch --out reader_errors_v3_20261017_formal01
```

凭证只从既有权限受限的`${REMOTE_HOME}/.bigmodel_env`读取到进程内存，不打印或另存。
`launch.json`记录真实PID；`workflow.json`记录阶段；
`questions/status.json`和`annotation/status.json`记录实际完成数量、API调用与预算；
候选合并结果位于`merged_candidate/manifest.json`。

若blocked_or_failed，先检查原因；同版本代码、同目录和不变预算可从阶段ledger恢复，
不得覆盖旧产物或跳过审核。对已完成candidate merge不要重复运行workflow。

## 正式启动记录

正式600题workflow已后台启动，首次PID **2132134**；是否存活和最新进度以远端为准。
预检4个包中3个生成成功、1个失败隔离，共9个待标注新问题。
15:53:46北京时间的预检标注快照为8条通过GLM二次复审、1条仍待处理，不是最终计数。
15:54:46北京时间的最终预检结果为9条全部处理，8条通过、1条隔离；共18次
标注/复审API调用、122774 accounted tokens。隔离题
`384939514ed8d5c331f347b9` 的原因是 `incomplete_output`，
题目生成阶段隔离的 `pack-0003` 也为 `incomplete_output`；
不把输出不完整误报为语义错误或网络超时，未修补这些失败产物。
另已对8条保留样本执行source完整性检查及annotate/review双receipt校验，
训练/验证在source_family、document、question三项均无重合。
助手另抽检6条：4条对照其引用全文，缺证与部分可答各1条对照全部20条全文。
尚未看到必须弃用本版管线的实质错误；一条动机题仍有4条事实、略长，保留这个限制，
不宣称精简问题已完全解决。详见 `READER_ERROR_V3_PREFLIGHT_REVIEW_20261017.json`。
预检产物不混入正式训练，正式标签仍必须逐条复审并接受后续分层抽检。

15:58:34北京时间的正式阶段快照：200包中5包生成通过、1包隔离、194包待处理；
7次API调用、97320 accounted tokens，无stop_reason。尚未完成600题标注。
同期逐层P-tuning v2训练已到283/750步，该步使用20条证据、`truncated=false`；
两个后台进程均存活。本节仅记录当时核验结果，不代替远端最新状态。
