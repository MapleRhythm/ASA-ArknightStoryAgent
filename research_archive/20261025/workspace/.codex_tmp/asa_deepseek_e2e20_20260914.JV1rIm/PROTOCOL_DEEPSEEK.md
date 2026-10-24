# DeepSeek 20题全链路配对回归（推理前冻结）

日期：2026-09-14。用户要求对 DeepSeek 跑与 4B 相同的20题。

复用4B原始questions.txt，SHA256：
8217bab42c62d6130582d970a46400434eefeab28bdb9eed3771fb89e2fcb4e7。
评分复用4B的PROTOCOL.md，SHA256：
2e5d297272385c258311ec35311c1f3890c297d424cf6c9d8636cabf2673ae7f。
20题分母固定；不可删除失败题、变更标准或挑选重跑结果。
原评分对“回答招式来历但没给学招建议”的Q16按完整处理，本次一致。

真实远端CPUInferencePipeline和CLI不变，启动时仅在内存将VllmRunner替换成
DeepSeek API runner。初始规划、可能的后续规划、结论和direct-answer fallback
全部调用deepseek-flash。真实混合检索、重排、top20完整正文、校验和最多2轮不变；
每题无对话历史。不用冻结证据，不把gold/旧答案/4B生成内容传给模型。

接口传输差异必须披露：

- 直接HTTP到api.deepseek.com/chat/completions；模型deepseek-flash；
  thinking disabled，与先前DeepSeek小测一致。没有API自动重试。
- 原ChatML中的system/user内容原样转换成messages，仅删除控制标记和末尾
  assistant预填充“{”/JSON开头。API返回完整JSON，交给同一解析器处理。
  不添加旧API入口附加system提示，不加json_object格式辅助。
- max_tokens与原管线调用预算一致（默认512，结论1536）；temperature沿用原调用。
  不发送vLLM专有repetition_penalty。按官方文档，non-thinking时top_p固定1.0，
  不伪称与vLLM的0.9完全一致，因此只说同管线配对比较，不说采样完全相同。
- 沿用4B tokenizer对原prompt检查“输入+输出预算<=10000”的容量门槛；
  不以DeepSeek更长context绕过原门槛，不裁剪/静默扩大预算。
  同时记录API服务端usage；两者token数不可混称。
- 记录发送的完整HTTP JSON body、响应content、finish_reason、usage、耗时。
  检查每条正文是否完整保留在原prompt与发送messages中；API无法提供实际
  服务端token IDs，因此不能声称做了vLLM那种服务端逐token一致核验。
- API单次timeout120秒，最多120次付费请求的安全上限；鉴权/余额/限流失败停止
  后续网络请求，保留失败分母。输出length与传输错误记录，不隐去。

仅使用GPU0上的原retriever，DeepSeek生成在API侧完成；不加载4B权重、不训练、
不改远端生产配置和代码、不停止其他作业。密钥仅通过隐藏终端输入保存在进程内存，
不写文件、命令参数、环境变量或日志；输出有密钥过滤保护。

事后按相同原始剧情核对最终对外答案，原始草稿不能顶替最终拒答。
报告严格完整/部分/错误/拒答/异常、宽松可用、API与流程成功、耗时、证据完整性，
并提供与4B同题对照。正确率是这20题开发回归的点估计，不保证线上达标。
