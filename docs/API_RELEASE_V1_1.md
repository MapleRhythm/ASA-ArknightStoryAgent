# ASA API 固定版 · api-v1.1.0

日期：2026-09-14。本次增加自定义 API，沿用独立发布分支，
不合并研究工作区、不修改远端模型实验，也不覆盖 `api-v1.0.0` 标签。

## 在网页切换接口

打开「连接与配置 → API 连接」，填写：

| 配置 | 说明 |
| --- | --- |
| 接口地址 | Chat Completions Base URL 或完整接口路径；自动补齐 `/chat/completions` |
| 模型 ID | 服务商提供的准确模型名称，支持带组织前缀的 ID |
| 请求格式 | 通用兼容不发送 DeepSeek 专属字段；DeepSeek 非思考额外发送 `thinking.type=disabled` |
| API 密钥 | 使用 Bearer 鉴权，输入框为密码类型，保存或关闭后清空 |
| 本机无需密钥 | 仅回环地址可明确启用；不会借用环境变量中的其他密钥 |

默认仍是 `deepseek-flash` 与 DeepSeek 非思考格式。编辑地址后，表单切换为通用兼容；
使用需要 DeepSeek 参数的兼容代理时，可显式选择 DeepSeek 非思考。
「填入 DeepSeek 默认值」只填表单，不自动保存或调用模型。

同一规范化地址留空密钥，可保留当前密钥；地址一旦改变（包括路径），必须重新输入，
不能将原密钥自动带到新服务。保存失败时继续使用旧配置。正在回答问题时禁止改配置，
不排队等待切换；停止后需等待该请求的本地子进程清理完成，再保存。

保存与准备检查不会调用服务商，不消耗模型额度，也**不代表 API 连通性已验证**。
只有用户发送问题时才会调用模型。服务商需要接受标准 `messages`、`model`、
`max_tokens`、`temperature`、`top_p`，并返回 `choices[].message.content`。
本界面不适配其他原生协议（如 Messages、GenerateContent、Responses）或自定义鉴权头。
通用格式的思考模式由服务商决定，不能承诺所有自定义模型都关闭思考。

## 配置寿命与安全

- 网页配置和密钥仅保留在当前服务进程内存，不写入配置文件、localStorage、
  sessionStorage、对话导出或 Git；重启服务后恢复启动时的配置。
- 浏览器只收到 `key_configured` 状态，不会取回真实密钥。应用主动清空密钥输入框，
  但不能控制浏览器扩展、密码管理器或用户设备本身。
- 密钥仅通过该次推理的专用环境变量传给子进程，不出现在命令参数；
  子进程不继承其他名称以 `_API_KEY` 结尾的环境变量。
- 问题、最近已完成的对话和检索证据将发送至所选 API。只使用可信服务；
  页面会显示当前目标主机，每条回答及其导出记录保留当时的模型名称。
- 远程 API 必须使用 HTTPS。HTTP 仅允许 `localhost` 或回环 IP；
  不允许 URL 内含用户信息、查询参数或片段。API 请求不跟随 HTTP 重定向，
  避免把凭证和证据转发到另一个地址。请填写最终接口地址。
- 这是本地单用户工具。修改设置只允许回环来源，并校验同源 JSON／Host。
  远程使用请通过 SSH 本地端口转发，或在服务端通过启动参数配置。
  不要直接暴露公网；反向代理会以本机来源转发请求，必须另行保护设置路由，
  并提供用户认证、TLS、访问控制和限额。回环来源检查不替代身份认证。

## 启动示例

先按 README 准备检索依赖、BGE 模型和索引。默认服务也可无密钥启动，再从网页配置：

```bash
bash scripts/run_web_ui.sh
```

自定义远程服务，使用隐藏输入的密钥：

```bash
bash scripts/run_web_ui.sh \
  --api-base-url https://your-provider.example/v1 \
  --model your-model-id --request-profile standard --ask-key
```

无交互启动时，用 `--api-key-env YOUR_PROVIDER_API_KEY` 指向服务进程已有的环境变量。
自定义地址不会自动沿用 `DEEPSEEK_API_KEY`。本项目不自动读取 `.env`，
也不替用户把密钥写入启动脚本或 Git。

本机已运行的兼容接口无需鉴权时：

```bash
bash scripts/run_web_ui.sh \
  --api-base-url http://127.0.0.1:8000/v1 \
  --model your-local-model --no-auth
```

这不会替你启动本机模型服务。Web UI 仍只通过 HTTP API 生成答案。

命令行直接推理的自定义方式：

```bash
bash scripts/run_api_stable.sh \
  --api-base-url https://your-provider.example/v1 \
  --model your-model-id --api-key-env YOUR_PROVIDER_API_KEY \
  --api-extra-body '{}' "你的剧情问题"
```

网页内修改不影响这个独立命令行入口，后者每次从固定配置和显式参数启动。

## 不变的发布边界

检索及证据策略不变：CPU BGE + BM25、无重排器、每次最多20个完整 chunk、
最多2轮、无隐式付费重试；不会在失败时自动切换本地4B或另一个 API。
“完整”指所选 chunk 不被字符截断，不代表整章文本或无限上下文。
超出服务商上下文上限或输出预算时明确失败，不返回残缺答案。

前端的会话、原文查看、复制／导出、IME、手机导航、停止与重试保留。
旧版本记录没有模型字段时按 v1.0 的 `deepseek-flash` 展示，不改标成新配置的模型。

## 验证

```bash
python3 -m pytest tests -q
npm run check
npm run test:ui
```

本次 Python 81 项、Chromium 15 项测试通过，回归包括：
配置校验、密钥隔离、修改互斥、同源／来源限制、
真实轻量子进程、模拟 HTTP 服务的地址／模型／请求字段／鉴权、拒绝重定向、
浏览器真实设置路由、模型标识与导出、桌面和手机设置面板。
全部凭证为显式虚构的测试字符串，没有请求真实模型服务。

未做付费 API 测试或新的问答准确率评测，不能直接套用历史远端 GPU 管线的成绩。
