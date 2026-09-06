# OPERA 面试演示工作台

## 输入

- 浏览器输入 1–1000 个 Unicode 字符的问题，提交前去除首尾空白。
- 固定请求 `{ "question": "...", "retrieval_scope": "all", "top_k": 3 }`，不发送 `case_id`。
- FastAPI `127.0.0.1:8082`、现有 HotpotQA 索引、Qdrant、DashScope 和 DeepSeek 必须可用。密钥与索引版本沿用仓库根 `.env`，浏览器和 Vite 不读取该文件。

## 输出

- 页面实时显示每次 Agent 的实际输入、实际 system prompt、返回文本、Schema 校验与修复尝试；Retriever 显示实际 query、段落标题和逐句正文。
- 输入在调用前发送，输出在该次模型调用返回后发送；不是逐 token 流式生成。每次 Schema 修复都有独立 `call_id`，不能与 Rewrite 混为同一种重试。
- `step_completed` 才代表 Executor 接受了证据与结果。Schema 通过不代表证据已被接受。
- 最终展示答案、证据引用及对应的真实检索句子、运行 ID。`insufficient` 是正常的证据不足结果；HTTP/运行错误单独展示。
- 页面只保留本次内存状态，不持久化历史。构建文件在 `out/opera-frontend/dist/`；验收截图和报告在 `out/opera-frontend/`。原后端本地 trace 和 Langfuse 配置保持原行为。

## 本地启动

在仓库根目录开启 Qdrant（Docker Desktop 需运行）：

```powershell
docker compose -f .\deploy\rag\docker-compose-qdrant.yaml up -d
```

终端一：

```powershell
Set-Location D:\Codes\Golang\Go_Project\ai-chat\app\ragservice\embedding-service
$env:OPERA_DEMO_STREAM_ENABLED = "1"
conda run --no-capture-output -n aiChatRAG python -m uvicorn main:app --host 127.0.0.1 --port 8082
```

终端二：

```powershell
Set-Location D:\Codes\Golang\Go_Project\ai-chat\frontend\opera
npm ci
npm run dev
```

访问 [本地演示页](http://127.0.0.1:5173)。点击“填入示例问题”，再点击“开始推理”。示例来自交接文档，不保证全库检索必然回答成功。

| 参数 / 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `OPERA_DEMO_STREAM_ENABLED` | 关闭 | `1` 或 `true` 开启实时演示接口，其他值返回 404。只设置在后端进程。 |
| `--host` | 上述命令指定 `127.0.0.1` | 仅监听本机，切勿将演示接口用于公网服务。 |
| `--port` | 上述命令指定 `8082` | Python HTTP 端口，与 Vite 代理一致。 |
| `-n aiChatRAG` | 命令显式指定 | 使用已有 RAG Conda 环境。 |
| `--no-capture-output` | 命令显式开启 | 将 Conda 子进程日志直接显示在终端。 |
| Vite host / port | `127.0.0.1` / `5173` | 开发与 preview 均固定；端口被占用时退出，不自动换端口。 |
| Vite proxy target | `http://127.0.0.1:8082` | 仅将 `/api/opera/ask/stream` 与 `/api/health` 转发到 Python。 |
| `top_k` | `3` | 演示页固定每次检索候选数，不提供评测参数入口。 |

支持 Node 22.18+ 或 Node 24；开发时无需旧 Gateway、JWT、MySQL、Redis。不修改旧 `frontend/index.html`。

## 实时接口合同

`POST /opera/ask/stream` 使用 SSE 编码，通过 `fetch` 消费 POST 响应体，不使用只支持 GET 的原生 `EventSource`。没有重连、自动重发或恢复历史的语义。

```text
id: 1
event: run_started
data: {"version":1,"run_id":"<uuid>","seq":1,"type":"run_started","elapsed_ms":0,"data":{"retrieval_scope":"all","top_k":3}}

```

所有业务事件包含 `version=1`、唯一 `run_id`、从 1 连续递增的 `seq`、`type`、`elapsed_ms` 与 `data`。SSE 注释是连接心跳，不代表 Agent 内部进度。

| 事件 | `data` 主要字段 | 时机 |
| --- | --- | --- |
| `run_started` | `retrieval_scope`, `top_k` | 初始化执行器前，提前提供 run ID |
| `initialization` | `stage`, `message` | 初始化的真实阶段切换 |
| `dependency_warning` | `category`, `message` | 可选依赖降级，运行继续 |
| `agent_started` | `call_id`, `agent`, `step_id`, `attempt`, `model`, `instructions`, `input`, `prompt_metadata` | 每次 provider 调用前，含实际修复输入 |
| `agent_output` | `call_id`, `step_id`, `output`, `duration_ms`, `usage` | 实际 provider 文本返回后、校验前 |
| `agent_validated` | `call_id`, `validation`, `will_retry` | 本地 Schema/Pydantic 校验完成 |
| `agent_failed` | `call_id`, `error_type`, 可选安全 `provider_metadata` | provider 异常或无文本 |
| `plan_ready` | `plan` | 计划通过校验与配置上限检查 |
| `step_started` | `step_id`, `step` | 开始子目标 |
| `retrieval_started` | `retrieval_id`, `step_id`, `query`, `top_k` | 开始检索 |
| `retrieval_completed` | `retrieval_id`, `step_id`, `query`, `paragraphs`, `summary` | 检索返回实际候选 |
| `step_completed` | `step_id`, `result` | 证据引用校验后接受子目标 |
| `step_insufficient` | `step_id`, `failure_reason` | 改写预算耗尽 |
| `rewrite_accepted` | `step_id`, `rewritten_query`, `reason` | 接受新的 query |
| `run_completed` | 原 `/opera/ask` 最终响应字段 | `completed` 或 `insufficient` |
| `run_failed` | `code`, `category`, `message`, 可选 `error_type` / `stage` | 安全的运行失败信息，不返回异常原文 |

开启前返回 404；远程客户端或非白名单 Origin 返回 403；请求校验失败返回 422；同进程已有流式运行返回 409。来源仅接受 `http://127.0.0.1:5173`、`http://localhost:5173`，无 Origin 的本机 CLI 也可测试。该限制不是生产鉴权，不能通过公开反向代理暴露。

SSE 响应开始后 HTTP 已是 200，运行错误使用 `run_failed.data.code` 区分 400（沿用原 Agent 输出无效分类）和 503（依赖不可用）。前端不得仅凭 HTTP 200 判定成功，必须收到终态事件。

## 断开、并发与数据边界

- 初始化事件会显示配置检查、Qdrant 连接、BM25 加载、索引一致性核验及提示词读取阶段。每次演示请求先预检配置、artifact 和 Qdrant，无需调用模型。
- Langfuse prompt 拉取设置 `fetch_timeout_seconds=3`、`max_retries=0`，失败后显示降级提示并使用本地 prompt；该超时参数沿用 SDK 网络层语义，不是整次运行时限。模型与提示词选择内容保持原配置。
- Qdrant 预检超时 3 秒，正式索引操作超时 10 秒；OPERA embedding 请求超时 20 秒。现有 SDK 内部重试仍按原行为执行。
- 演示初始化最长等待 60 秒，超时发送 `run_failed`，分类为 `initialization_timeout`，语义码 504。工作线程到下一个事件边界才停止，不提前释放运行槽。
- 前端连接响应等待 8 秒、无业务事件等待 90 秒、整次运行等待 5 分钟。SSE 心跳不重置业务进展计时。超时保留已展示过程，解除页面等待，不自动重新计费运行。
- Vite 检测到后端连接失败时返回 503 与 `code=backend_unavailable`，页面明确提示启动 8082 Python 服务；Qdrant 连接失败、collection 缺失、BM25 异常、模型与 embedding 不可达使用不同的安全中文文案。

- 每个服务进程最多一个流式演示运行（请使用单个 uvicorn worker）；前端同时使用同步提交锁与禁用按钮，防止连点。旧非流式接口不受此限制。
- 同步执行器在独立线程运行；事件订阅通过 `ContextVar` 隔离，不修改共享模型客户端上的回调。
- 事件队列上限 64 条，持续背压 10 秒后停止后续执行。连接断开在下一个事件边界停止后续模型与检索调用；已在途的调用不能撤回，也不保证停止计费。运行槽在工作线程退出后才释放。
- 不提供重新连接或自动恢复；手动重新提交会产生新的请求及成本。原有后端 Schema 修复与 provider SDK 行为保持不变。
- 新事件仅发送本次调用的业务白名单字段，绝不序列化 client 对象、请求头、环境变量或密钥。浏览器显示的返回文本和证据均通过 React 文本节点渲染，不执行 HTML。
- 控制台仅记录运行状态、run ID；输入、系统提示词和返回正文不写入前端日志。实际模型 prompt 本身也不应包含密钥。

## 验证与构建

```powershell
Set-Location D:\Codes\Golang\Go_Project\ai-chat\frontend\opera
npm test
npm run build
npm run preview
```

`preview` 与开发页使用相同的本机地址、端口和代理，请不要同时启动二者。构建配置 `emptyOutDir: false`，不会清空原有产物；离线生成的静态文件不能直接双击 HTML 调用 API。

后端测试：

```powershell
Set-Location D:\Codes\Golang\Go_Project\ai-chat\app\ragservice\embedding-service
conda run -n aiChatRAG python -m unittest discover -s tests
```

测试替身只用于测试文件，不在产品页面中提供伪装成真实运行的模拟数据。`/health` 仅证明 HTTP 进程可达，不证明 OPERA provider 和索引就绪。

实现参考：[Vite 代理配置](https://vite.dev/config/server-options)、[Starlette StreamingResponse](https://starlette.dev/responses/)。
