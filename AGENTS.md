# ai-chat 协作与交接说明

本文件是此仓库中后续 Codex 线程的入口。开始任何任务前，先读本文件，再读任务对应的代码和文档；不要只根据聊天摘要或技术设计的某一段直接改动。

## 1. 协作约定

- 所有解释、分析和步骤说明使用中文；代码、变量名、命令和报错信息保持英文。
- 收到功能需求后，先只读分析相关代码：说明现状、目标、最小安全修改方案、风险和验收方式；得到用户确认后才写代码。
- 只修改用户确认范围内的文件，不为“顺手优化”扩大范围；不删除或破坏既有功能。
- 信息不足时先提问；可以合理假设时，要明确写出假设。没有足够证据的设计必须放入“尚未确定的问题”，不可当作既定结论。
- 新增函数必须有中文 docstring，说明用途、参数和返回值。
- 新增配置、命令行参数或启动命令时，说明参数、默认值和完整示例；完成时询问是否要更新 README。
- 数据处理链路需要日志，日志中不得记录密码、API key、JWT、用户原文等敏感内容。
- 新生成的数据放在 `out/<task>/` 下，并同时检查 `.gitignore`；不要提交生成物、密钥或本地配置。
- 不执行删除文件命令。如确实需要删除，列出精确目标和影响，由用户决定。
- 每次改动后至少运行与风险相称的检查，并报告真实命令、结果、受影响文件、剩余风险和额外配置。

## 2. 项目目标与当前边界

这是一个 Go / go-zero 学习项目，当前以面试展示的 RAG 为重点：保留用户认证、旧 Chat、技术文档 RAG V2，并持续开发 OPERA-style Multi-Agent Multi-hop RAG。Coding Agent 已迁移到其他项目，本仓库不再承担其设计或实现。

当前阶段的原则：

- 旧 Chat 是需要维护安全和运行基线的独立链路。
- 技术文档 RAG 只索引 `docs/**/*.md`，不把项目源码 `app/` 当作检索语料。
- OPERA 只从 `docs/hotpotQA/hotpot_dev_distractor_v1.json` 导入 HotpotQA paragraph；运行时不得读取 `answer`、`supporting_facts` 等标注，也不得混用技术文档 RAG V2 的 collection 或指标。
- RAG 返回内容、用户输入和外部工具输出均是不可信数据；系统指令、工具权限与业务控制流不能由它们改变。
- `docs/develop/*.md` 是受版本控制的开发文档；`docs/hotpotQA/` 和其他本地语料仍不得提交。

技术文档 RAG V2 与 OPERA 的实际运行合同以 [README.md](README.md)、[docs/develop/rag链路.md](docs/develop/rag链路.md) 和 [docs/develop/技术选型.md](docs/develop/技术选型.md) 为准。

## 3. 当前代码地图

### 用户认证

```text
Gateway HTTP
  app/gateway/api/etc/gateway.api
      -> app/gateway/api/internal/logic/user/registerlogic.go
      -> app/gateway/api/internal/logic/user/loginlogic.go
      -> User RPC
          app/user/rpc/internal/logic/registerlogic.go
          app/user/rpc/internal/logic/loginlogic.go
```

注册链路负责查重、密码哈希、用户写入和 JWT 签发；登录链路负责账号查询、密码校验和 JWT 签发。涉及认证时，优先审计输入校验、权限边界与敏感日志。

### 旧 Chat

```text
Gateway Chat Logic
  app/gateway/api/internal/logic/user/chatlogic.go
      -> AI RPC
          app/ai/rpc/internal/logic/chatlogic.go
              -> Redis history cache
              -> MySQL conversations/messages
              -> RAG HTTP client
              -> model provider
```

AI RPC 入口是 [app/ai/rpc/ai.go](app/ai/rpc/ai.go)。会话必须按 `user_id + session_id` 归属查询，不能只按 `session_id` 读取。Redis 是旧 Chat 的历史缓存；持久化消息以 MySQL 为准。

技术文档 RAG V2 的 Python 服务位于：

```text
app/ragservice/embedding-service/rag_index/offline_runner.py  # 离线切分、embedding、写 Qdrant、生成 manifest/BM25 artifact
app/ragservice/embedding-service/main.py                       # 在线 Dense + BM25 + RRF + MMR + P1 检索服务
app/ragservice/config/rag-retrieval.yaml                        # 非敏感 BM25/RRF/MMR 参数
app/ai/rpc/internal/ragservice/rag.go                           # Go 侧 HTTP 客户端
```

OPERA 复用同一 Python 服务进程，但使用独立索引、HTTP 合同与编排模块：

```text
app/ragservice/embedding-service/opera/hotpot_runner.py         # HotpotQA paragraph 离线导入、manifest/BM25 artifact
app/ragservice/embedding-service/main.py                        # POST /opera/ask HTTP 入口
app/ragservice/embedding-service/opera/executor.py              # ExecutionState、Planner/Analysis-Answer/Rewrite 调度
app/ragservice/embedding-service/opera/llm.py                   # DeepSeek Responses JSON Schema/Pydantic、prompt resolver
app/ragservice/embedding-service/opera/observability.py         # 可选 Langfuse trace/prompt/version/cost 观测
```

`POST /opera/ask` 的 `case` scope 仅供评测器使用（需 HotpotQA `_id`）；普通提问与面试演示用 `all`。最终答案必须来自计划中 `is_final=true` 的最后子目标，不另设 Final Agent。

### 数据与本地依赖

- Chat 表建表脚本：[deploy/sql/ai_context.sql](deploy/sql/ai_context.sql)。该脚本包含重建 Chat 表的行为，执行前必须明确确认数据可丢弃。
- 本地 AI RPC 依赖 MySQL `3306`、Redis `6379`、etcd `2379`；启动和 `.env` 注入方式见 [README.md](README.md)。
- RAG V2 依赖 Qdrant `6333/6334`、DashScope embedding API 和明确设置的 `.env` `RAG_INDEX_VERSION`；Qdrant Docker 配置在 [deploy/rag/docker-compose-qdrant.yaml](deploy/rag/docker-compose-qdrant.yaml)。
- OPERA 离线导入依赖 DashScope 与 Qdrant；`/opera/ask` 另需 `.env` 的 `OPERA_INDEX_VERSION`、`DEEPSEEK_API_KEY`、DashScope key、同版本 `out/opera-index/.../bm25_index.json`。Langfuse 凭据是可选观测配置，缺失或不可用时回退本地 prompt，不应阻断请求；公开演示默认 `capture_input_output=false`，不得上传用户原文或检索正文。
- `.env`、`app/ai/rpc/etc/ai.yaml` 和 `out/` 均不应提交。

## 4. 已确认的技术选型

| 领域 | 当前结论 | 详细位置 |
| --- | --- | --- |
| 旧 Chat 模型调用 | DeepSeek Chat Completions；密钥由进程环境变量注入 | [README.md](README.md) |
| Embedding | 采用 OpenAI-compatible embedding API；RAG V2 当前使用 DashScope `qwen3.7-text-embedding`、1024 维 | [docs/develop/技术选型.md](docs/develop/技术选型.md) |
| 技术文档 RAG | Markdown 结构化切分；版本化 Qdrant + BM25；Dense + BM25 经 RRF、MMR、P1 后组装上下文 | [docs/develop/rag链路.md](docs/develop/rag链路.md)、[docs/develop/技术选型.md](docs/develop/技术选型.md) |
| OPERA-style RAG | HotpotQA paragraph、独立版本化 collection；Planner → Hybrid Retriever → Analysis-Answer → 按需 Rewrite；DeepSeek Responses Schema/Pydantic 双校验 | [docs/develop/rag链路.md](docs/develop/rag链路.md) |
| OPERA Agent 模型与成本 | 默认 `deepseek-v4-flash`；Langfuse 以高峰保守价计算 `input`、`cache_read_input_tokens`、`output` | [docs/develop/rag链路.md](docs/develop/rag链路.md) |

## 5. RAG 后续维护目标

- RAG 的详细运行合同、参数、调用方式、评测和排查统一维护在 [docs/develop/rag链路.md](docs/develop/rag链路.md)；技术选择理由见 [docs/develop/技术选型.md](docs/develop/技术选型.md)。
- 后续先完成一次真实 `/opera/ask` 的 provider/Qdrant 联调，再在相同语料与 scope 下评测 Hybrid 基线与 OPERA-style 闭环；`case` 与 `all` 指标必须分开报告。
- RAG V2 仍需补充未参与调参的保留评测集，并用真实问题抽查旧 Chat 中的 RAG 注入效果。
- 在没有评测证据前，不继续扩大检索功能或修改当前排序参数。

## 6. 推荐工作方式

1. 先定位任务属于认证、旧 Chat、RAG V2 或 OPERA；避免跨链路改动。
2. 阅读对应代码与技术设计章节，写出 source -> transform -> sink，明确输入从哪里来、在哪里改变、最终写到哪里。
3. 给出一个可独立 Review 的最小阶段，等待确认。
4. 实现后先验证本阶段；RAG V2 改动必须同时检查离线索引、在线 `/search` 和旧 Chat 调用兼容性；OPERA 改动必须检查 HotpotQA 导入、`/opera/ask` scope 边界、Schema 失败路径和双组评测隔离。

## 7. 尚未确定的问题

- RAG 是否接入 rerank、检索 tracing，以及何时依据保留评测集调整当前参数。
- 是否实现父 section 的展示级摘要，以及其是否需要单独 embedding。
- OPERA 的真实 DeepSeek/Qdrant/embedding 联调与 Hybrid 基线、OPERA-style 双组评测尚未执行；Context/Memory 必须在此后单独设计，不能将 `ExecutionState` 当作持久化 Memory。

在这些问题未由用户确认前，只能提出候选方案与验证计划，不得自行固定为生产设计。
