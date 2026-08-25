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

这是一个 Go / go-zero 学习项目。当前存在用户认证、旧 Chat 和技术文档 RAG 三条链路；后续主线是新增一个与旧 Chat 解耦的 Coding Agent。

当前阶段的原则：

- 旧 Chat 是需要维护安全和运行基线的独立链路，不应成为 Coding Agent 的实现前置条件。
- Coding Agent 使用独立的 Responses 链路和自维护上下文；不要依赖某一模型厂商的上下文 ID。
- 技术文档 RAG 只索引 `docs/**/*.md`，不把项目源码 `app/` 当作检索语料。
- RAG 返回内容、用户输入和外部工具输出均是不可信数据；系统指令、工具权限与业务控制流不能由它们改变。

详细方案和计划见 [docs/coding-agent技术设计.md](docs/coding-agent技术设计.md)。该文档同时包含已确认设计与未来计划；实施前必须区分两者。

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

旧 RAG 的 Python 服务位于：

```text
app/ragservice/embedding-service/rag_index/offline_runner.py  # V2 离线切分、embedding、写 Qdrant
app/ragservice/embedding-service/main.py               # 在线检索服务
app/ai/rpc/internal/ragservice/rag.go                   # Go 侧 HTTP 客户端
```

### 数据与本地依赖

- Chat 表建表脚本：[deploy/sql/ai_context.sql](deploy/sql/ai_context.sql)。该脚本包含重建 Chat 表的行为，执行前必须明确确认数据可丢弃。
- 本地 AI RPC 依赖 MySQL `3306`、Redis `6379`、etcd `2379`；启动和 `.env` 注入方式见 [README.md](README.md)。
- `.env`、`app/ai/rpc/etc/ai.yaml` 和 `out/` 均不应提交。

## 4. 已确认的技术选型

| 领域 | 当前结论 | 详细位置 |
| --- | --- | --- |
| 模型调用 | 统一采用 OpenAI-compatible 合同；Responses Adapter 与 Chat Completions Adapter 都保留 | 技术设计第 7.2 节 |
| Embedding | 采用 OpenAI-compatible embedding API，模型和供应商可替换 | 技术设计第 7.3 节 |
| Coding Agent | 独立于旧 Chat；先规划、Writer 编码、Auditor 审计、受控修订 | 技术设计第 3.1、5、9、19 节 |
| 长期记忆 | 单用户 Qdrant-only；scope 仅 `conversation`、`global` | 技术设计第 8.1、10.1、10.2 节 |
| 上下文 | Coding Agent 自维护上下文快照；Redis 不作为其恢复来源 | 技术设计第 8.3、10.3 节 |
| 技术文档 RAG | Markdown 结构化切分；Dense + BM25，经 RRF、MMR 后再组装上下文 | 技术设计第 3.3、10.4 节 |
| 可观测性 | Langfuse Cloud 追踪 LLM；Loki 仅作为后续传统服务日志演进 | 技术设计第 14 节 |
| 异步事件 | Redis Streams、Outbox、消费幂等属于 V2 以后按需演进，不是 V1 前置条件 | 技术设计第 10.5、19 节 |

## 5. RAG V2 已确认的切分约束

- 标题路径使用可变长标题栈：每项保存 `level` 和 `text`；新标题出现时弹出所有 `level >= 新等级` 的项再压入，不假定 Markdown 必须三级或标题等级连续。
- 先按 Markdown 结构切分，再按段落和句子进行长度兜底；列表、表格、引用和 fenced code block 尽量作为完整语义块保留。
- embedding 输入包含标题路径和正文；payload 也保留标题路径、文档标识、section 标识和相邻 chunk 指针。
- 不机械拼接“上一段”。只允许在同一标题区间内按自然语义块合并，避免跨标题污染。
- Qdrant 不规定单个 point 的 token 上限；上限受 embedding 模型输入窗口、召回精度和最终上下文成本共同约束。具体生产阈值必须由评测集确定，不能把讨论中的临时数值当作正式配置。
- V1 仅对每个可检索 chunk 建一个 embedding；命中后按 token 预算读取 `prev_chunk_id` / `next_chunk_id`。不要同时为父子内容大量重复建向量。

## 6. 推荐工作方式

1. 先定位任务属于认证、旧 Chat、RAG，还是未来 Coding Agent；避免跨链路改动。
2. 阅读对应代码与技术设计章节，写出 source -> transform -> sink，明确输入从哪里来、在哪里改变、最终写到哪里。
3. 给出一个可独立 Review 的最小阶段，等待确认。
4. 实现后先验证本阶段，再开始下一模块；不要把 Chunker、Embedding、Qdrant、BM25 和 Chat 接入一次性混改。

## 7. 当前优先级

1. 技术文档 RAG V2 的 Markdown Chunker：标题栈、语义块、长度兜底、邻接元数据和测试。
2. 再做 embedding 写入和新的 Qdrant collection。
3. 再做 BM25、RRF、MMR 与召回上下文组装。
4. RAG 验收后，再实现独立 Coding Agent 的 Provider Gateway、Memory 和状态机。

## 8. 尚未确定的问题

- 最终使用哪一家云端 embedding 服务及其模型输入窗口、维度、价格。
- Chunk 的正式 soft/hard token 阈值，以及 Top-K、RRF、MMR 等参数；必须由评测集结果决定。
- 是否实现父 section 的展示级摘要，以及其是否需要单独 embedding。
- Coding Agent 的工作区隔离、文件写工具权限和实际前端形态。

在这些问题未由用户确认前，只能提出候选方案与验证计划，不得自行固定为生产设计。
