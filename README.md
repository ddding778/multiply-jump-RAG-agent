# OPERA-style Multi-Agent Multi-hop RAG

这是一个用于面试展示的 Go / go-zero + Python RAG 项目：重点是基于 HotpotQA 的 OPERA-style 多跳检索闭环，同时保留可独立运行的技术文档 RAG V2、用户认证和旧 Chat 链路。

## 项目展示重点

- **结构化技术文档 RAG V2**：Markdown 按标题层级切分，使用 Dense + BM25 + RRF + MMR 检索，并按相邻标题关系做 P1 上下文扩展。
- **OPERA-style 多跳 RAG**：Planner 拆解子目标，Hybrid Retriever 给出证据，Analysis-Answer 作答；证据不足时由 Rewrite 改写查询后继续检索。
- **可观测与安全边界**：运行态与持久化 Memory 分离，Langfuse 用于可选观测，公开演示默认不上传问题、正文或模型输出。

## 输入与输出

| 链路 | 输入 | 输出 |
| --- | --- | --- |
| 技术文档 RAG V2 | `docs/**/*.md`、DashScope embedding、Qdrant | 版本化 collection、BM25 artifact、`POST /search` 检索结果 |
| OPERA 多跳 RAG | 本地 HotpotQA paragraph、问题、Qdrant、DashScope、DeepSeek | `POST /opera/ask` 的最终答案、证据定位、`run_id` 与步骤计数 |
| 旧 Chat | 已登录用户、会话与消息 | 持久化会话消息与模型回答；可调用技术文档 RAG |

## OPERA 处理流程

```text
Question
  -> Planner
  -> Hybrid Retriever (Dense + BM25 + RRF + MMR)
  -> Analysis-Answer
  -> insufficient ? Rewrite -> Retriever : Final answer + evidence
```

OPERA 运行时只读取 HotpotQA paragraph，不读取 `answer`、`supporting_facts` 等评测标注；技术文档 RAG V2 与 OPERA 使用独立 collection、artifact 和指标，避免混用。

## 快速演示

前置条件：启动 Qdrant，准备未提交的 `.env` 中的 `DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`、`OPERA_INDEX_VERSION`。首次使用还需先完成 HotpotQA 离线导入。

```powershell
docker compose -f .\deploy\rag\docker-compose-qdrant.yaml up -d
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m uvicorn main:app --host 127.0.0.1 --port 8082
```

```powershell
$body = @{ question = "<your multi-hop question>"; retrieval_scope = "all"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8082/opera/ask -ContentType "application/json" -Body $body
```

`case` scope 仅用于带 HotpotQA `_id` 的评测；普通演示应使用 `all`。响应含 `run_id`、`status`、`answer`、最终证据的 `chunk_id` / `sentence_index` 及完成步骤数。
