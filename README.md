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

## 文档与当前状态

- [RAG 链路与运行合同](docs/develop/rag链路.md)：离线导入、在线接口、配置、评测与排查。
- [技术选型](docs/develop/技术选型.md)：结构化切分、混合检索、版本化索引等选择理由。
- [OPERA 前端交接](docs/develop/OPERA前端交接.md)：前端事件合同与联调边界。

已完成 OPERA 的离线导入、独立索引、请求编排、Schema/Pydantic 校验和 mock 测试。正式的 Hybrid 基线与 OPERA-style 双组质量评测尚未实现，因此仓库不宣称已有可比较的答案质量指标。

## 安全与展示边界

- `.env`、本地语料 `docs/hotpotQA/`、`out/` 及本地服务配置不提交；`docs/develop/*.md` 是受版本控制的开发文档。
- `opera.langfuse.capture_input_output` 默认是 `false`；只有本地且确认数据不敏感时才临时开启原文观测，并在完成后恢复。
- 当前 HTTP 服务面向本机演示，应监听 `127.0.0.1`，不要直接暴露到公网；任何 API key 均不得传给前端。

## 详细本地运行资料与兼容链路

## AI RPC 本地启动

### 输入

- MySQL：本地 `3306` 端口，连接配置位于未提交的 `app/ai/rpc/etc/ai.yaml`。
- Redis：本地 `6379` 端口。
- etcd：本地 `2379` 端口。AI RPC 会在启动时注册服务，未启动 etcd 时 RPC 无法监听。
- DeepSeek 密钥：项目根目录 `.env` 中的 `DEEPSEEK_API_KEY`，该文件已被 Git 忽略。

`.env` 示例：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 启动命令

Go 服务不会自动读取 `.env`。在 PowerShell 中先将 key 注入当前进程环境，再启动 AI RPC：

```powershell
$deepSeekLine = Get-Content .env | Where-Object { $_ -match '^\s*DEEPSEEK_API_KEY\s*=' } | Select-Object -First 1
if ($null -eq $deepSeekLine) { throw 'DEEPSEEK_API_KEY missing from .env' }
$env:DEEPSEEK_API_KEY = ($deepSeekLine -split '=', 2)[1].Trim().Trim('"').Trim("'")
go run .\app\ai\rpc\ai.go -f .\app\ai\rpc\etc\ai.yaml
```

运行环境变量中的 `DEEPSEEK_API_KEY` 会覆盖 `ai.yaml` 的同名配置。命令不会输出密钥；启动成功后，AI RPC 默认监听 `8080`。

## RAG V2 离线索引

当前阶段已实现“Markdown 结构化切分 → Qwen embedding → 版本化 Qdrant collection + BM25 artifact → Dense/BM25 → RRF → MMR → P1 上下文扩展”。rerank 与 RAG V2 `/search` 的 Langfuse 监控尚未实现；OPERA 的独立观测见下文。

## OPERA HotpotQA 离线索引（第一阶段）

### 输入

- 数据集：本地 `docs/hotpotQA/hotpot_dev_distractor_v1.json`。每个 `context` 元素是一个 paragraph；导入时将标题和完整句子列表作为一个 embedding chunk。
- 配置：[app/ragservice/config/rag-retrieval.yaml](app/ragservice/config/rag-retrieval.yaml) 的 `opera.retrieval`。它使用独立 collection 前缀 `hotpot_distractor_v1`，不会混用 Markdown RAG V2 的 `tech_docs_v2` collection。
- 密钥：真实 embedding 仍复用 `.env` 的 `DASHSCOPE_API_KEY`。`--dry-run` 不需要密钥、Qdrant、embedding 或 DeepSeek。

### 输出

- Qdrant：真实运行创建 `hotpot_distractor_v1_<index_version>` 格式的独立 collection；版本值由数据集摘要、paragraph 构建规则、embedding 合同与 BM25 分词规则共同计算。
- 本地 artifact：`out/opera-index/hotpot_distractor_v1/<index_version>/manifest.jsonl` 与 `bm25_index.json`。后者只分词一次，同时保存全量与 `case_id → chunk_id[]` 映射，运行时支持 `all` 全库 BM25 和 `case` 单题 10 段 BM25。
- 恢复状态：真实导入在同目录写入 `embedding-checkpoint.json`。它只保存索引合同、恢复游标和最后一批输入摘要，不保存向量、paragraph 正文或密钥。

### 运行

先执行不会产生 API 成本的本地 dry-run：

```powershell
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m opera.hotpot_runner --dry-run
```

确认 manifest 与 BM25 artifact 后，真实 embedding 命令与 V2 相同地读取 `.env` 和 Qdrant：

```powershell
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m opera.hotpot_runner
```

`qwen3.7-text-embedding` 的默认单次请求批量为 20（provider 上限）；启动后先写 checkpoint，再在每个 `upsert(wait=True)` 成功后原子推进。网络或进程中断时，使用相同命令自动从 checkpoint 恢复；若 provider 已响应但本地未收到响应，该批可能被重新调用，无法保证绝对不重复计费。当前 Hotpot 导入使用新的索引合同版本，因而创建新 collection，不复用历史 partial collection。

已完成 `POST /opera/ask`、串行 `ExecutionState`、Planner、Analysis-Answer、Rewrite 及 DeepSeek Responses JSON Schema/Pydantic 双校验的本地实现和 mock 测试。OPERA 已接入 Langfuse：每个请求、子 Agent、Hybrid Retriever 和 Responses generation 会形成嵌套 trace；本地 `out/opera-traces/<run_id>.json` 额外保存每步实际检索 paragraph、结构化 Agent 结果与安全的 provider 状态诊断。2026-09-01 已用 DashScope 将 HotpotQA dev distractor 的 7,405 个 case、73,700 个 paragraph 全量 embedding，并写入独立 Qdrant collection；当前可用版本为 `f2745236b78afab54b6bd9a2055588e77b9e2339a3198484ea3355f81c8c4484`。2026-09-02 已完成真实 DeepSeek `case` scope smoke test；当前模型显式使用 `reasoning.effort=low`，Planner/Analysis 的上限为 `2000`，Rewrite 为 `3000`。真实调用可返回 `completed`、`insufficient` 或 provider 不完整输出；HotpotQA Hybrid 基线 vs OPERA-style 的正式双组评测器尚未实现，因此暂无可报告的质量指标。

### OPERA 在线运行与 `POST /opera/ask`

#### 输入与前置条件

`POST /opera/ask` 与 Markdown RAG V2 共用同一个 FastAPI 进程，但使用独立的 HotpotQA collection 与 artifact。请求前需要同时满足以下条件：

| 输入 / 依赖 | 用途 |
| --- | --- |
| `.env` 的 `OPERA_INDEX_VERSION` | 指向一次成功的 HotpotQA 导入版本；服务据此读取 `hotpot_distractor_v1_<index_version>` 和对应 `out/opera-index/.../bm25_index.json`。 |
| `.env` 的 `DASHSCOPE_API_KEY`（或兼容变量 `ALIYUN_API_KEY`） | 为每个检索 query 生成 embedding。 |
| `.env` 的 `DEEPSEEK_API_KEY` | Planner、Analysis-Answer、Rewrite 通过 DeepSeek Responses 生成受 Schema 约束的结果。 |
| Qdrant `localhost:6334` | 保存与检索 HotpotQA paragraph 向量。 |

`case` 是评测模式：必须传 HotpotQA 的 `_id`，检索只在该题的候选 paragraph 中竞争。`all` 是面向普通提问的全库模式：不传 `case_id`，在所有已导入 paragraph 中检索。两种模式的指标不能混报。

#### 运行与 HTTP 合同

服务启动命令与 `/search` 相同：

```powershell
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m uvicorn main:app --host 127.0.0.1 --port 8082
```

全库提问示例：

```powershell
$body = @{
  question = "<your multi-hop question>"
  retrieval_scope = "all"
  top_k = 3
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8082/opera/ask -ContentType "application/json" -Body $body
```

`question` 最长 1000 个字符；`top_k` 可选，范围为 1 到 10，默认 3。评测时将 `retrieval_scope` 改为 `case` 并附加 `case_id: "<HotpotQA _id>"`；`case` 缺少 ID、或 `all` 携带 ID，均会返回 `400`。

成功响应包含 `run_id`、`status`、`answer`、最终证据的 `chunk_id` 与 `sentence_index`，以及 `completed_step_count`。最终回答仅由 Planner 标记 `is_final=true` 的最后子目标产出；系统没有额外的 Final Agent。索引、provider 或 Qdrant 不可用时，接口返回 `503`，不会退化为未经检索的回答；当前 provider 返回空/不完整文本会进入 Agent 输出校验并返回 `400`，该错误分类仍待后续修正。

### OPERA Langfuse prompt 与成本观测

#### 输入

- `.env`：`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL`。密钥只保留在本地 `.env`，不要写入 YAML、日志或 Git。
- 配置：[app/ragservice/config/rag-retrieval.yaml](app/ragservice/config/rag-retrieval.yaml) 的 `opera.langfuse`。当前为 `enabled: true`、读取 `production` 标签、缓存 300 秒、`capture_input_output: false`。
- 本地 fallback：`app/ragservice/embedding-service/opera/prompts/*.md`。三份内容均是没有变量的 `text` prompt。

#### 输出

- Langfuse prompt：`opera-planner-system`、`opera-analysis-answer-system`、`opera-rewrite-system` 的 `production` 版本；每次同步会创建同名的新版本。
- Langfuse trace：`opera-ask` 根 Agent，下含 `opera-planner`、`opera-hybrid-retrieval`、`opera-analysis-answer`、按需的 `opera-rewrite`，以及每次 DeepSeek Responses generation。generation 写入 Schema、prompt 来源/版本、`reasoning_effort`、`max_output_tokens`、输入/缓存/输出 token、耗时和成本；Retriever 输出包含实际 paragraph 与候选数/耗时摘要。
- 隐私：默认不向 Langfuse 上传问题、检索 query、paragraph 正文或 Agent 输出，只保留字符数、范围、候选数、token 与错误类型；不会上传 API secret。仅本地且确认数据不敏感的排障场景可临时将 `capture_input_output` 改为 `true`，并在完成后恢复为 `false`。

首次或需要将本地修改发布到 Langfuse 时，显式运行同步命令：

```powershell
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m opera.prompt_sync
```

运行服务后，`/opera/ask` 会优先读取对应 `production` prompt；远端不可达、凭据缺失或远端 prompt 无效时会记录不含原文的 `opera_prompt_fallback` 事件并继续使用本地文件。OPERA 当前模型为 YAML 的 `deepseek-v4-flash`；Langfuse 使用 `(?i)^deepseek-v4-flash$` 精确匹配该模型，并以高峰价格作为成本上界。价格键与代码上报的互斥 token bucket 一一对应：`input=0.00000044637`、`cache_read_input_tokens=0.000000014879`、`output=0.00000133911`（USD/token）。它们由 DeepSeek 高峰价 `3.0`、`0.10`、`9.0` 元/百万 token，按 `2026-08-28` 的 `1 CNY = 0.14879 USD` 换算；DeepSeek 价格或汇率变化时，需要更新 Langfuse 模型定义，已产生的 trace 不会回填成本。

## RAG V2：输入、输出与运行

### 输入

- 文档：需要切分和索引的 Markdown 应放在项目根目录 `docs/` 下；离线索引器递归匹配 `docs/**/*.md`。仅 `docs/develop/*.md` 作为开发文档提交，HotpotQA 与其他本地语料仍被 Git 忽略；不会索引 `app/` 源码或 `.txt` 文件。
- HotpotQA：`docs/hotpotQA/*.json` 是本地多跳 RAG 评测数据，不属于 Markdown 技术文档索引输入；后续由独立的 Hotpot 导入链路读取。
- embedding：项目根目录未提交的 `.env` 中的 `DASHSCOPE_API_KEY`；旧变量名 `ALIYUN_API_KEY` 仅作兼容回退。
- 在线版本：`.env` 中必须设置 `RAG_INDEX_VERSION`，其值必须是一次成功 embedding 的 manifest 内 `index_version`。
- 检索参数：[app/ragservice/config/rag-retrieval.yaml](app/ragservice/config/rag-retrieval.yaml)，不包含密钥。离线构建和在线服务读取同一份文件。
- 向量库：本地 Qdrant gRPC `localhost:6334`。Docker 配置在 [deploy/rag/docker-compose-qdrant.yaml](deploy/rag/docker-compose-qdrant.yaml)。

`.env` 示例：

```dotenv
DASHSCOPE_API_KEY=your_dashscope_api_key
RAG_INDEX_VERSION=your_successful_index_version
```

### 输出

- Qdrant collection：每个 `index_version` 创建独立的 `tech_docs_v2_<index_version>`；不会覆盖或删除旧 `tech_docs`、旧 `tech_docs_v2` 或其他版本 collection。
- 本地 manifest：`out/rag-index/tech_docs_v2/<index_version>/manifest.jsonl`；BM25 artifact：同目录下的 `bm25_index.json`。正常索引仅在 embedding 全部写入、Qdrant 数量核对成功后生成二者；`--dry-run` 会生成不含向量的两份预览产物；`out/` 已被 Git 忽略。
- 每个 point 的 payload 含 `document_id`、`section_id`、`parent_section_id`、`chunk_id`、`chunk_order`、完整 `heading_path`、源行区间、内容哈希、embedding 输入哈希、前后邻居、`has_code` 与 `index_version`。

### 前置条件

使用 Conda 环境 `aiChatRAG`。若该环境缺少依赖，安装到该环境，不要安装到系统 Python：

```powershell
conda run -n aiChatRAG python -m pip install -r .\app\ragservice\embedding-service\requirements.txt
docker compose -f .\deploy\rag\docker-compose-qdrant.yaml up -d
```

Qdrant 的 Docker named volume 用于中间件持久化，不会写入 `out/`。

### 默认配置与参数

| 配置项 / 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 无，必填 | 百炼 OpenAI-compatible API key；优先于 `ALIYUN_API_KEY`。 |
| `RAG_INDEX_VERSION` | 无，必填 | 在线查询唯一允许使用的已完成索引批次；缺失时 `/search` 返回 503，并记录 `configured=false`。 |
| `--docs-dir` | `<repo>/docs` | 待索引 Markdown 根目录。 |
| `--output-dir` | `<repo>/out/rag-index` | 版本化 manifest 与 BM25 artifact 的根目录。 |
| `--retrieval-config` | `app/ragservice/config/rag-retrieval.yaml` | 离线构建与在线检索共用的非敏感 YAML 合同。 |
| `--qdrant-host` / `QDRANT_HOST` | `localhost` | Qdrant 主机。 |
| `--qdrant-grpc-port` / `QDRANT_GRPC_PORT` | `6334` | Qdrant gRPC 端口。 |
| `--embedding-base-url` / `RAG_EMBEDDING_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 百炼 OpenAI-compatible base URL。 |
| `--embedding-model` / `RAG_EMBEDDING_MODEL` | `qwen3.7-text-embedding` | embedding 模型名称。 |
| `--embedding-dimensions` / `RAG_EMBEDDING_DIMENSIONS` | `1024` | Qdrant collection 和 embedding 返回向量维度。 |
| `--tokenizer-encoding` / `RAG_TOKENIZER_ENCODING` | `cl100k_base` | 本地切分时的 token 估算编码。 |
| `--soft-limit-tokens` / `RAG_SOFT_LIMIT_TOKENS` | `800` | 优先在自然边界切分的目标阈值。 |
| `--hard-limit-tokens` / `RAG_HARD_LIMIT_TOKENS` | `1200` | 单个“标题路径 + 正文”embedding 输入的切分硬上限。 |
| `--embedding-batch-size` / `RAG_EMBEDDING_BATCH_SIZE` | `10` | 单次 embedding API 请求的 chunk 数。 |
| `--dry-run` | 关闭 | 只切分并写 manifest，不调用 embedding API 或 Qdrant。 |
| `RAG_DENSE_TOP_K_MAX` | `10` | `/search` 可接受的最大 Dense 核心命中数。 |
| `RAG_P1_NO_EXPAND_THRESHOLD` | `800` | 核心命中达到该 token 数时不做 P1 扩展。 |
| `RAG_P1_CONTEXT_TOKEN_BUDGET` | `1200` | 每个核心命中连同 P1 扩展可使用的总 token 预算。 |

`rag-retrieval.yaml` 的参数如下：

| YAML 字段 | 当前值 | 作用 |
| --- | ---: | --- |
| `collection_prefix` | `tech_docs_v2` | 与完整 `index_version` 拼接为独立 Qdrant collection 名称。 |
| `bm25.candidate_top_k` | `15` | Dense 与 BM25 各自参与 RRF 的候选数，必须不小于 `RAG_DENSE_TOP_K_MAX`。 |
| `bm25.tokenizer_version` | `jieba-identifier-v1` | 中文 `jieba` 加标识符拆词合同；变更后必须重新构建索引。 |
| `rrf.rank_constant` | `20` | RRF 排名常数。 |
| `rrf.dense_weight` | `1.0` | Dense 排名在加权 RRF 中的贡献。 |
| `rrf.bm25_weight` | `0.005` | BM25 排名在加权 RRF 中的贡献；当前由本地校准集调优。 |
| `mmr.lambda` | `0.7` | MMR 中相关性权重，剩余权重用于抑制候选间重复。 |

修改 `collection_prefix` 或 `bm25.tokenizer_version` 后必须重新完整构建索引；仅修改候选数、RRF 权重、RRF 常数或 MMR 权重时，只需重启服务并重新评测。

`cl100k_base` 不是 Qwen 的官方 tokenizer，因此 800/1200 是稳定的本地工程估算，而不是服务端精确计数；应在后续标注集评测和真实 API 运行中校准。

### 运行

先在不访问外部服务的情况下验证切分和 manifest：

```powershell
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m rag_index.offline_runner --dry-run
```

确认 dry-run 的 manifest 与 `bm25_index.json` 后，执行真实 embedding 与 Qdrant 写入：

```powershell
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m rag_index.offline_runner
```

执行成功后，日志只输出 collection、`index_version`、批次范围、chunk 数和 artifact 路径，不输出 API key、查询正文、文档正文或向量内容。

### 在线混合检索与 P1 扩展

在线服务保持原有 `POST /search` 合同：请求为 `{"query":"...","top_k":3}`，响应为 `{"documents":["..."]}`，因此现有 Go Chat 客户端不需要修改。

检索时先以 `RAG_INDEX_VERSION` 推导唯一 collection，并校验 Qdrant 与 `bm25_index.json` 的版本、collection、分词合同和完整 `chunk_id` 集合一致。随后将用户问题向量化，取得 Dense Top-N；BM25 同时取得 Top-N；加权 RRF 合并 `chunk_id` 排名，MMR 读取融合候选的 Qdrant 向量去除高度重复项，最后才执行 P1。若核心命中 `token_count >= 800` 则不扩展，否则只读取顺序中实际存在的前一块和后一块，连同核心最多三块。候选只有同 `section_id` 或同文档、同非空 `parent_section_id` 才能加入，且核心上下文组不能超过 1200 token。文档开头或结尾不存在的顺序不会触发 Qdrant 候选查询；不会跨父标题，也不会使用 rerank。

从 `manifest.jsonl` 的首行读取成功索引版本并写入 `.env` 后，启动服务：

```powershell
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m uvicorn main:app --host 127.0.0.1 --port 8082
```

未设置 `RAG_INDEX_VERSION` 时，`/search` 会返回 503，并写入 `event=rag_index_version_selected configured=false selection=none`；服务不会猜测“最新”版本，避免混用不同 embedding 或切分批次。

### Dense 基线与完整链路评测

评测命令只比较两组：朴素 `Dense` 基线与完整 `Dense + BM25 → RRF → MMR` 链路。每个问题只生成一次 query embedding，再复用给两组结果；P1 是命中后的上下文扩展，不参与核心召回排名指标。

```powershell
$env:RAG_INDEX_VERSION = "<successful_index_version>"
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m rag_index.evaluation `
  --eval-set <path-to-your-eval-set.jsonl> `
  --top-k-max 10
```

输入 JSONL 每条至少需要 `case_id`、`query`，以及 `target_chunk_id` 或 `relevant_chunks[].chunk_id`。评测结果写入 `out/rag-eval/<index_version>/<dataset>-dense-vs-hybrid.json`，包含 `HitRate@1/3/5/10`、`MRR@10`、`stress_type` 分组、评测集 SHA-256 与本次 RRF/MMR 配置；不写入 query、文档正文、向量或密钥。

当前 24 题校准集上，Dense 基线的 `MRR@10` 为 `0.8507`，完整 Hybrid 链路为 `0.8542`；`HitRate@3` 从 `0.8750` 提升到 `0.9167`。该集合已参与调参，只能作为参数校准依据；上线前应使用未参与调参的保留集复验泛化效果。
