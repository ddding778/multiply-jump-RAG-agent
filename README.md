# 第一个学习go-zero的项目
- 参考了采用了looklook的整体架构，RAG 开发文档见 [docs/develop/rag链路.md](docs/develop/rag链路.md)。
- 前端页面还没调试，只是ai做的初版前端

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

已完成 `POST /opera/ask`、串行 `ExecutionState`、Planner、Analysis-Answer、Rewrite 及 DeepSeek Responses JSON Schema/Pydantic 双校验的本地实现和 mock 测试。OPERA 已接入 Langfuse：每个请求、子 Agent、Hybrid Retriever 和 Responses generation 会形成嵌套 trace；真实 DeepSeek、Qdrant 与 embedding 联调及 OPERA 双组评测仍在后续阶段。

### OPERA Langfuse prompt 与成本观测

#### 输入

- `.env`：`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL`。密钥只保留在本地 `.env`，不要写入 YAML、日志或 Git。
- 配置：[app/ragservice/config/rag-retrieval.yaml](app/ragservice/config/rag-retrieval.yaml) 的 `opera.langfuse`。默认 `enabled: true`、读取 `production` 标签、缓存 300 秒、`capture_input_output: false`。
- 本地 fallback：`app/ragservice/embedding-service/opera/prompts/*.md`。三份内容均是没有变量的 `text` prompt。

#### 输出

- Langfuse prompt：`opera-planner-system`、`opera-analysis-answer-system`、`opera-rewrite-system` 的 `production` 版本；每次同步会创建同名的新版本。
- Langfuse trace：`opera-ask` 根 Agent，下含 `opera-planner`、`opera-hybrid-retrieval`、`opera-analysis-answer`、按需的 `opera-rewrite`，以及每次 DeepSeek Responses generation。generation 写入模型名、Schema、prompt 来源/版本、输入/缓存/输出 token 和耗时。
- 隐私：默认只上传字符数、范围、候选数、token 和错误类型；不上传问题、检索 query、paragraph 正文、模型原始输出或密钥。只有显式将 `capture_input_output` 改为 `true` 才上传原文。

首次或需要将本地修改发布到 Langfuse 时，显式运行同步命令：

```powershell
Set-Location .\app\ragservice\embedding-service
conda run -n aiChatRAG python -m opera.prompt_sync
```

运行服务后，`/opera/ask` 会优先读取对应 `production` prompt；远端不可达、凭据缺失或远端 prompt 无效时会记录不含原文的 `opera_prompt_fallback` 事件并继续使用本地文件。为让 Langfuse 按真实价格自动计算 DeepSeek 成本，需要在 Langfuse 项目中为 YAML 的 `deepseek-chat` 模型配置与 `input`、`cache_read_input_tokens`、`output` 对应的模型价格。

### 输入

- 文档：需要切分和索引的 Markdown 应放在项目根目录 `docs/` 下；离线索引器递归匹配 `docs/**/*.md`。`docs/` 是本地语料目录，当前被 Git 忽略；不会索引 `app/` 源码或 `.txt` 文件。
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
