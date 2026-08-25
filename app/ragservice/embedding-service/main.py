"""RAG V2 在线 Dense 检索服务入口。"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from rag_index.bm25_index import Bm25IndexError, load_bm25_index
from rag_index.retrieval import (
    DenseRetrievalService,
    RetrievalConfigurationError,
    RetrievalDataError,
    RetrievalRequestError,
    RetrievalSettings,
    RetrievalUnavailableError,
)
from rag_index.retrieval_config import RetrievalConfigError, collection_name_for_version, load_retrieval_algorithm_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_TOP_K = 3
DEFAULT_TOP_K_MAX = 10
DEFAULT_QUERY_MAX_CHARACTERS = 1000

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="RAG V2 Service")
logger = logging.getLogger("rag_service")
logging.getLogger("jieba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
_service: DenseRetrievalService | None = None
_service_lock = threading.Lock()


class SearchRequest(BaseModel):
    """定义兼容现有 Go 客户端的在线检索请求。

    参数 query 为用户技术问题，top_k 为 Dense 核心命中数；返回对象由 FastAPI 从 JSON 请求体构建。
    """

    query: str
    top_k: int = DEFAULT_TOP_K


class SearchResponse(BaseModel):
    """定义兼容现有 Go 客户端的在线检索响应。

    参数 documents 为混合检索核心命中及其 P1 扩展渲染后的参考资料文本；
    返回对象不包含向量、密钥或内部 Qdrant point ID。
    """

    documents: list[str]


def _positive_int_from_environment(name: str, default: int) -> int:
    """读取并校验正整数环境变量。

    参数 name 为环境变量名，default 为变量缺失时的默认值；返回正整数，格式错误时抛出 RetrievalConfigurationError。
    """

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise RetrievalConfigurationError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise RetrievalConfigurationError(f"{name} must be positive")
    return parsed


def _load_settings() -> RetrievalSettings:
    """加载 V2 在线混合检索配置并强制要求明确 index_version。

    无参数；返回固定的 RetrievalSettings。未配置 RAG_INDEX_VERSION 时记录 configured=false 日志并抛出配置错误。
    """

    index_version = os.getenv("RAG_INDEX_VERSION", "").strip()
    if not index_version:
        logger.error("event=rag_index_version_selected configured=false selection=none")
        raise RetrievalConfigurationError("RAG_INDEX_VERSION is required")
    try:
        algorithm_config = load_retrieval_algorithm_config()
        collection_name = collection_name_for_version(algorithm_config.collection_prefix, index_version)
    except RetrievalConfigError as error:
        raise RetrievalConfigurationError("RAG retrieval YAML config is invalid") from error
    logger.info(
        "event=rag_index_version_selected configured=true selection=explicit index_version=%s",
        index_version,
    )
    settings = RetrievalSettings(
        collection_name=collection_name,
        index_version=index_version,
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_dimensions=_positive_int_from_environment("RAG_EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS),
        dense_top_k_max=_positive_int_from_environment("RAG_DENSE_TOP_K_MAX", DEFAULT_TOP_K_MAX),
        candidate_top_k=algorithm_config.bm25_candidate_top_k,
        rrf_rank_constant=algorithm_config.rrf_rank_constant,
        rrf_dense_weight=algorithm_config.rrf_dense_weight,
        rrf_bm25_weight=algorithm_config.rrf_bm25_weight,
        mmr_lambda=algorithm_config.mmr_lambda,
        p1_no_expand_threshold=_positive_int_from_environment("RAG_P1_NO_EXPAND_THRESHOLD", 800),
        p1_context_token_budget=_positive_int_from_environment("RAG_P1_CONTEXT_TOKEN_BUDGET", 1200),
    )
    if settings.p1_context_token_budget < settings.p1_no_expand_threshold:
        raise RetrievalConfigurationError("RAG_P1_CONTEXT_TOKEN_BUDGET must be at least RAG_P1_NO_EXPAND_THRESHOLD")
    return settings


def _create_service() -> DenseRetrievalService:
    """创建并验证一次运行固定的 embedding、BM25 与 Qdrant 混合检索服务。

    无参数；返回可执行在线搜索的 DenseRetrievalService。缺少 API key 或外部索引不可用时抛出相应错误。
    """

    settings = _load_settings()
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
    if not api_key:
        raise RetrievalConfigurationError("DASHSCOPE_API_KEY is required")
    embedding_client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("RAG_EMBEDDING_BASE_URL", DEFAULT_BASE_URL),
    )
    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        grpc_port=_positive_int_from_environment("QDRANT_GRPC_PORT", 6334),
        prefer_grpc=True,
    )
    try:
        algorithm_config = load_retrieval_algorithm_config()
        bm25_index = load_bm25_index(
            path=PROJECT_ROOT
            / "out"
            / "rag-index"
            / algorithm_config.collection_prefix
            / settings.index_version
            / "bm25_index.json",
            expected_index_version=settings.index_version,
            expected_collection_name=settings.collection_name,
            expected_tokenizer_version=algorithm_config.bm25_tokenizer_version,
        )
    except (RetrievalConfigError, Bm25IndexError) as error:
        raise RetrievalUnavailableError("BM25 index is unavailable") from error
    return DenseRetrievalService(
        embedding_client=embedding_client,
        qdrant=qdrant,
        bm25_index=bm25_index,
        settings=settings,
        logger=logger,
    )


def _get_service() -> DenseRetrievalService:
    """延迟创建并复用线程安全的在线检索服务实例。

    无参数；返回进程内唯一的 DenseRetrievalService。首次请求时会校验 index_version 与 Qdrant 状态。
    """

    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            _service = _create_service()
    return _service


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """处理用户问题并返回当前版本的 Dense、BM25、RRF、MMR 与 P1 参考资料。

    参数 req 为包含 query 与 top_k 的 HTTP 请求；返回与旧 Go 客户端兼容的 documents 列表。
    配置缺失、索引不可用或输入非法时返回不泄露内部细节的 HTTP 错误。
    """

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    if len(query) > DEFAULT_QUERY_MAX_CHARACTERS:
        raise HTTPException(status_code=400, detail="query is too long")
    try:
        documents = _get_service().search(query=query, top_k=req.top_k)
        return SearchResponse(documents=documents)
    except RetrievalRequestError as error:
        logger.info("event=rag_search_rejected error_type=%s", type(error).__name__)
        raise HTTPException(status_code=400, detail="top_k is outside the configured range") from error
    except RetrievalConfigurationError as error:
        logger.error("event=rag_search_rejected error_type=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="RAG_INDEX_VERSION must be configured") from error
    except RetrievalUnavailableError as error:
        logger.error("event=rag_search_unavailable error_type=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="RAG service is unavailable") from error
    except RetrievalDataError as error:
        logger.error("event=rag_search_unavailable error_type=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="RAG index data is unavailable") from error


@app.get("/health")
async def health() -> dict[str, str]:
    """返回在线服务进程健康状态。

    无参数；返回进程可达状态。该接口不读取用户问题，也不触发 embedding 或 Qdrant 查询。
    """

    return {"status": "ok", "service": "rag-v2"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8082)
