"""执行 RAG V2 离线 embedding 与 Qdrant 写入。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .bm25_index import Bm25Document, write_bm25_index
from .indexer import IndexConfig, IndexPlan, build_index_plan, build_payload, write_manifest
from .markdown_chunker import TiktokenTokenCounter
from .retrieval_config import DEFAULT_RETRIEVAL_CONFIG_PATH, RetrievalAlgorithmConfig, load_retrieval_algorithm_config


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"
DEFAULT_EMBEDDING_DIMENSIONS = 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析离线索引命令行参数。

    参数 argv 为可选的参数列表；返回包含文档、Qdrant、embedding 与 token 配置的命名空间。
    """

    parser = argparse.ArgumentParser(description="Build RAG V2 embeddings and write them to Qdrant.")
    parser.add_argument("--docs-dir", type=Path, default=PROJECT_ROOT / "docs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "out" / "rag-index")
    parser.add_argument("--retrieval-config", type=Path, default=DEFAULT_RETRIEVAL_CONFIG_PATH)
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-grpc-port", type=int, default=int(os.getenv("QDRANT_GRPC_PORT", "6334")))
    parser.add_argument("--embedding-base-url", default=os.getenv("RAG_EMBEDDING_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--embedding-dimensions", type=int, default=int(os.getenv("RAG_EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS))))
    parser.add_argument("--tokenizer-encoding", default=os.getenv("RAG_TOKENIZER_ENCODING", "cl100k_base"))
    parser.add_argument("--soft-limit-tokens", type=int, default=int(os.getenv("RAG_SOFT_LIMIT_TOKENS", "800")))
    parser.add_argument("--hard-limit-tokens", type=int, default=int(os.getenv("RAG_HARD_LIMIT_TOKENS", "1200")))
    parser.add_argument("--embedding-batch-size", type=int, default=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "10")))
    parser.add_argument("--dry-run", action="store_true", help="Only build and write the manifest; do not call embedding or Qdrant.")
    return parser.parse_args(argv)


def configure_logging() -> logging.Logger:
    """创建仅记录结构化索引元数据的日志器。

    无参数；返回写入标准输出的日志器，不输出原文、向量或任何密钥。
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("jieba").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logging.getLogger("rag_index")


def load_environment() -> None:
    """在可用时加载项目本地 .env 文件。

    无参数；无返回值。缺少 python-dotenv 时继续使用进程环境变量。
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def build_config(args: argparse.Namespace, algorithm_config: RetrievalAlgorithmConfig) -> IndexConfig:
    """从命令行参数生成并校验索引合同。

    参数 args 为解析后的命名空间，algorithm_config 为 YAML 检索合同；返回 IndexConfig。数值参数非法时抛出 ValueError。
    """

    if args.embedding_dimensions <= 0:
        raise ValueError("embedding_dimensions must be positive")
    if args.embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive")
    return IndexConfig(
        collection_prefix=algorithm_config.collection_prefix,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        tokenizer_encoding=args.tokenizer_encoding,
        soft_limit_tokens=args.soft_limit_tokens,
        hard_limit_tokens=args.hard_limit_tokens,
        embedding_batch_size=args.embedding_batch_size,
        bm25_tokenizer_version=algorithm_config.bm25_tokenizer_version,
    )


def create_qdrant_client(host: str, grpc_port: int) -> Any:
    """创建使用 gRPC 的 Qdrant 客户端。

    参数 host 为 Qdrant 主机，grpc_port 为 gRPC 端口；返回 QdrantClient，依赖缺失时抛出 RuntimeError。
    """

    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise RuntimeError("缺少 qdrant-client，请先安装 requirements.txt 中的依赖") from error
    return QdrantClient(host=host, grpc_port=grpc_port, prefer_grpc=True)


def ensure_collection(qdrant: Any, collection_name: str, config: IndexConfig) -> None:
    """创建新 collection 或验证既有 collection 的向量维度。

    参数 qdrant 为客户端，collection_name 为本批独立 collection，config 为索引合同；无返回值。若既有集合维度不一致则抛出 ValueError。
    """

    from qdrant_client.models import Distance, VectorParams

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=config.embedding_dimensions, distance=Distance.COSINE),
        )
        return

    collection = qdrant.get_collection(collection_name)
    vectors = collection.config.params.vectors
    existing_dimensions = getattr(vectors, "size", None)
    if existing_dimensions != config.embedding_dimensions:
        raise ValueError(
            f"collection vector size mismatch: expected {config.embedding_dimensions}, got {existing_dimensions}"
        )


def ensure_payload_indexes(qdrant: Any, collection_name: str) -> None:
    """创建后续版本过滤与 P1 扩展会用到的 payload 索引。

    参数 qdrant 为客户端，collection_name 为目标集合；无返回值。重复调用由 Qdrant 幂等处理。
    """

    from qdrant_client.models import PayloadSchemaType

    for field_name in ("index_version", "document_id", "section_id", "parent_section_id", "chunk_id"):
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )


def create_embedding_client(api_key: str, base_url: str) -> Any:
    """创建 OpenAI-compatible embedding 客户端。

    参数 api_key 为 DashScope 密钥，base_url 为兼容接口根地址；返回 OpenAI 客户端。
    """

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("缺少 openai，请先安装 requirements.txt 中的依赖") from error
    return OpenAI(api_key=api_key, base_url=base_url)


def embed_and_upsert(qdrant: Any, embedding_client: Any, plan: IndexPlan, logger: logging.Logger) -> None:
    """按批调用 embedding 并同步写入 Qdrant。

    参数 qdrant 为客户端，embedding_client 为模型客户端，plan 为索引计划，logger 为安全日志器；
    无返回值。任意批次失败即抛出，调用方不会写 manifest。
    """

    from qdrant_client.models import PointStruct

    batch_size = plan.config.embedding_batch_size
    for start in range(0, len(plan.records), batch_size):
        records = plan.records[start : start + batch_size]
        response = embedding_client.embeddings.create(
            model=plan.config.embedding_model,
            input=[record.chunk.embedding_text for record in records],
            dimensions=plan.config.embedding_dimensions,
        )
        embeddings = _ordered_embeddings(response, len(records))
        points = [
            PointStruct(
                id=record.point_id,
                vector=embedding,
                payload=build_payload(record, plan.index_version),
            )
            for record, embedding in zip(records, embeddings, strict=True)
        ]
        qdrant.upsert(collection_name=plan.collection_name, points=points, wait=True)
        logger.info(
            "embedding_batch_upserted collection=%s index_version=%s batch_start=%d batch_count=%d",
            plan.collection_name,
            plan.index_version,
            start,
            len(records),
        )


def verify_index_count(qdrant: Any, plan: IndexPlan) -> None:
    """验证本批 index_version 的已写入 point 数与计划一致。

    参数 qdrant 为客户端，plan 为索引计划；无返回值。数量不一致时抛出 RuntimeError。
    """

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    result = qdrant.count(
        collection_name=plan.collection_name,
        count_filter=Filter(
            must=[FieldCondition(key="index_version", match=MatchValue(value=plan.index_version))]
        ),
        exact=True,
    )
    if result.count != len(plan.records):
        raise RuntimeError(f"Qdrant point count mismatch: expected {len(plan.records)}, got {result.count}")


def _ordered_embeddings(response: Any, expected_count: int) -> list[list[float]]:
    """按响应 index 恢复 embedding 的原始输入顺序并校验维度数量。

    参数 response 为 OpenAI-compatible embedding 响应，expected_count 为本批记录数；
    返回顺序一致的向量列表，响应缺项、重复或多余时抛出 RuntimeError。
    """

    indexed: dict[int, list[float]] = {}
    for item in response.data:
        if item.index in indexed:
            raise RuntimeError("embedding response contains duplicate indexes")
        indexed[item.index] = item.embedding
    if set(indexed) != set(range(expected_count)):
        raise RuntimeError("embedding response count does not match request")
    return [indexed[index] for index in range(expected_count)]


def write_local_artifacts(plan: IndexPlan, output_root: Path) -> tuple[Path, Path]:
    """写入与已验证索引计划绑定的 manifest 和 BM25 artifact。

    参数 plan 为完整索引计划，output_root 为 `out/rag-index` 根目录；返回 manifest 和 bm25_index 的绝对路径。
    BM25 构建失败时抛出异常，调用方不得将该批次作为可供混合检索的完成版本。
    """

    version_output_dir = output_root / plan.config.collection_prefix
    manifest_path = write_manifest(plan, version_output_dir)
    bm25_path = write_bm25_index(
        documents=[
            Bm25Document(chunk_id=record.chunk.chunk_id, text=record.chunk.embedding_text)
            for record in plan.records
        ],
        output_path=manifest_path.parent / "bm25_index.json",
        index_version=plan.index_version,
        collection_name=plan.collection_name,
        tokenizer_version=plan.config.bm25_tokenizer_version,
    )
    return manifest_path, bm25_path


def main(argv: list[str] | None = None) -> int:
    """运行离线索引主流程。

    参数 argv 为可选参数列表；成功时返回 0，任一步骤失败时返回 1 且不写成功 manifest。
    """

    load_environment()
    args = parse_args(argv)
    logger = configure_logging()
    try:
        algorithm_config = load_retrieval_algorithm_config(args.retrieval_config)
        config = build_config(args, algorithm_config)
        plan = build_index_plan(args.docs_dir, config, TiktokenTokenCounter(config.tokenizer_encoding))
        logger.info(
            "index_plan_ready collection=%s index_version=%s markdown_files_root=%s chunk_count=%d",
            plan.collection_name,
            plan.index_version,
            args.docs_dir,
            len(plan.records),
        )
        if args.dry_run:
            manifest_path, bm25_path = write_local_artifacts(plan, args.output_dir)
            logger.info(
                "dry_run_artifacts_written collection=%s index_version=%s manifest=%s bm25_index=%s chunk_count=%d",
                plan.collection_name,
                plan.index_version,
                manifest_path,
                bm25_path,
                len(plan.records),
            )
            return 0

        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
        if not api_key:
            raise RuntimeError("请设置 DASHSCOPE_API_KEY；为兼容旧配置，也可使用 ALIYUN_API_KEY")
        qdrant = create_qdrant_client(args.qdrant_host, args.qdrant_grpc_port)
        ensure_collection(qdrant, plan.collection_name, config)
        ensure_payload_indexes(qdrant, plan.collection_name)
        embedding_client = create_embedding_client(api_key, args.embedding_base_url)
        embed_and_upsert(qdrant, embedding_client, plan, logger)
        verify_index_count(qdrant, plan)
        manifest_path, bm25_path = write_local_artifacts(plan, args.output_dir)
        logger.info(
            "index_completed collection=%s index_version=%s chunk_count=%d manifest=%s bm25_index=%s",
            plan.collection_name,
            plan.index_version,
            len(plan.records),
            manifest_path,
            bm25_path,
        )
        return 0
    except Exception as error:
        logger.error("index_failed error_type=%s", type(error).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
