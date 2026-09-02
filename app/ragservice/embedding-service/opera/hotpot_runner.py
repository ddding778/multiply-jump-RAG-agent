"""执行 HotpotQA distractor paragraph 的离线 embedding 与 Qdrant 写入。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag_index.retrieval_config import load_opera_retrieval_config, load_retrieval_algorithm_config

from .embedding_checkpoint import (
    checkpoint_path_for_plan,
    completed_record_count,
    load_or_initialize_checkpoint,
    mark_batch_completed,
    next_uncompleted_start,
    save_checkpoint,
)
from .hotpot_data import (
    HotpotIndexConfig,
    HotpotIndexPlan,
    HotpotIndexRecord,
    build_hotpot_index_plan,
    build_hotpot_payload,
    write_hotpot_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_EMBEDDING_BATCH_SIZE = 20
EMBEDDING_MODEL_MAX_BATCH_SIZES = {
    "qwen3.7-text-embedding": 20,
    "qwen3.7-text-embedding-flash": 20,
    "text-embedding-v4": 10,
    "text-embedding-v3": 10,
    "text-embedding-v2": 25,
    "text-embedding-v1": 25,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 HotpotQA 离线索引命令参数。

    参数 argv 为可选参数列表；返回数据集、输出、Qdrant、embedding 与 dry-run 配置。
    """

    parser = argparse.ArgumentParser(description="Build HotpotQA distractor embeddings and write them to Qdrant.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "docs" / "hotpotQA" / "hotpot_dev_distractor_v1.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "out" / "opera-index")
    parser.add_argument("--retrieval-config", type=Path, default=PROJECT_ROOT / "app" / "ragservice" / "config" / "rag-retrieval.yaml")
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-grpc-port", type=int, default=int(os.getenv("QDRANT_GRPC_PORT", "6334")))
    parser.add_argument("--embedding-base-url", default=os.getenv("RAG_EMBEDDING_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--embedding-dimensions", type=int, default=int(os.getenv("RAG_EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS))))
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only validate/import and write local artifacts; do not call embedding or Qdrant.")
    return parser.parse_args(argv)


def configure_logging() -> logging.Logger:
    """创建不记录问题、段落正文、向量或密钥的索引日志器。"""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("jieba").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logging.getLogger("opera_hotpot_index")


def build_config(args: argparse.Namespace) -> HotpotIndexConfig:
    """根据命令行和共用 YAML 构建 HotpotQA 独立索引合同。

    参数 args 为已解析参数；返回 HotpotIndexConfig。数值非法、YAML 读取失败时抛出 ValueError。
    """

    opera_config = load_opera_retrieval_config(args.retrieval_config)
    algorithm_config = load_retrieval_algorithm_config(args.retrieval_config)
    if args.embedding_dimensions <= 0 or args.embedding_batch_size <= 0:
        raise ValueError("embedding dimensions and batch size must be positive")
    maximum_batch_size = embedding_batch_size_limit(args.embedding_model)
    if maximum_batch_size is not None and args.embedding_batch_size > maximum_batch_size:
        raise ValueError(
            f"embedding batch size {args.embedding_batch_size} exceeds the {args.embedding_model} limit of {maximum_batch_size}"
        )
    return HotpotIndexConfig(
        collection_prefix=opera_config.collection_prefix,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        embedding_batch_size=args.embedding_batch_size,
        bm25_tokenizer_version=algorithm_config.bm25_tokenizer_version,
    )


def embedding_batch_size_limit(embedding_model: str) -> int | None:
    """返回已知 embedding 模型单次请求允许的最大文本条数。

    参数 embedding_model 为 OpenAI-compatible API 使用的模型名；返回正整数上限，
    未内置限制的第三方模型返回 None，由其 provider 在调用时校验。
    """

    return EMBEDDING_MODEL_MAX_BATCH_SIZES.get(embedding_model.strip().lower())


def create_qdrant_client(host: str, grpc_port: int) -> Any:
    """创建 HotpotQA 索引使用的 Qdrant gRPC 客户端。

    参数 host 和 grpc_port 为服务地址；返回 QdrantClient。依赖缺失时抛出 RuntimeError。
    """

    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise RuntimeError("qdrant-client is required for HotpotQA indexing") from error
    return QdrantClient(host=host, grpc_port=grpc_port, prefer_grpc=True)


def ensure_collection(qdrant: Any, plan: HotpotIndexPlan) -> None:
    """创建或校验当前 OPERA collection 的向量维度。

    参数 qdrant 为客户端，plan 为索引计划；无返回值。已存在 collection 的维度不一致时抛出 ValueError。
    """

    from qdrant_client.models import Distance, VectorParams

    if not qdrant.collection_exists(plan.collection_name):
        qdrant.create_collection(
            collection_name=plan.collection_name,
            vectors_config=VectorParams(size=plan.config.embedding_dimensions, distance=Distance.COSINE),
        )
        return
    dimensions = getattr(qdrant.get_collection(plan.collection_name).config.params.vectors, "size", None)
    if dimensions != plan.config.embedding_dimensions:
        raise ValueError("Hotpot collection vector size does not match index plan")


def ensure_payload_indexes(qdrant: Any, collection_name: str) -> None:
    """为版本过滤、case 过滤和 BM25 候选回读创建 Qdrant payload 索引。

    参数 qdrant 为客户端，collection_name 为目标 collection；无返回值。重复调用由 Qdrant 幂等处理。
    """

    from qdrant_client.models import PayloadSchemaType

    for field_name in ("index_version", "case_id", "chunk_id"):
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )


def create_embedding_client(api_key: str, base_url: str) -> Any:
    """创建 HotpotQA 索引使用的 OpenAI-compatible embedding 客户端。"""

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def embed_and_upsert(
    qdrant: Any,
    embedding_client: Any,
    plan: HotpotIndexPlan,
    checkpoint_path: Path,
    logger: logging.Logger,
) -> None:
    """按批 embedding Hotpot paragraph 并同步写入对应 Qdrant collection。

    参数 qdrant、embedding_client、plan、checkpoint_path、logger 分别为外部依赖、计划、恢复状态路径和安全日志器；
    无返回值。Qdrant 成功确认后才会原子更新 checkpoint，批次失败即抛出异常。
    """

    from qdrant_client.models import PointStruct

    checkpoint, checkpoint_existed = load_or_initialize_checkpoint(checkpoint_path, plan)
    if not checkpoint_existed:
        save_checkpoint(checkpoint_path, checkpoint)
    logger.info(
        "event=opera_hotpot_embedding_checkpoint_ready collection=%s index_version=%s checkpoint=%s existed=%s completed_record_count=%d",
        plan.collection_name,
        plan.index_version,
        checkpoint_path,
        checkpoint_existed,
        completed_record_count(checkpoint),
    )

    if checkpoint_existed:
        recovery_start = next_uncompleted_start(checkpoint)
        if recovery_start < len(plan.records):
            recovery_records = plan.records[recovery_start : recovery_start + plan.config.embedding_batch_size]
            if batch_is_already_upserted(qdrant, plan, recovery_records):
                checkpoint = mark_batch_completed(checkpoint, recovery_start, recovery_records)
                save_checkpoint(checkpoint_path, checkpoint)
                logger.info(
                    "event=opera_hotpot_embedding_checkpoint_reconciled collection=%s index_version=%s batch_start=%d batch_count=%d",
                    plan.collection_name,
                    plan.index_version,
                    recovery_start,
                    len(recovery_records),
                )

    for start in range(0, len(plan.records), plan.config.embedding_batch_size):
        if start < next_uncompleted_start(checkpoint):
            continue
        records = plan.records[start : start + plan.config.embedding_batch_size]
        response = embedding_client.embeddings.create(
            model=plan.config.embedding_model,
            input=[record.paragraph.embedding_text for record in records],
            dimensions=plan.config.embedding_dimensions,
        )
        embeddings = _ordered_embeddings(response, len(records))
        qdrant.upsert(
            collection_name=plan.collection_name,
            points=[
                PointStruct(id=record.point_id, vector=embedding, payload=build_hotpot_payload(record, plan.index_version))
                for record, embedding in zip(records, embeddings, strict=True)
            ],
            wait=True,
        )
        checkpoint = mark_batch_completed(checkpoint, start, records)
        save_checkpoint(checkpoint_path, checkpoint)
        logger.info(
            "event=opera_hotpot_embedding_batch_upserted collection=%s index_version=%s batch_start=%d batch_count=%d",
            plan.collection_name,
            plan.index_version,
            start,
            len(records),
        )


def batch_is_already_upserted(qdrant: Any, plan: HotpotIndexPlan, records: tuple[HotpotIndexRecord, ...]) -> bool:
    """按确定性 point ID 回读一个未确认批次，判断其是否已完整写入 Qdrant。

    参数 qdrant 为客户端，plan 为当前索引计划，records 为单个有序批次；
    返回 True 表示每个 point 的索引版本和输入摘要均与计划一致，否则返回 False。函数不扫描整个 collection。
    """

    if not records:
        return False
    recovered_points = qdrant.retrieve(
        collection_name=plan.collection_name,
        ids=[record.point_id for record in records],
        with_payload=["index_version", "embedding_input_hash"],
        with_vectors=False,
    )
    points_by_id = {str(point.id): point for point in recovered_points}
    for record in records:
        point = points_by_id.get(record.point_id)
        payload = getattr(point, "payload", None)
        if not isinstance(payload, dict):
            return False
        if payload.get("index_version") != plan.index_version:
            return False
        if payload.get("embedding_input_hash") != record.embedding_input_hash:
            return False
    return len(points_by_id) == len(records)


def verify_index_count(qdrant: Any, plan: HotpotIndexPlan) -> None:
    """校验当前 OPERA index_version 的 Qdrant point 数与计划完全一致。"""

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    result = qdrant.count(
        collection_name=plan.collection_name,
        count_filter=Filter(must=[FieldCondition(key="index_version", match=MatchValue(value=plan.index_version))]),
        exact=True,
    )
    if result.count != len(plan.records):
        raise RuntimeError(f"Hotpot Qdrant point count mismatch: expected {len(plan.records)}, got {result.count}")


def main(argv: list[str] | None = None) -> int:
    """运行 HotpotQA 离线导入、embedding 与 artifact 写入主流程。

    参数 argv 为可选命令行参数；成功返回 0，失败返回 1。dry-run 不连接 embedding provider 或 Qdrant。
    """

    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    logger = configure_logging()
    try:
        plan = build_hotpot_index_plan(args.dataset, build_config(args))
        logger.info(
            "event=opera_hotpot_index_plan_ready collection=%s index_version=%s dataset_sha256=%s case_count=%d paragraph_count=%d",
            plan.collection_name,
            plan.index_version,
            plan.dataset_sha256,
            len({record.paragraph.case_id for record in plan.records}),
            len(plan.records),
        )
        if args.dry_run:
            manifest_path, bm25_path = write_hotpot_artifacts(plan, args.output_dir)
            logger.info(
                "event=opera_hotpot_dry_run_completed collection=%s index_version=%s manifest=%s bm25_index=%s",
                plan.collection_name,
                plan.index_version,
                manifest_path,
                bm25_path,
            )
            return 0
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for HotpotQA indexing")
        qdrant = create_qdrant_client(args.qdrant_host, args.qdrant_grpc_port)
        ensure_collection(qdrant, plan)
        ensure_payload_indexes(qdrant, plan.collection_name)
        checkpoint_path = checkpoint_path_for_plan(args.output_dir, plan)
        embed_and_upsert(qdrant, create_embedding_client(api_key, args.embedding_base_url), plan, checkpoint_path, logger)
        verify_index_count(qdrant, plan)
        manifest_path, bm25_path = write_hotpot_artifacts(plan, args.output_dir)
        logger.info(
            "event=opera_hotpot_index_completed collection=%s index_version=%s paragraph_count=%d manifest=%s bm25_index=%s",
            plan.collection_name,
            plan.index_version,
            len(plan.records),
            manifest_path,
            bm25_path,
        )
        return 0
    except Exception as error:
        logger.error("event=opera_hotpot_index_failed error_type=%s", type(error).__name__)
        return 1


def _ordered_embeddings(response: Any, expected_count: int) -> list[list[float]]:
    """按 provider 响应 index 恢复 embedding 输入顺序并校验数量。"""

    indexed: dict[int, list[float]] = {}
    for item in response.data:
        if item.index in indexed:
            raise RuntimeError("Hotpot embedding response contains duplicate indexes")
        indexed[item.index] = item.embedding
    if set(indexed) != set(range(expected_count)):
        raise RuntimeError("Hotpot embedding response count does not match request")
    return [indexed[index] for index in range(expected_count)]


if __name__ == "__main__":
    sys.exit(main())
