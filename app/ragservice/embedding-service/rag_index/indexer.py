"""RAG V2 离线索引计划、payload 与 manifest 的纯本地构建逻辑。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .markdown_chunker import MarkdownChunker, TokenCounter
from .models import MarkdownChunk


CHUNKER_VERSION = "markdown-structure-v1"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexConfig:
    """保存一次离线索引的确定性合同。

    参数包含 collection 前缀、embedding 模型和维度、切分与 BM25 分词合同、请求批大小；
    返回实例同时用于构建 index_version 和运行时校验。
    """

    collection_prefix: str
    embedding_model: str
    embedding_dimensions: int
    tokenizer_encoding: str
    soft_limit_tokens: int
    hard_limit_tokens: int
    embedding_batch_size: int
    bm25_tokenizer_version: str


@dataclass(frozen=True)
class IndexRecord:
    """保存一个可写入 Qdrant 的 chunk 及其稳定 point ID。

    参数 chunk 为结构化切分结果，source_path 为 docs 根目录下的相对路径，
    point_id 为 Qdrant UUID，embedding_input_hash 用于判断 embedding 输入是否变化。
    """

    chunk: MarkdownChunk
    source_path: str
    point_id: str
    embedding_input_hash: str


@dataclass(frozen=True)
class IndexPlan:
    """表示一批尚未写入 Qdrant 的完整索引计划。

    参数 config 为索引合同，records 为有序 chunk，index_version 为本批次的稳定版本，collection_name
    为由前缀和完整版本派生的 Qdrant collection；
    返回对象可用于 embedding、upsert 与 manifest 落盘。
    """

    config: IndexConfig
    records: list[IndexRecord]
    index_version: str
    collection_name: str


def build_index_plan(docs_dir: Path, config: IndexConfig, token_counter: TokenCounter) -> IndexPlan:
    """递归切分 docs 下的 Markdown 文件并构建确定性索引计划。

    参数 docs_dir 为仅允许读取的文档根目录，config 为索引合同，token_counter 为 token 计数器；
    返回按相对路径和 chunk 顺序排序的 IndexPlan。目录不存在或不是目录时抛出 ValueError。
    """

    if not docs_dir.is_dir():
        raise ValueError(f"docs directory does not exist or is not a directory: {docs_dir}")

    chunker = MarkdownChunker(
        token_counter=token_counter,
        soft_limit_tokens=config.soft_limit_tokens,
        hard_limit_tokens=config.hard_limit_tokens,
    )
    records: list[IndexRecord] = []
    for file_path in sorted(docs_dir.rglob("*.md"), key=lambda item: item.relative_to(docs_dir).as_posix()):
        source_path = file_path.relative_to(docs_dir).as_posix()
        for chunk in chunker.chunk_file(file_path, docs_dir):
            records.append(
                IndexRecord(
                    chunk=chunk,
                    source_path=source_path,
                    point_id=_point_id(chunk.chunk_id),
                    embedding_input_hash=_sha256(chunk.embedding_text),
                )
            )

    _validate_unique_ids(records)
    index_version = _build_index_version(config, records)
    from .retrieval_config import collection_name_for_version

    collection_name = collection_name_for_version(config.collection_prefix, index_version)
    return IndexPlan(config=config, records=records, index_version=index_version, collection_name=collection_name)


def build_payload(record: IndexRecord, index_version: str) -> dict[str, object]:
    """生成单条 Qdrant point 的 payload。

    参数 record 为索引计划记录，index_version 为本次完整构建版本；
    返回只包含检索、上下文扩展和可追溯所需元数据的字典，不包含向量与 API key。
    """

    chunk = record.chunk
    return {
        "record_type": "chunk",
        "source_path": record.source_path,
        "document_id": chunk.document_id,
        "section_id": chunk.section_id,
        "parent_section_id": chunk.parent_section_id,
        "chunk_id": chunk.chunk_id,
        "chunk_order": chunk.chunk_order,
        "heading_path": [heading.as_dict() for heading in chunk.heading_path],
        "text": chunk.text,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "content_hash": chunk.content_hash,
        "embedding_input_hash": record.embedding_input_hash,
        "token_count": chunk.token_count,
        "previous_chunk_id": chunk.previous_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "has_code": chunk.has_code,
        "index_version": index_version,
    }


def write_manifest(plan: IndexPlan, output_dir: Path) -> Path:
    """将已验证索引计划原子写入本地 JSONL manifest。

    参数 plan 为已完成写入校验的索引计划，output_dir 为本次任务的 out 目录；
    返回 manifest 绝对路径。每个 index_version 使用独立子目录，避免不同批次互相覆盖；
    文件先写临时路径后替换，避免生成半截 manifest。
    """

    version_output_dir = output_dir / plan.index_version
    version_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = version_output_dir / "manifest.jsonl"
    metadata = {
        "record_type": "manifest_metadata",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "index_version": plan.index_version,
        "chunker_version": CHUNKER_VERSION,
        "collection_name": plan.collection_name,
        "collection_prefix": plan.config.collection_prefix,
        "embedding_model": plan.config.embedding_model,
        "embedding_dimensions": plan.config.embedding_dimensions,
        "tokenizer_encoding": plan.config.tokenizer_encoding,
        "soft_limit_tokens": plan.config.soft_limit_tokens,
        "hard_limit_tokens": plan.config.hard_limit_tokens,
        "bm25_tokenizer_version": plan.config.bm25_tokenizer_version,
        "chunk_count": len(plan.records),
    }

    temporary_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=version_output_dir,
        prefix="manifest-",
        suffix=".tmp",
        delete=False,
    )
    try:
        with temporary_file as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
            for record in plan.records:
                stream.write(
                    json.dumps(
                        {
                            "record_type": "chunk",
                            "point_id": record.point_id,
                            "payload": build_payload(record, plan.index_version),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(temporary_file.name, manifest_path)
    except Exception:
        # 失败时保留临时文件，避免本程序执行删除操作，也便于人工排查。
        raise
    return manifest_path


def _build_index_version(config: IndexConfig, records: list[IndexRecord]) -> str:
    """根据索引合同及所有 embedding 输入生成稳定版本。

    参数 config 为索引合同，records 为有序记录；返回 SHA-256 十六进制版本号。
    """

    contract = {
        "chunker_version": CHUNKER_VERSION,
        "collection_prefix": config.collection_prefix,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "tokenizer_encoding": config.tokenizer_encoding,
        "soft_limit_tokens": config.soft_limit_tokens,
        "hard_limit_tokens": config.hard_limit_tokens,
        "bm25_tokenizer_version": config.bm25_tokenizer_version,
        "records": [
            {
                "source_path": record.source_path,
                "chunk_id": record.chunk.chunk_id,
                "embedding_input_hash": record.embedding_input_hash,
            }
            for record in records
        ],
    }
    return _sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _point_id(chunk_id: str) -> str:
    """将业务 chunk ID 映射为 Qdrant 使用的确定性 UUID。

    参数 chunk_id 为 SHA-256 业务标识；返回 UUIDv5 字符串，重复构建会得到同一点 ID。
    """

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ai-chat:tech-docs:{chunk_id}"))


def _sha256(value: str) -> str:
    """计算 UTF-8 字符串的 SHA-256。

    参数 value 为待哈希文本；返回小写十六进制摘要。
    """

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_unique_ids(records: list[IndexRecord]) -> None:
    """检查业务 chunk ID 和 Qdrant point ID 没有冲突。

    参数 records 为一次构建的所有记录；若同一 ID 对应不同来源或 embedding 输入则抛出 ValueError。
    """

    seen_chunk_ids: dict[str, tuple[str, str]] = {}
    seen_point_ids: dict[str, str] = {}
    for record in records:
        identity = (record.source_path, record.embedding_input_hash)
        existing_identity = seen_chunk_ids.setdefault(record.chunk.chunk_id, identity)
        if existing_identity != identity:
            raise ValueError("chunk_id collision detected")
        existing_chunk_id = seen_point_ids.setdefault(record.point_id, record.chunk.chunk_id)
        if existing_chunk_id != record.chunk.chunk_id:
            raise ValueError("Qdrant point_id collision detected")
