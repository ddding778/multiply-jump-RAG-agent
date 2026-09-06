"""保存 HotpotQA 离线 embedding 的可恢复批次状态。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .hotpot_data import HotpotIndexPlan, HotpotIndexRecord


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompletedEmbeddingBatch:
    """表示一个已被 Qdrant 确认接收的 embedding 批次。

    参数 start 与 end_exclusive 标识原始索引计划中的半开区间，record_fingerprint 绑定该区间的稳定记录，
    completed_at 记录确认时间；返回对象不保存 paragraph 原文、向量或密钥。
    """

    start: int
    end_exclusive: int
    record_fingerprint: str
    completed_at: str


@dataclass(frozen=True)
class EmbeddingCheckpoint:
    """表示一个 OPERA 索引版本的本地 embedding 恢复状态。

    参数中的索引合同字段用于拒绝错误复用旧 checkpoint，next_uncompleted_start 是顺序执行的恢复游标，
    last_completed_batch 用于核验最后一次已确认写入；返回对象仅保存可公开的版本和摘要信息，
    不包含 paragraph、向量或模型响应。
    """

    index_version: str
    collection_name: str
    dataset_sha256: str
    embedding_model: str
    embedding_dimensions: int
    total_records: int
    batch_size: int
    next_uncompleted_start: int
    last_completed_batch: CompletedEmbeddingBatch | None


def checkpoint_path_for_plan(output_root: Path, plan: HotpotIndexPlan) -> Path:
    """返回指定 OPERA 索引计划对应的 checkpoint 文件路径。

    参数 output_root 为 `out/opera-index` 根目录，plan 为当前确定性索引计划；
    返回位于当前 collection 前缀和 index_version 目录下的绝对或相对 Path，不执行文件写入。
    """

    return output_root / plan.config.collection_prefix / plan.index_version / "embedding-checkpoint.json"


def load_or_initialize_checkpoint(path: Path, plan: HotpotIndexPlan) -> tuple[EmbeddingCheckpoint, bool]:
    """读取匹配计划的 checkpoint，缺失时创建内存中的初始状态。

    参数 path 为 checkpoint 路径，plan 为当前索引计划；返回 `(checkpoint, existed)`，其中 existed 表示文件是否已存在。
    JSON 损坏、字段缺失或索引合同不匹配时抛出 ValueError，调用方不得继续写入该索引。
    """

    if not path.exists():
        return _new_checkpoint(plan), False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError("Hotpot embedding checkpoint cannot be read") from error
    except json.JSONDecodeError as error:
        raise ValueError("Hotpot embedding checkpoint is not valid JSON") from error
    checkpoint = _checkpoint_from_raw(raw)
    _validate_checkpoint(checkpoint, plan)
    return checkpoint, True


def save_checkpoint(path: Path, checkpoint: EmbeddingCheckpoint) -> None:
    """原子保存 embedding checkpoint，避免进程中断留下半截 JSON。

    参数 path 为目标文件，checkpoint 为已校验状态；无返回值。写入失败时抛出 RuntimeError，
    已创建的临时文件会保留以便人工排查，函数不执行删除操作。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_file = Path(file.name)
            json.dump(_checkpoint_to_raw(checkpoint), file, ensure_ascii=False, sort_keys=True, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_file, path)
    except OSError as error:
        raise RuntimeError("Hotpot embedding checkpoint cannot be written") from error


def mark_batch_completed(checkpoint: EmbeddingCheckpoint, start: int, records: tuple[HotpotIndexRecord, ...]) -> EmbeddingCheckpoint:
    """在 Qdrant 成功写入后返回带有该完成批次的新 checkpoint。

    参数 checkpoint 为当前状态，start 为批次起点，records 为该批有序索引记录；
    返回不可变的新 EmbeddingCheckpoint。批次范围、顺序或已完成状态不合法时抛出 ValueError。
    """

    expected_end = min(start + checkpoint.batch_size, checkpoint.total_records)
    if start < 0 or start >= checkpoint.total_records or start != next_uncompleted_start(checkpoint):
        raise ValueError("Hotpot embedding checkpoint batch start is invalid")
    if len(records) != expected_end - start:
        raise ValueError("Hotpot embedding checkpoint batch record count is invalid")
    completed_batch = CompletedEmbeddingBatch(
        start=start,
        end_exclusive=expected_end,
        record_fingerprint=_records_fingerprint(records),
        completed_at=datetime.now(UTC).isoformat(),
    )
    return EmbeddingCheckpoint(
        index_version=checkpoint.index_version,
        collection_name=checkpoint.collection_name,
        dataset_sha256=checkpoint.dataset_sha256,
        embedding_model=checkpoint.embedding_model,
        embedding_dimensions=checkpoint.embedding_dimensions,
        total_records=checkpoint.total_records,
        batch_size=checkpoint.batch_size,
        next_uncompleted_start=expected_end,
        last_completed_batch=completed_batch,
    )


def next_uncompleted_start(checkpoint: EmbeddingCheckpoint) -> int:
    """返回当前 checkpoint 按顺序尚未确认的第一个批次起点。

    参数 checkpoint 为已校验的恢复状态；返回范围在 `0..total_records` 的整数，等于 total_records 表示全部完成。
    """

    return checkpoint.next_uncompleted_start


def completed_record_count(checkpoint: EmbeddingCheckpoint) -> int:
    """计算 checkpoint 已确认写入 Qdrant 的 paragraph 数。

    参数 checkpoint 为已校验状态；返回顺序恢复游标之前的记录总数，不访问 Qdrant。
    """

    return checkpoint.next_uncompleted_start


def _new_checkpoint(plan: HotpotIndexPlan) -> EmbeddingCheckpoint:
    """根据确定性索引计划构建尚无完成批次的恢复状态。"""

    return EmbeddingCheckpoint(
        index_version=plan.index_version,
        collection_name=plan.collection_name,
        dataset_sha256=plan.dataset_sha256,
        embedding_model=plan.config.embedding_model,
        embedding_dimensions=plan.config.embedding_dimensions,
        total_records=len(plan.records),
        batch_size=plan.config.embedding_batch_size,
        next_uncompleted_start=0,
        last_completed_batch=None,
    )


def _checkpoint_from_raw(raw: object) -> EmbeddingCheckpoint:
    """将 JSON 对象严格转换为 checkpoint 数据结构。"""

    if not isinstance(raw, dict) or raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Hotpot embedding checkpoint schema version is invalid")
    required_text_fields = ("index_version", "collection_name", "dataset_sha256", "embedding_model")
    if any(not isinstance(raw.get(field_name), str) or not raw[field_name] for field_name in required_text_fields):
        raise ValueError("Hotpot embedding checkpoint text fields are invalid")
    required_number_fields = ("embedding_dimensions", "total_records", "batch_size")
    if any(not isinstance(raw.get(field_name), int) or raw[field_name] <= 0 for field_name in required_number_fields):
        raise ValueError("Hotpot embedding checkpoint numeric fields are invalid")
    if not isinstance(raw.get("next_uncompleted_start"), int) or raw["next_uncompleted_start"] < 0:
        raise ValueError("Hotpot embedding checkpoint next_uncompleted_start is invalid")
    raw_last_completed_batch = raw.get("last_completed_batch")
    last_completed_batch: CompletedEmbeddingBatch | None = None
    if raw_last_completed_batch is not None:
        if not isinstance(raw_last_completed_batch, dict):
            raise ValueError("Hotpot embedding checkpoint last_completed_batch is invalid")
        start = raw_last_completed_batch.get("start")
        end_exclusive = raw_last_completed_batch.get("end_exclusive")
        record_fingerprint = raw_last_completed_batch.get("record_fingerprint")
        completed_at = raw_last_completed_batch.get("completed_at")
        if (
            not isinstance(start, int)
            or not isinstance(end_exclusive, int)
            or not isinstance(record_fingerprint, str)
            or not record_fingerprint
            or not isinstance(completed_at, str)
            or not completed_at
        ):
            raise ValueError("Hotpot embedding checkpoint last_completed_batch fields are invalid")
        last_completed_batch = CompletedEmbeddingBatch(
            start=start,
            end_exclusive=end_exclusive,
            record_fingerprint=record_fingerprint,
            completed_at=completed_at,
        )
    return EmbeddingCheckpoint(
        index_version=raw["index_version"],
        collection_name=raw["collection_name"],
        dataset_sha256=raw["dataset_sha256"],
        embedding_model=raw["embedding_model"],
        embedding_dimensions=raw["embedding_dimensions"],
        total_records=raw["total_records"],
        batch_size=raw["batch_size"],
        next_uncompleted_start=raw["next_uncompleted_start"],
        last_completed_batch=last_completed_batch,
    )


def _checkpoint_to_raw(checkpoint: EmbeddingCheckpoint) -> dict[str, object]:
    """将 checkpoint 转换为稳定排序且不包含敏感内容的 JSON 对象。"""

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "index_version": checkpoint.index_version,
        "collection_name": checkpoint.collection_name,
        "dataset_sha256": checkpoint.dataset_sha256,
        "embedding_model": checkpoint.embedding_model,
        "embedding_dimensions": checkpoint.embedding_dimensions,
        "total_records": checkpoint.total_records,
        "batch_size": checkpoint.batch_size,
        "next_uncompleted_start": checkpoint.next_uncompleted_start,
        "last_completed_batch": (
            None
            if checkpoint.last_completed_batch is None
            else {
                "start": checkpoint.last_completed_batch.start,
                "end_exclusive": checkpoint.last_completed_batch.end_exclusive,
                "record_fingerprint": checkpoint.last_completed_batch.record_fingerprint,
                "completed_at": checkpoint.last_completed_batch.completed_at,
            }
        ),
    }


def _validate_checkpoint(checkpoint: EmbeddingCheckpoint, plan: HotpotIndexPlan) -> None:
    """校验已落盘 checkpoint 与当前索引计划可安全复用。"""

    expected_values = {
        "index_version": plan.index_version,
        "collection_name": plan.collection_name,
        "dataset_sha256": plan.dataset_sha256,
        "embedding_model": plan.config.embedding_model,
        "embedding_dimensions": plan.config.embedding_dimensions,
        "total_records": len(plan.records),
        "batch_size": plan.config.embedding_batch_size,
    }
    actual_values = {
        "index_version": checkpoint.index_version,
        "collection_name": checkpoint.collection_name,
        "dataset_sha256": checkpoint.dataset_sha256,
        "embedding_model": checkpoint.embedding_model,
        "embedding_dimensions": checkpoint.embedding_dimensions,
        "total_records": checkpoint.total_records,
        "batch_size": checkpoint.batch_size,
    }
    if actual_values != expected_values:
        raise ValueError("Hotpot embedding checkpoint does not match the current index plan")
    next_start = checkpoint.next_uncompleted_start
    if next_start < 0 or next_start > checkpoint.total_records:
        raise ValueError("Hotpot embedding checkpoint next_uncompleted_start is invalid")
    if next_start not in range(0, checkpoint.total_records + 1, checkpoint.batch_size) and next_start != checkpoint.total_records:
        raise ValueError("Hotpot embedding checkpoint next_uncompleted_start is not batch-aligned")
    batch = checkpoint.last_completed_batch
    if next_start == 0 and batch is not None:
        raise ValueError("Hotpot embedding checkpoint initial state has a completed batch")
    if next_start > 0 and batch is None:
        raise ValueError("Hotpot embedding checkpoint completed state is missing its last batch")
    if batch is not None:
        last_batch_size = checkpoint.total_records % checkpoint.batch_size or checkpoint.batch_size
        expected_start = next_start - (last_batch_size if next_start == checkpoint.total_records else checkpoint.batch_size)
        expected_records = plan.records[expected_start:next_start]
        if (
            batch.start != expected_start
            or batch.end_exclusive != next_start
            or batch.record_fingerprint != _records_fingerprint(expected_records)
        ):
            raise ValueError("Hotpot embedding checkpoint last batch does not match the index plan")


def _records_fingerprint(records: tuple[HotpotIndexRecord, ...]) -> str:
    """计算一个有序批次的业务 ID、point ID 与输入摘要指纹。"""

    payload = [
        {
            "chunk_id": record.paragraph.chunk_id,
            "point_id": record.point_id,
            "embedding_input_hash": record.embedding_input_hash,
        }
        for record in records
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
