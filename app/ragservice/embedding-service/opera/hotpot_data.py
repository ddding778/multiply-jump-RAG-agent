"""HotpotQA paragraph 导入、版本化索引计划与本地 artifact 写入。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_index.bm25_index import tokenize_for_bm25
from rag_index.retrieval_config import collection_name_for_version


HOTPOT_INDEX_SCHEMA_VERSION = 1
HOTPOT_CHUNKER_VERSION = "hotpot-paragraph-v1"


@dataclass(frozen=True)
class HotpotParagraph:
    """表示 HotpotQA 一个可独立检索的候选 paragraph。

    参数 case_id 为题目 ID，paragraph_index 为该题 context 中的顺序，title 为文档标题，sentences 为完整句子列表；
    返回对象只包含运行时允许检索的候选文本，不包含 answer 或 supporting_facts 标注。
    """

    case_id: str
    paragraph_index: int
    title: str
    sentences: tuple[str, ...]

    @property
    def chunk_id(self) -> str:
        """返回由题目和 paragraph 顺序组成的稳定业务 chunk ID。"""

        return f"{self.case_id}:{self.paragraph_index}"

    @property
    def text(self) -> str:
        """返回供检索使用的 paragraph 正文，并跳过不含内容的原始句子空位。"""

        return " ".join(sentence for sentence in self.sentences if sentence)

    @property
    def embedding_text(self) -> str:
        """返回送入 embedding 与 BM25 的标题加正文文本。"""

        return f"Title: {self.title}\n\n{self.text}"


@dataclass(frozen=True)
class HotpotCase:
    """表示一个可供 OPERA 检索的 HotpotQA 问题及其候选 paragraph。

    参数 case_id 为题目稳定 ID，question 为原题，paragraphs 为候选段落；返回对象不保存离线答案标注。
    """

    case_id: str
    question: str
    paragraphs: tuple[HotpotParagraph, ...]


@dataclass(frozen=True)
class HotpotIndexConfig:
    """保存 HotpotQA 独立索引的确定性配置。

    参数 collection_prefix、embedding_model、embedding_dimensions、embedding_batch_size 与 bm25_tokenizer_version
    共同参与 index_version 计算；返回对象不包含密钥或 Qdrant 连接。
    """

    collection_prefix: str
    embedding_model: str
    embedding_dimensions: int
    embedding_batch_size: int
    bm25_tokenizer_version: str


@dataclass(frozen=True)
class HotpotIndexRecord:
    """表示即将写入 Qdrant 的一个 Hotpot paragraph 记录。

    参数 paragraph 为候选段落，point_id 为确定性 Qdrant UUID，embedding_input_hash 用于版本追溯；
    返回对象不保存向量，向量仅在 embedding 请求返回后短暂使用。
    """

    paragraph: HotpotParagraph
    point_id: str
    embedding_input_hash: str


@dataclass(frozen=True)
class HotpotIndexPlan:
    """表示一批尚未 embedding 的 HotpotQA 索引计划。

    参数 config 为索引配置，dataset_sha256 绑定源数据，records 为有序 paragraph，index_version 和 collection_name
    为版本化存储标识；返回对象可用于 Qdrant 写入和 artifact 生成。
    """

    config: HotpotIndexConfig
    dataset_sha256: str
    records: tuple[HotpotIndexRecord, ...]
    index_version: str
    collection_name: str


def load_hotpot_cases(dataset_path: Path) -> tuple[HotpotCase, ...]:
    """读取并严格校验 HotpotQA distractor JSON 的运行时可用字段。

    参数 dataset_path 为本地 JSON 文件；返回按输入顺序排列的 HotpotCase 元组。
    answer、supporting_facts 等离线标注不会进入返回值；文件格式或候选段落非法时抛出 ValueError。
    """

    try:
        raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError("HotpotQA dataset cannot be read") from error
    except json.JSONDecodeError as error:
        raise ValueError("HotpotQA dataset is not valid JSON") from error
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("HotpotQA dataset must be a non-empty JSON array")

    cases: list[HotpotCase] = []
    seen_case_ids: set[str] = set()
    for row_index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"HotpotQA case {row_index} must be an object")
        case_id = raw_case.get("_id")
        question = raw_case.get("question")
        context = raw_case.get("context")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"HotpotQA case {row_index} has invalid _id")
        if case_id in seen_case_ids:
            raise ValueError("HotpotQA dataset contains duplicate _id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"HotpotQA case {case_id} has invalid question")
        if not isinstance(context, list) or not context:
            raise ValueError(f"HotpotQA case {case_id} has invalid context")
        paragraphs = tuple(_parse_paragraph(case_id, paragraph_index, raw_paragraph) for paragraph_index, raw_paragraph in enumerate(context))
        seen_case_ids.add(case_id)
        cases.append(HotpotCase(case_id=case_id, question=question.strip(), paragraphs=paragraphs))
    return tuple(cases)


def build_hotpot_index_plan(dataset_path: Path, config: HotpotIndexConfig) -> HotpotIndexPlan:
    """从 HotpotQA 数据集构建确定性 paragraph 索引计划。

    参数 dataset_path 为本地数据集，config 为索引合同；返回包含全部候选 paragraph 的 HotpotIndexPlan。
    配置非法、chunk ID 冲突或源数据不合规时抛出 ValueError。
    """

    _validate_index_config(config)
    cases = load_hotpot_cases(dataset_path)
    records = tuple(
        HotpotIndexRecord(
            paragraph=paragraph,
            point_id=_point_id(paragraph.chunk_id),
            embedding_input_hash=_sha256(paragraph.embedding_text),
        )
        for case in cases
        for paragraph in case.paragraphs
    )
    _validate_unique_records(records)
    dataset_sha256 = _sha256_bytes(dataset_path.read_bytes())
    index_version = _build_index_version(config, dataset_sha256, records)
    return HotpotIndexPlan(
        config=config,
        dataset_sha256=dataset_sha256,
        records=records,
        index_version=index_version,
        collection_name=collection_name_for_version(config.collection_prefix, index_version),
    )


def build_hotpot_payload(record: HotpotIndexRecord, index_version: str) -> dict[str, object]:
    """构造 HotpotQA Qdrant payload。

    参数 record 为索引记录，index_version 为所属版本；返回支持 case 过滤、证据句定位和结果渲染的 payload，
    不包含 answer、supporting_facts、向量或密钥。
    """

    paragraph = record.paragraph
    return {
        "record_type": "hotpot_paragraph",
        "case_id": paragraph.case_id,
        "paragraph_index": paragraph.paragraph_index,
        "chunk_id": paragraph.chunk_id,
        "title": paragraph.title,
        "sentences": list(paragraph.sentences),
        "text": paragraph.text,
        "embedding_input_hash": record.embedding_input_hash,
        "index_version": index_version,
    }


def write_hotpot_artifacts(plan: HotpotIndexPlan, output_root: Path) -> tuple[Path, Path]:
    """原子写入 Hotpot manifest 与可按 case 查询的 BM25 artifact。

    参数 plan 为已构建索引计划，output_root 为 `out/opera-index` 根目录；返回 manifest 和 BM25 artifact 的绝对路径。
    写入失败时保留临时文件供人工排查，不执行删除操作。
    """

    output_dir = output_root / plan.config.collection_prefix / plan.index_version
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest(plan, output_dir)
    bm25_path = _write_hotpot_bm25_index(plan, output_dir / "bm25_index.json")
    return manifest_path, bm25_path


def _parse_paragraph(case_id: str, paragraph_index: int, raw_paragraph: object) -> HotpotParagraph:
    """校验一个 context 元素并转换为 HotpotParagraph。

    参数 case_id 为所属题目，paragraph_index 为段落顺序，raw_paragraph 为 JSON 原始元素；返回清洗后的 paragraph。
    标题、句子列表或句子内容不合法时抛出 ValueError。
    """

    if not isinstance(raw_paragraph, list) or len(raw_paragraph) != 2:
        raise ValueError(f"HotpotQA case {case_id} context {paragraph_index} must contain title and sentences")
    title, raw_sentences = raw_paragraph
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"HotpotQA case {case_id} context {paragraph_index} has invalid title")
    if not isinstance(raw_sentences, list) or not raw_sentences:
        raise ValueError(f"HotpotQA case {case_id} context {paragraph_index} has invalid sentences")
    if not all(isinstance(sentence, str) for sentence in raw_sentences):
        raise ValueError(f"HotpotQA case {case_id} context {paragraph_index} contains invalid sentence")
    # 保留空字符串占位，确保后续 evidence 的 sentence_index 与 HotpotQA 原始标注一致；
    # embedding_text 和 text 属性会跳过空内容，避免它影响检索文本。
    sentences = tuple(sentence.strip() for sentence in raw_sentences)
    if not any(sentences):
        raise ValueError(f"HotpotQA case {case_id} context {paragraph_index} has no non-empty sentence")
    return HotpotParagraph(case_id=case_id, paragraph_index=paragraph_index, title=title.strip(), sentences=sentences)


def _validate_index_config(config: HotpotIndexConfig) -> None:
    """校验 HotpotQA 索引配置中的基础数值与文本字段。

    参数 config 为待校验配置；无返回值。字段为空、非正数或分词版本缺失时抛出 ValueError。
    """

    if not config.collection_prefix.strip():
        raise ValueError("Hotpot collection_prefix must not be empty")
    if not config.embedding_model.strip():
        raise ValueError("Hotpot embedding_model must not be empty")
    if config.embedding_dimensions <= 0 or config.embedding_batch_size <= 0:
        raise ValueError("Hotpot embedding dimensions and batch size must be positive")
    if not config.bm25_tokenizer_version.strip():
        raise ValueError("Hotpot BM25 tokenizer version must not be empty")


def _validate_unique_records(records: tuple[HotpotIndexRecord, ...]) -> None:
    """确认同一次计划内的业务 ID 与 Qdrant UUID 都唯一。

    参数 records 为全部索引记录；无返回值。任何重复都会抛出 ValueError，避免覆盖候选 paragraph。
    """

    chunk_ids = [record.paragraph.chunk_id for record in records]
    point_ids = [record.point_id for record in records]
    if len(chunk_ids) != len(set(chunk_ids)) or len(point_ids) != len(set(point_ids)):
        raise ValueError("HotpotQA index contains duplicate chunk or point IDs")


def _write_manifest(plan: HotpotIndexPlan, output_dir: Path) -> Path:
    """将 HotpotQA 索引计划原子写为 JSONL manifest。

    参数 plan 为索引计划，output_dir 为本次版本目录；返回 manifest 路径。
    """

    manifest_path = output_dir / "manifest.jsonl"
    metadata = {
        "record_type": "hotpot_manifest_metadata",
        "schema_version": HOTPOT_INDEX_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "index_version": plan.index_version,
        "collection_name": plan.collection_name,
        "collection_prefix": plan.config.collection_prefix,
        "dataset_sha256": plan.dataset_sha256,
        "chunker_version": HOTPOT_CHUNKER_VERSION,
        "embedding_model": plan.config.embedding_model,
        "embedding_dimensions": plan.config.embedding_dimensions,
        "bm25_tokenizer_version": plan.config.bm25_tokenizer_version,
        "chunk_count": len(plan.records),
    }
    temporary_file = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output_dir, prefix="hotpot-manifest-", suffix=".tmp", delete=False
    )
    try:
        with temporary_file as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
            for record in plan.records:
                stream.write(
                    json.dumps(
                        {"record_type": "hotpot_paragraph", "point_id": record.point_id, "payload": build_hotpot_payload(record, plan.index_version)},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(temporary_file.name, manifest_path)
    except Exception:
        raise
    return manifest_path.resolve()


def _write_hotpot_bm25_index(plan: HotpotIndexPlan, output_path: Path) -> Path:
    """写入附带 case 到 chunk 映射的 HotpotQA BM25 artifact。

    参数 plan 为索引计划，output_path 为目标 JSON 路径；返回成功写入的绝对路径。
    """

    case_chunk_ids = {
        case_id: [record.paragraph.chunk_id for record in plan.records if record.paragraph.case_id == case_id]
        for case_id in sorted({record.paragraph.case_id for record in plan.records})
    }
    temporary_file = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output_path.parent, prefix="hotpot-bm25-", suffix=".tmp", delete=False
    )
    try:
        with temporary_file as stream:
            # entries 按记录逐条分词和写出，避免同时持有 7 万余段的 token 列表与完整 JSON 对象。
            # case 映射仅保存 chunk ID，体积很小；后续 case/all 查询复用这一份 token artifact。
            stream.write(
                json.dumps(
                    {
                        "record_type": "hotpot_bm25_index",
                        "schema_version": HOTPOT_INDEX_SCHEMA_VERSION,
                        "index_version": plan.index_version,
                        "collection_name": plan.collection_name,
                        "tokenizer_version": plan.config.bm25_tokenizer_version,
                        "chunk_count": len(plan.records),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )[:-1]
            )
            stream.write(',"entries":[')
            for index, record in enumerate(plan.records):
                if index:
                    stream.write(",")
                json.dump(
                    {
                        "chunk_id": record.paragraph.chunk_id,
                        "tokens": tokenize_for_bm25(record.paragraph.embedding_text, plan.config.bm25_tokenizer_version),
                    },
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            stream.write('],"case_chunk_ids":')
            json.dump(case_chunk_ids, stream, ensure_ascii=False, sort_keys=True)
            stream.write("}")
            stream.write("\n")
        os.replace(temporary_file.name, output_path)
    except Exception:
        raise
    return output_path.resolve()


def _build_index_version(config: HotpotIndexConfig, dataset_sha256: str, records: tuple[HotpotIndexRecord, ...]) -> str:
    """根据 HotpotQA 数据集、索引合同与全部 embedding 输入生成版本号。

    参数 config 为索引配置，dataset_sha256 为源文件摘要，records 为有序记录；返回 SHA-256 十六进制版本。
    """

    contract = {
        "chunker_version": HOTPOT_CHUNKER_VERSION,
        "collection_prefix": config.collection_prefix,
        "dataset_sha256": dataset_sha256,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "bm25_tokenizer_version": config.bm25_tokenizer_version,
        "records": [
            {"chunk_id": record.paragraph.chunk_id, "embedding_input_hash": record.embedding_input_hash} for record in records
        ],
    }
    return _sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _point_id(chunk_id: str) -> str:
    """将 HotpotQA 业务 chunk ID 映射为确定性 Qdrant UUID。"""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ai-chat:hotpot:{chunk_id}"))


def _sha256(value: str) -> str:
    """计算 UTF-8 文本的 SHA-256 小写十六进制摘要。"""

    return _sha256_bytes(value.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    """计算字节串的 SHA-256 小写十六进制摘要。"""

    return hashlib.sha256(value).hexdigest()
