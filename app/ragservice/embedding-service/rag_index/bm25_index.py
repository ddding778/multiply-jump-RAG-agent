"""RAG V2 的中文和标识符 BM25 离线索引与查询能力。"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BM25_INDEX_SCHEMA_VERSION = 1
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class Bm25IndexError(ValueError):
    """表示 BM25 artifact 缺失、内容损坏或与当前索引版本不匹配。"""


@dataclass(frozen=True)
class Bm25Document:
    """表示准备写入 BM25 artifact 的一个 chunk 文本。

    参数 chunk_id 为业务 chunk 标识，text 为标题路径和正文拼接后的检索文本；
    返回对象不包含向量、密钥或其他运行时状态。
    """

    chunk_id: str
    text: str


@dataclass(frozen=True)
class Bm25Candidate:
    """表示 BM25 返回的一个按分数排序的 chunk 候选。

    参数 chunk_id 为业务 chunk 标识，score 为 BM25 原始分数，rank 为从 1 开始的排名；
    返回对象供 RRF 使用，不包含正文。
    """

    chunk_id: str
    score: float
    rank: int


class Bm25Index:
    """保存加载后的 BM25 模型和与索引版本绑定的 token 化 chunk 集合。"""

    def __init__(self, index_version: str, collection_name: str, tokenizer_version: str, entries: list[tuple[str, list[str]]]) -> None:
        """从已经校验的 token 条目构建内存 BM25 模型。

        参数 index_version、collection_name、tokenizer_version 用于运行时追溯，entries 为 chunk ID 和 token 序列；
        无返回值。依赖缺失或条目为空时抛出 Bm25IndexError。
        """

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:
            raise Bm25IndexError("rank-bm25 is required to use the BM25 index") from error
        if not entries:
            raise Bm25IndexError("BM25 index has no entries")
        self.index_version = index_version
        self.collection_name = collection_name
        self.tokenizer_version = tokenizer_version
        self._chunk_ids = [chunk_id for chunk_id, _ in entries]
        self._tokenized_corpus = [tokens for _, tokens in entries]
        self._model = BM25Okapi(self._tokenized_corpus)

    @property
    def chunk_ids(self) -> set[str]:
        """返回 BM25 artifact 中所有业务 chunk ID 的副本。

        无参数；返回用于与 Qdrant 当前 index_version 校验的集合。
        """

        return set(self._chunk_ids)

    def search(self, query: str, limit: int) -> list[Bm25Candidate]:
        """按 BM25 分数检索用户问题对应的 chunk 候选。

        参数 query 为原始检索问题，limit 为最大候选数；返回分数大于零且按分数、chunk ID 稳定排序的候选。
        limit 非正数时抛出 Bm25IndexError。
        """

        if limit <= 0:
            raise Bm25IndexError("BM25 search limit must be positive")
        query_tokens = tokenize_for_bm25(query, self.tokenizer_version)
        if not query_tokens:
            return []
        scores = self._model.get_scores(query_tokens)
        ranked = sorted(
            ((chunk_id, float(score)) for chunk_id, score in zip(self._chunk_ids, scores, strict=True) if score > 0),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
        return [Bm25Candidate(chunk_id=chunk_id, score=score, rank=rank) for rank, (chunk_id, score) in enumerate(ranked, start=1)]


def tokenize_for_bm25(text: str, tokenizer_version: str) -> list[str]:
    """按配置的中文和标识符规则将文本切分为稳定的 BM25 token 序列。

    参数 text 为标题路径、正文或用户问题，tokenizer_version 为 artifact 声明的分词合同；
    返回保留中文词、完整标识符及其 camelCase/snake_case/path 组成部分的 token 列表。
    """

    if tokenizer_version != "jieba-identifier-v1":
        raise Bm25IndexError("unsupported BM25 tokenizer version")
    if not isinstance(text, str):
        raise Bm25IndexError("BM25 text must be a string")
    try:
        import jieba
    except ImportError as error:
        raise Bm25IndexError("jieba is required to build the BM25 index") from error
    jieba.setLogLevel(logging.WARNING)

    normalized = unicodedata.normalize("NFKC", text).strip()
    if not normalized:
        return []
    tokens = [token.strip().lower() for token in jieba.lcut(normalized) if _is_meaningful_token(token)]
    for identifier_match in _IDENTIFIER_PATTERN.finditer(normalized):
        raw_identifier = identifier_match.group(0)
        tokens.append(raw_identifier.lower())
        for raw_segment in re.split(r"[._/-]+", raw_identifier):
            if not raw_segment:
                continue
            tokens.append(raw_segment.lower())
            tokens.extend(part.lower() for part in _CAMEL_CASE_BOUNDARY.split(raw_segment) if part)
    return tokens


def write_bm25_index(
    documents: Iterable[Bm25Document],
    output_path: Path,
    index_version: str,
    collection_name: str,
    tokenizer_version: str,
) -> Path:
    """将 token 化后的 BM25 索引原子写入指定版本目录。

    参数 documents 为待索引 chunk 文本，output_path 为 bm25_index.json 路径，index_version 和 collection_name
    为批次合同，tokenizer_version 为分词合同；返回成功写入的绝对路径。输入重复或无效时抛出 Bm25IndexError。
    """

    entries = _tokenize_documents(documents, tokenizer_version)
    artifact = {
        "record_type": "bm25_index",
        "schema_version": BM25_INDEX_SCHEMA_VERSION,
        "index_version": index_version,
        "collection_name": collection_name,
        "tokenizer_version": tokenizer_version,
        "chunk_count": len(entries),
        "entries": [{"chunk_id": chunk_id, "tokens": tokens} for chunk_id, tokens in entries],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output_path.parent,
        prefix="bm25-",
        suffix=".tmp",
        delete=False,
    )
    with temporary_file as stream:
        json.dump(artifact, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_file.name, output_path)
    return output_path.resolve()


def load_bm25_index(
    path: Path,
    expected_index_version: str,
    expected_collection_name: str,
    expected_tokenizer_version: str,
) -> Bm25Index:
    """读取并校验与当前 Qdrant 批次绑定的 BM25 artifact。

    参数 path 为 artifact 路径，expected_* 为运行时明确指定的版本、collection 和分词合同；
    返回可查询的 Bm25Index。文件格式、版本、collection、tokenizer 或条目不一致时抛出 Bm25IndexError。
    """

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise Bm25IndexError("BM25 index file cannot be read") from error
    except json.JSONDecodeError as error:
        raise Bm25IndexError("BM25 index file is not valid JSON") from error
    if not isinstance(artifact, dict):
        raise Bm25IndexError("BM25 index root must be an object")
    if artifact.get("record_type") != "bm25_index" or artifact.get("schema_version") != BM25_INDEX_SCHEMA_VERSION:
        raise Bm25IndexError("BM25 index schema is unsupported")
    if artifact.get("index_version") != expected_index_version:
        raise Bm25IndexError("BM25 index_version does not match RAG_INDEX_VERSION")
    if artifact.get("collection_name") != expected_collection_name:
        raise Bm25IndexError("BM25 collection_name does not match current collection")
    if artifact.get("tokenizer_version") != expected_tokenizer_version:
        raise Bm25IndexError("BM25 tokenizer version does not match runtime config")
    entries = _validate_artifact_entries(artifact.get("entries"), artifact.get("chunk_count"))
    return Bm25Index(
        index_version=expected_index_version,
        collection_name=expected_collection_name,
        tokenizer_version=expected_tokenizer_version,
        entries=entries,
    )


def _tokenize_documents(documents: Iterable[Bm25Document], tokenizer_version: str) -> list[tuple[str, list[str]]]:
    """校验业务 chunk ID 并生成可写入 artifact 的 token 条目。

    参数 documents 为离线构建的 chunk 文本，tokenizer_version 为分词合同；返回稳定排序的 chunk ID/token 列表。
    重复 ID、空 ID 或非法文本时抛出 Bm25IndexError。
    """

    entries: list[tuple[str, list[str]]] = []
    seen_ids: set[str] = set()
    for document in documents:
        if not isinstance(document.chunk_id, str) or not document.chunk_id:
            raise Bm25IndexError("BM25 document chunk_id is invalid")
        if document.chunk_id in seen_ids:
            raise Bm25IndexError("BM25 document chunk_id is duplicated")
        seen_ids.add(document.chunk_id)
        entries.append((document.chunk_id, tokenize_for_bm25(document.text, tokenizer_version)))
    if not entries:
        raise Bm25IndexError("BM25 index has no documents")
    return sorted(entries, key=lambda item: item[0])


def _validate_artifact_entries(raw_entries: object, expected_count: object) -> list[tuple[str, list[str]]]:
    """校验 JSON artifact 中的 BM25 token 条目。

    参数 raw_entries 为 JSON 反序列化后的 entries，expected_count 为声明的 chunk 数；
    返回已校验的条目。字段类型、重复 ID 或数量不匹配时抛出 Bm25IndexError。
    """

    if not isinstance(raw_entries, list) or not isinstance(expected_count, int) or expected_count <= 0:
        raise Bm25IndexError("BM25 index entries are invalid")
    entries: list[tuple[str, list[str]]] = []
    seen_ids: set[str] = set()
    for item in raw_entries:
        if not isinstance(item, dict):
            raise Bm25IndexError("BM25 index entry is invalid")
        chunk_id = item.get("chunk_id")
        tokens = item.get("tokens")
        if not isinstance(chunk_id, str) or not chunk_id or not isinstance(tokens, list) or not all(
            isinstance(token, str) and token for token in tokens
        ):
            raise Bm25IndexError("BM25 index entry fields are invalid")
        if chunk_id in seen_ids:
            raise Bm25IndexError("BM25 index contains duplicate chunk_id")
        seen_ids.add(chunk_id)
        entries.append((chunk_id, tokens))
    if len(entries) != expected_count:
        raise Bm25IndexError("BM25 index chunk_count does not match entries")
    return entries


def _is_meaningful_token(token: str) -> bool:
    """判断 jieba 切分结果是否包含可用于 BM25 的文字或数字。

    参数 token 为一个 jieba 分词结果；返回至少含一个 Unicode 字母或数字时为真。
    """

    return isinstance(token, str) and any(character.isalnum() for character in token)
