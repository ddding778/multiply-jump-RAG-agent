"""支持 HotpotQA case/all 范围的 BM25 artifact 加载与查询。"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from rag_index.bm25_index import Bm25Candidate, Bm25IndexError, tokenize_for_bm25


HOTPOT_BM25_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScopedBm25Result:
    """表示一次 HotpotQA BM25 查询的候选与缓存状态。

    参数 candidates 为按分数稳定排序的候选，cache_hit 表示 case 模型是否命中 LRU；返回对象不含正文。
    """

    candidates: tuple[Bm25Candidate, ...]
    cache_hit: bool


class HotpotBm25Index:
    """保存一次分词结果，并分别支持全库与单题 BM25 视图。"""

    def __init__(
        self,
        index_version: str,
        collection_name: str,
        tokenizer_version: str,
        entries: dict[str, tuple[str, ...]],
        case_chunk_ids: dict[str, tuple[str, ...]],
        case_cache_size: int,
    ) -> None:
        """使用已校验 token 与 case 映射初始化全局 BM25 和空的 case LRU 缓存。

        参数 index_version、collection_name、tokenizer_version 用于追溯，entries 为 chunk token，case_chunk_ids 为题目候选映射，
        case_cache_size 为缓存上限；依赖缺失或条目非法时抛出 Bm25IndexError。
        """

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:
            raise Bm25IndexError("rank-bm25 is required to use the BM25 index") from error
        if not entries or not case_chunk_ids or case_cache_size <= 0:
            raise Bm25IndexError("Hotpot BM25 index has invalid entries or cache size")
        self.index_version = index_version
        self.collection_name = collection_name
        self.tokenizer_version = tokenizer_version
        self._entries = entries
        self._case_chunk_ids = case_chunk_ids
        self._case_cache_size = case_cache_size
        self._global_chunk_ids = tuple(sorted(entries))
        self._global_model = BM25Okapi([list(entries[chunk_id]) for chunk_id in self._global_chunk_ids])
        self._case_models: OrderedDict[str, tuple[tuple[str, ...], object]] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def chunk_ids(self) -> set[str]:
        """返回 artifact 内所有业务 chunk ID 的副本。"""

        return set(self._entries)

    @property
    def case_ids(self) -> set[str]:
        """返回 artifact 内所有 case ID 的副本。"""

        return set(self._case_chunk_ids)

    def search(self, query: str, scope: str, case_id: str | None, limit: int) -> ScopedBm25Result:
        """按 all 或 case 范围执行 BM25 查询。

        参数 query 为原始问题，scope 为 all 或 case，case_id 仅在 case 时必填，limit 为候选数；
        返回候选和缓存命中状态。非法范围、未知题目或非正 limit 时抛出 Bm25IndexError。
        """

        if limit <= 0:
            raise Bm25IndexError("Hotpot BM25 search limit must be positive")
        query_tokens = tokenize_for_bm25(query, self.tokenizer_version)
        if not query_tokens:
            return ScopedBm25Result(candidates=(), cache_hit=False)
        if scope == "all":
            return ScopedBm25Result(candidates=tuple(self._rank(self._global_chunk_ids, self._global_model, query_tokens, limit)), cache_hit=False)
        if scope != "case" or not case_id:
            raise Bm25IndexError("Hotpot case scope requires case_id")
        chunk_ids, model, cache_hit = self._case_model(case_id)
        return ScopedBm25Result(candidates=tuple(self._rank(chunk_ids, model, query_tokens, limit)), cache_hit=cache_hit)

    def _case_model(self, case_id: str) -> tuple[tuple[str, ...], object, bool]:
        """读取或按需构建一个题目对应的 10 段 BM25 模型。

        参数 case_id 为题目 ID；返回 chunk ID 顺序、BM25 模型和缓存命中标记。题目不存在时抛出 Bm25IndexError。
        """

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:
            raise Bm25IndexError("rank-bm25 is required to use the BM25 index") from error
        with self._lock:
            cached = self._case_models.pop(case_id, None)
            if cached is not None:
                self._case_models[case_id] = cached
                return cached[0], cached[1], True
            chunk_ids = self._case_chunk_ids.get(case_id)
            if chunk_ids is None:
                raise Bm25IndexError("HotpotQA case_id is not indexed")
            model = BM25Okapi([list(self._entries[chunk_id]) for chunk_id in chunk_ids])
            self._case_models[case_id] = (chunk_ids, model)
            if len(self._case_models) > self._case_cache_size:
                self._case_models.popitem(last=False)
            return chunk_ids, model, False

    def _rank(self, chunk_ids: tuple[str, ...], model: object, query_tokens: list[str], limit: int) -> list[Bm25Candidate]:
        """将 BM25 原始分数转换为稳定的业务候选排名。

        参数 chunk_ids 为模型语料顺序，model 为 BM25 实例，query_tokens 为查询词，limit 为上限；返回正分候选。
        """

        scores = model.get_scores(query_tokens)
        ranked = sorted(
            ((chunk_id, float(score)) for chunk_id, score in zip(chunk_ids, scores, strict=True) if score > 0),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
        return [Bm25Candidate(chunk_id=chunk_id, score=score, rank=rank) for rank, (chunk_id, score) in enumerate(ranked, start=1)]


def load_hotpot_bm25_index(
    path: Path,
    expected_index_version: str,
    expected_collection_name: str,
    expected_tokenizer_version: str,
    case_cache_size: int,
) -> HotpotBm25Index:
    """读取并校验 HotpotQA scope-aware BM25 artifact。

    参数 path 为 artifact 路径，expected_* 为当前运行合同，case_cache_size 为 LRU 上限；返回 HotpotBm25Index。
    文件、版本、映射或 token 异常时抛出 Bm25IndexError。
    """

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise Bm25IndexError("Hotpot BM25 index cannot be read") from error
    except json.JSONDecodeError as error:
        raise Bm25IndexError("Hotpot BM25 index is not valid JSON") from error
    if not isinstance(artifact, dict) or artifact.get("record_type") != "hotpot_bm25_index":
        raise Bm25IndexError("Hotpot BM25 index schema is unsupported")
    if artifact.get("schema_version") != HOTPOT_BM25_SCHEMA_VERSION:
        raise Bm25IndexError("Hotpot BM25 index schema version is unsupported")
    if artifact.get("index_version") != expected_index_version or artifact.get("collection_name") != expected_collection_name:
        raise Bm25IndexError("Hotpot BM25 index version or collection does not match runtime")
    if artifact.get("tokenizer_version") != expected_tokenizer_version:
        raise Bm25IndexError("Hotpot BM25 tokenizer version does not match runtime")
    entries = _parse_entries(artifact.get("entries"))
    case_chunk_ids = _parse_case_chunk_ids(artifact.get("case_chunk_ids"), set(entries))
    return HotpotBm25Index(
        index_version=expected_index_version,
        collection_name=expected_collection_name,
        tokenizer_version=expected_tokenizer_version,
        entries=entries,
        case_chunk_ids=case_chunk_ids,
        case_cache_size=case_cache_size,
    )


def _parse_entries(raw_entries: object) -> dict[str, tuple[str, ...]]:
    """校验 artifact token entries 并按 chunk ID 建立映射。

    参数 raw_entries 为反序列化 entries 字段；返回 chunk 到不可变 token 元组的映射。
    """

    if not isinstance(raw_entries, list) or not raw_entries:
        raise Bm25IndexError("Hotpot BM25 entries are invalid")
    entries: dict[str, tuple[str, ...]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise Bm25IndexError("Hotpot BM25 entry is invalid")
        chunk_id = raw_entry.get("chunk_id")
        tokens = raw_entry.get("tokens")
        if not isinstance(chunk_id, str) or not chunk_id or not isinstance(tokens, list) or not all(isinstance(token, str) and token for token in tokens):
            raise Bm25IndexError("Hotpot BM25 entry fields are invalid")
        if chunk_id in entries:
            raise Bm25IndexError("Hotpot BM25 entries contain duplicate chunk_id")
        entries[chunk_id] = tuple(tokens)
    return entries


def _parse_case_chunk_ids(raw_mapping: object, entry_ids: set[str]) -> dict[str, tuple[str, ...]]:
    """校验每个 case 的候选 chunk 映射与全局 token 集合一致。

    参数 raw_mapping 为 artifact 映射，entry_ids 为全部 token 条目 ID；返回不可变 case 映射。
    """

    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise Bm25IndexError("Hotpot BM25 case mapping is invalid")
    mapping: dict[str, tuple[str, ...]] = {}
    flattened: list[str] = []
    for case_id, raw_chunk_ids in raw_mapping.items():
        if not isinstance(case_id, str) or not case_id or not isinstance(raw_chunk_ids, list) or not raw_chunk_ids:
            raise Bm25IndexError("Hotpot BM25 case mapping entry is invalid")
        chunk_ids = tuple(raw_chunk_ids)
        if not all(isinstance(chunk_id, str) and chunk_id in entry_ids for chunk_id in chunk_ids):
            raise Bm25IndexError("Hotpot BM25 case mapping references unknown chunk")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise Bm25IndexError("Hotpot BM25 case mapping has duplicate chunk")
        mapping[case_id] = chunk_ids
        flattened.extend(chunk_ids)
    if set(flattened) != entry_ids or len(flattened) != len(entry_ids):
        raise Bm25IndexError("Hotpot BM25 case mapping does not cover entries exactly once")
    return mapping
