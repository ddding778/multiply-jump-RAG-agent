"""HotpotQA paragraph 的 case/all Dense + BM25 + RRF + MMR 检索。"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import Any

from rag_index.bm25_index import Bm25IndexError
from rag_index.retrieval import FusionCandidate, RetrievalDataError, fuse_rrf, select_mmr

from .hotpot_bm25 import HotpotBm25Index


class HotpotRetrievalError(RuntimeError):
    """表示 HotpotQA 索引、请求或外部依赖不可安全使用。"""


@dataclass(frozen=True)
class HotpotRetrievalSettings:
    """保存一次运行固定的 HotpotQA Hybrid Retriever 配置。

    参数包括 collection、版本、embedding 合同、候选数和 RRF/MMR 参数；返回对象不保存密钥或用户问题。
    """

    collection_name: str
    index_version: str
    embedding_model: str
    embedding_dimensions: int
    candidate_top_k: int
    rrf_rank_constant: int
    rrf_dense_weight: float
    rrf_bm25_weight: float
    mmr_lambda: float


@dataclass(frozen=True)
class RetrievedParagraph:
    """表示从 HotpotQA Qdrant collection 读取的一个候选 paragraph。

    参数保存稳定 ID、标题、句子、分数和版本；返回对象供 RRF、MMR、证据定位和后续 Agent 使用。
    """

    point_id: str
    score: float
    case_id: str
    paragraph_index: int
    chunk_id: str
    title: str
    sentences: tuple[str, ...]
    text: str
    index_version: str

    @classmethod
    def from_point(cls, point: Any) -> "RetrievedParagraph":
        """从 Qdrant point 解析并严格校验 Hotpot paragraph payload。

        参数 point 为 Qdrant 返回对象；返回完整 RetrievedParagraph。缺字段、类型错误或句子不一致时抛出 HotpotRetrievalError。
        """

        payload = getattr(point, "payload", None)
        if not isinstance(payload, dict):
            raise HotpotRetrievalError("Hotpot Qdrant point payload is missing")
        point_id = getattr(point, "id", None)
        score = float(getattr(point, "score", 0.0))
        case_id = payload.get("case_id")
        paragraph_index = payload.get("paragraph_index")
        chunk_id = payload.get("chunk_id")
        title = payload.get("title")
        sentences = payload.get("sentences")
        text = payload.get("text")
        index_version = payload.get("index_version")
        if not isinstance(point_id, (str, int)) or not math.isfinite(score):
            raise HotpotRetrievalError("Hotpot Qdrant point ID or score is invalid")
        if not isinstance(case_id, str) or not case_id or not isinstance(paragraph_index, int) or paragraph_index < 0:
            raise HotpotRetrievalError("Hotpot Qdrant case metadata is invalid")
        if not isinstance(chunk_id, str) or not chunk_id or not isinstance(title, str) or not title:
            raise HotpotRetrievalError("Hotpot Qdrant paragraph metadata is invalid")
        if not isinstance(sentences, list) or not sentences or not all(isinstance(sentence, str) for sentence in sentences) or not any(sentences):
            raise HotpotRetrievalError("Hotpot Qdrant sentences are invalid")
        if not isinstance(text, str) or not text or not isinstance(index_version, str) or not index_version:
            raise HotpotRetrievalError("Hotpot Qdrant text or version is invalid")
        if text != " ".join(sentence for sentence in sentences if sentence):
            raise HotpotRetrievalError("Hotpot Qdrant text does not match sentences")
        return cls(
            point_id=str(point_id),
            score=score,
            case_id=case_id,
            paragraph_index=paragraph_index,
            chunk_id=chunk_id,
            title=title,
            sentences=tuple(sentences),
            text=text,
            index_version=index_version,
        )


@dataclass(frozen=True)
class HotpotRetrievalTrace:
    """保存不含问题正文的单次 Hybrid Retriever 元数据。"""

    scope: str
    case_id: str | None
    dense_candidate_count: int
    bm25_candidate_count: int
    fused_candidate_count: int
    bm25_case_cache_hit: bool
    duration_ms: int


@dataclass(frozen=True)
class HotpotSearchResult:
    """保存同一次 query embedding 下的 Dense 基线与 Hybrid paragraph 结果。"""

    dense: tuple[RetrievedParagraph, ...]
    hybrid: tuple[RetrievedParagraph, ...]
    trace: HotpotRetrievalTrace


class HotpotHybridRetriever:
    """协调同一 Hotpot collection 中按范围过滤的 Dense 与 BM25 检索。"""

    def __init__(self, embedding_client: Any, qdrant: Any, bm25_index: HotpotBm25Index, settings: HotpotRetrievalSettings, logger: logging.Logger) -> None:
        """初始化检索器并验证 Qdrant 与 BM25 artifact 属于相同 OPERA 版本。

        参数 embedding_client、qdrant、bm25_index、settings 和 logger 分别为外部依赖、artifact、固定配置和安全日志器；
        无返回值。索引不一致或不可用时抛出 HotpotRetrievalError。
        """

        self._embedding_client = embedding_client
        self._qdrant = qdrant
        self._bm25_index = bm25_index
        self._settings = settings
        self._logger = logger
        self._validate_active_index()

    def search(self, query: str, scope: str, case_id: str | None, top_k: int) -> HotpotSearchResult:
        """在明确 case 或 all 范围内执行完整 Hybrid 检索。

        参数 query 为已校验问题，scope 为检索范围，case_id 为 case 范围题目 ID，top_k 为最终命中数；
        返回 Dense 基线、Hybrid 结果和脱敏 trace。输入或依赖异常时抛出 HotpotRetrievalError。
        """

        self._validate_request(scope, case_id, top_k)
        started_at = time.perf_counter()
        vector = self._embed_query(query)
        dense = self._query_dense(vector, scope, case_id, self._settings.candidate_top_k)
        try:
            bm25_result = self._bm25_index.search(query=query, scope=scope, case_id=case_id, limit=self._settings.candidate_top_k)
        except Bm25IndexError as error:
            raise HotpotRetrievalError("Hotpot BM25 query failed") from error
        bm25_chunks = self._fetch_by_chunk_ids([candidate.chunk_id for candidate in bm25_result.candidates], scope, case_id)
        if {chunk.chunk_id for chunk in bm25_chunks} != {candidate.chunk_id for candidate in bm25_result.candidates}:
            raise HotpotRetrievalError("Hotpot BM25 candidates do not match Qdrant scope")
        fused = fuse_rrf(
            dense_chunks=dense,
            bm25_chunks=bm25_chunks,
            bm25_ranks={candidate.chunk_id: candidate.rank for candidate in bm25_result.candidates},
            rank_constant=self._settings.rrf_rank_constant,
            dense_weight=self._settings.rrf_dense_weight,
            bm25_weight=self._settings.rrf_bm25_weight,
        )
        selected = select_mmr(
            candidates=fused,
            candidate_vectors=self._fetch_candidate_vectors(fused, scope, case_id),
            top_k=top_k,
            mmr_lambda=self._settings.mmr_lambda,
        )
        trace = HotpotRetrievalTrace(
            scope=scope,
            case_id=case_id,
            dense_candidate_count=len(dense),
            bm25_candidate_count=len(bm25_result.candidates),
            fused_candidate_count=len(fused),
            bm25_case_cache_hit=bm25_result.cache_hit,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
        )
        self._logger.info(
            "event=opera_hotpot_search_completed collection=%s index_version=%s scope=%s case_id=%s top_k=%d dense_candidate_count=%d bm25_candidate_count=%d fused_candidate_count=%d bm25_case_cache_hit=%s duration_ms=%d",
            self._settings.collection_name,
            self._settings.index_version,
            scope,
            case_id or "none",
            top_k,
            trace.dense_candidate_count,
            trace.bm25_candidate_count,
            trace.fused_candidate_count,
            trace.bm25_case_cache_hit,
            trace.duration_ms,
        )
        return HotpotSearchResult(dense=tuple(dense[:top_k]), hybrid=tuple(selected), trace=trace)

    def _validate_active_index(self) -> None:
        """校验 collection 维度、版本 point 数和 BM25 业务 ID 完整一致。

        无参数；无返回值。外部依赖错误或 payload 不匹配时抛出 HotpotRetrievalError。
        """

        try:
            collection = self._qdrant.get_collection(self._settings.collection_name)
            dimensions = getattr(collection.config.params.vectors, "size", None)
            if dimensions != self._settings.embedding_dimensions:
                raise HotpotRetrievalError("Hotpot collection vector size does not match runtime")
            records, _ = self._qdrant.scroll(
                collection_name=self._settings.collection_name,
                scroll_filter=self._scope_filter(None),
                limit=100_000,
                with_payload=["chunk_id", "case_id", "index_version"],
                with_vectors=False,
            )
        except HotpotRetrievalError:
            raise
        except Exception as error:
            raise HotpotRetrievalError("Hotpot collection is unavailable") from error
        chunk_ids: set[str] = set()
        case_ids: set[str] = set()
        for record in records:
            payload = getattr(record, "payload", None)
            if not isinstance(payload, dict) or payload.get("index_version") != self._settings.index_version:
                raise HotpotRetrievalError("Hotpot collection contains invalid index version payload")
            chunk_id, case_id = payload.get("chunk_id"), payload.get("case_id")
            if not isinstance(chunk_id, str) or not chunk_id or not isinstance(case_id, str) or not case_id or chunk_id in chunk_ids:
                raise HotpotRetrievalError("Hotpot collection contains invalid or duplicate chunk ID")
            chunk_ids.add(chunk_id)
            case_ids.add(case_id)
        if not chunk_ids or chunk_ids != self._bm25_index.chunk_ids or case_ids != self._bm25_index.case_ids:
            raise HotpotRetrievalError("Hotpot Qdrant and BM25 artifacts do not match")

    def _validate_request(self, scope: str, case_id: str | None, top_k: int) -> None:
        """校验调用方的检索范围、题目 ID 和候选上限。

        参数 scope、case_id、top_k 分别为请求范围、可选题目和最终数；无返回值，非法时抛出 HotpotRetrievalError。
        """

        if scope not in {"case", "all"}:
            raise HotpotRetrievalError("Hotpot retrieval scope must be case or all")
        if scope == "case" and (not case_id or case_id not in self._bm25_index.case_ids):
            raise HotpotRetrievalError("Hotpot case scope requires indexed case_id")
        if scope == "all" and case_id is not None:
            raise HotpotRetrievalError("Hotpot all scope must not include case_id")
        if top_k <= 0 or top_k > self._settings.candidate_top_k:
            raise HotpotRetrievalError("Hotpot top_k is outside the configured range")

    def _embed_query(self, query: str) -> list[float]:
        """调用 embedding provider 获取与 OPERA collection 同维的查询向量。

        参数 query 为原始问题；返回向量。provider 错误、响应数量或维度异常时抛出 HotpotRetrievalError。
        """

        try:
            response = self._embedding_client.embeddings.create(
                model=self._settings.embedding_model,
                input=[query],
                dimensions=self._settings.embedding_dimensions,
            )
            vector = response.data[0].embedding if len(response.data) == 1 and response.data[0].index == 0 else None
        except Exception as error:
            raise HotpotRetrievalError("Hotpot query embedding failed") from error
        if not isinstance(vector, list) or len(vector) != self._settings.embedding_dimensions:
            raise HotpotRetrievalError("Hotpot query embedding dimension does not match collection")
        return vector

    def _query_dense(self, vector: list[float], scope: str, case_id: str | None, limit: int) -> list[RetrievedParagraph]:
        """使用 version 和可选 case 过滤器查询 Qdrant Dense 候选。

        参数 vector 为查询向量，scope 和 case_id 为范围，limit 为候选数；返回校验后的 paragraph 列表。
        """

        try:
            result = self._qdrant.query_points(
                collection_name=self._settings.collection_name,
                query=vector,
                query_filter=self._scope_filter(case_id if scope == "case" else None),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            chunks = [RetrievedParagraph.from_point(point) for point in result.points]
        except HotpotRetrievalError:
            raise
        except Exception as error:
            raise HotpotRetrievalError("Hotpot Dense query failed") from error
        if any(chunk.index_version != self._settings.index_version for chunk in chunks):
            raise HotpotRetrievalError("Hotpot Dense query returned another index version")
        if scope == "case" and any(chunk.case_id != case_id for chunk in chunks):
            raise HotpotRetrievalError("Hotpot Dense query escaped requested case")
        return chunks

    def _fetch_by_chunk_ids(self, chunk_ids: list[str], scope: str, case_id: str | None) -> list[RetrievedParagraph]:
        """按业务 chunk ID 回读 BM25 候选，并再次应用当前 scope 过滤。

        参数 chunk_ids 为候选 ID，scope 和 case_id 为当前范围；返回按照输入顺序重排的 paragraph。
        """

        if not chunk_ids:
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        filters = list(self._scope_filter(case_id if scope == "case" else None).must)
        filters.append(FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids)))
        try:
            records, _ = self._qdrant.scroll(
                collection_name=self._settings.collection_name,
                scroll_filter=Filter(must=filters),
                limit=len(chunk_ids),
                with_payload=True,
                with_vectors=False,
            )
            fetched = [RetrievedParagraph.from_point(record) for record in records]
        except Exception as error:
            raise HotpotRetrievalError("Hotpot BM25 candidate lookup failed") from error
        fetched_by_id = {chunk.chunk_id: chunk for chunk in fetched}
        return [fetched_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in fetched_by_id]

    def _fetch_candidate_vectors(self, candidates: list[FusionCandidate], scope: str, case_id: str | None) -> dict[str, list[float]]:
        """读取 RRF 候选向量，为复用的 MMR 排名提供相互相似度。

        参数 candidates 为融合候选，scope 和 case_id 为范围；返回 chunk ID 到向量映射，异常时抛出 HotpotRetrievalError。
        """

        if not candidates:
            return {}
        point_ids = [candidate.chunk.point_id for candidate in candidates]
        try:
            records = self._qdrant.retrieve(
                collection_name=self._settings.collection_name,
                ids=point_ids,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as error:
            raise HotpotRetrievalError("Hotpot candidate vector lookup failed") from error
        vectors: dict[str, list[float]] = {}
        point_id_set = set(point_ids)
        for record in records:
            if str(getattr(record, "id", "")) not in point_id_set:
                continue
            chunk = RetrievedParagraph.from_point(record)
            if chunk.index_version != self._settings.index_version or (scope == "case" and chunk.case_id != case_id):
                raise HotpotRetrievalError("Hotpot candidate vector escaped requested scope")
            vector = getattr(record, "vector", None)
            if not isinstance(vector, list) or len(vector) != self._settings.embedding_dimensions:
                raise HotpotRetrievalError("Hotpot candidate vector is invalid")
            vectors[chunk.chunk_id] = vector
        if set(vectors) != {candidate.chunk.chunk_id for candidate in candidates}:
            raise HotpotRetrievalError("Hotpot candidate vectors do not match RRF candidates")
        return vectors

    def _scope_filter(self, case_id: str | None) -> Any:
        """构造固定 index_version 加可选 case_id 的 Qdrant 硬过滤器。

        参数 case_id 为空时表示全库范围；返回 Qdrant Filter。
        """

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [FieldCondition(key="index_version", match=MatchValue(value=self._settings.index_version))]
        if case_id is not None:
            must.append(FieldCondition(key="case_id", match=MatchValue(value=case_id)))
        return Filter(must=must)
