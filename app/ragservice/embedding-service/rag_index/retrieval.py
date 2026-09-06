"""RAG V2 在线 Dense 检索与 P1 上下文扩展。"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from .bm25_index import Bm25Index


class RetrievalConfigurationError(RuntimeError):
    """表示在线 RAG 缺少或使用了错误配置。"""


class RetrievalRequestError(RuntimeError):
    """表示在线 RAG 收到了不符合合同的检索请求。"""


class RetrievalUnavailableError(RuntimeError):
    """表示 embedding 或 Qdrant 当前不可用于检索。"""


class RetrievalDataError(RuntimeError):
    """表示 Qdrant payload 不满足已确认的 RAG V2 合同。"""


@dataclass(frozen=True)
class RetrievalSettings:
    """保存在线混合检索和 P1 扩展所需配置。

    参数包含 collection、当前 index_version、embedding、BM25/RRF/MMR 与 P1 合同；
    返回实例只描述一次服务运行的只读检索合同。
    """

    collection_name: str
    index_version: str
    embedding_model: str
    embedding_dimensions: int
    dense_top_k_max: int
    candidate_top_k: int
    rrf_rank_constant: int
    rrf_dense_weight: float
    rrf_bm25_weight: float
    mmr_lambda: float
    p1_no_expand_threshold: int
    p1_context_token_budget: int


@dataclass(frozen=True)
class FusionCandidate:
    """保存某个 chunk 在 Dense、BM25 与 RRF 中的合并排名信息。

    参数 chunk 为完整 payload，rrf_score 为两路排名融合分数，dense_rank 与 bm25_rank 为可选来源排名；
    返回对象供 MMR 选择，不向 HTTP 响应暴露内部评分。
    """

    chunk: "RetrievedChunk"
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None


@dataclass(frozen=True)
class RetrievalTrace:
    """记录一次混合检索的安全元数据，供日志和评测摘要使用。

    参数只包含候选数量和 MMR 淘汰数量；返回对象不包含用户问题、正文、向量或密钥。
    """

    dense_candidate_count: int
    bm25_candidate_count: int
    fused_candidate_count: int
    mmr_discarded_count: int


@dataclass(frozen=True)
class RetrievedChunk:
    """保存从 Qdrant payload 解析出的一个 RAG V2 chunk。

    参数对应离线索引写入的 payload 字段；score 仅在 Dense 核心命中时有值，
    返回对象用于 P1 关系校验、token 预算和最终上下文渲染。
    """

    point_id: str
    score: float | None
    document_id: str
    section_id: str
    parent_section_id: str | None
    chunk_id: str
    chunk_order: int
    heading_path: list[dict[str, object]]
    text: str
    line_start: int
    line_end: int
    token_count: int
    previous_chunk_id: str | None
    next_chunk_id: str | None
    index_version: str

    @classmethod
    def from_point(cls, point: Any) -> "RetrievedChunk":
        """从 Qdrant ScoredPoint 或 Record 解析并校验 RAG V2 payload。

        参数 point 为带 id、payload 和可选 score 的 Qdrant 返回对象；
        返回解析后的 RetrievedChunk，缺少字段或字段类型错误时抛出 RetrievalDataError。
        """

        payload = getattr(point, "payload", None)
        if not isinstance(payload, dict):
            raise RetrievalDataError("Qdrant point payload is missing")
        required_strings = (
            "document_id",
            "section_id",
            "chunk_id",
            "text",
            "index_version",
        )
        for field_name in required_strings:
            if not isinstance(payload.get(field_name), str) or not payload[field_name]:
                raise RetrievalDataError(f"Qdrant payload field is invalid: {field_name}")
        if not isinstance(payload.get("chunk_order"), int):
            raise RetrievalDataError("Qdrant payload field is invalid: chunk_order")
        if not isinstance(payload.get("line_start"), int) or not isinstance(payload.get("line_end"), int):
            raise RetrievalDataError("Qdrant payload line range is invalid")
        if not isinstance(payload.get("token_count"), int) or payload["token_count"] <= 0:
            raise RetrievalDataError("Qdrant payload token_count is invalid")
        if not isinstance(payload.get("heading_path"), list):
            raise RetrievalDataError("Qdrant payload heading_path is invalid")

        parent_section_id = payload.get("parent_section_id")
        previous_chunk_id = payload.get("previous_chunk_id")
        next_chunk_id = payload.get("next_chunk_id")
        for field_name, value in (
            ("parent_section_id", parent_section_id),
            ("previous_chunk_id", previous_chunk_id),
            ("next_chunk_id", next_chunk_id),
        ):
            if value is not None and not isinstance(value, str):
                raise RetrievalDataError(f"Qdrant payload field is invalid: {field_name}")

        score = getattr(point, "score", None)
        if score is not None and not isinstance(score, (int, float)):
            raise RetrievalDataError("Qdrant point score is invalid")
        return cls(
            point_id=str(getattr(point, "id")),
            score=float(score) if score is not None else None,
            document_id=payload["document_id"],
            section_id=payload["section_id"],
            parent_section_id=parent_section_id,
            chunk_id=payload["chunk_id"],
            chunk_order=payload["chunk_order"],
            heading_path=payload["heading_path"],
            text=payload["text"],
            line_start=payload["line_start"],
            line_end=payload["line_end"],
            token_count=payload["token_count"],
            previous_chunk_id=previous_chunk_id,
            next_chunk_id=next_chunk_id,
            index_version=payload["index_version"],
        )

    def render_for_context(self) -> str:
        """将标题路径和正文重建为可安全注入 Chat 的参考资料文本。

        无参数；返回包含完整标题路径、行号区间和正文的字符串，不将其中内容视为可信指令。
        """

        heading_texts = [
            str(item.get("text", "")).strip()
            for item in self.heading_path
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]
        heading_path = " > ".join(heading_texts) if heading_texts else "未命名标题区间"
        return f"标题路径：{heading_path}\n来源行：{self.line_start}-{self.line_end}\n\n正文：\n{self.text}"


@dataclass(frozen=True)
class ContextGroup:
    """保存一个核心 Dense 命中及其 P1 扩展后的上下文组。

    参数 core_chunk_id 为核心命中业务 ID，chunks 为按文档顺序排列的唯一 chunk；
    返回对象可渲染为兼容旧 Go 客户端的单个 documents 字符串。
    """

    core_chunk_id: str
    chunks: list[RetrievedChunk]

    def render(self) -> str:
        """渲染一个核心命中的完整 P1 上下文组。

        无参数；返回由分隔线连接的标题路径和正文，不返回向量或内部 Qdrant point ID。
        """

        return "\n\n---\n\n".join(chunk.render_for_context() for chunk in self.chunks)


class P1ContextAssembler:
    """按已确认的同标题/同父标题规则扩展 Dense 核心命中。

    参数 no_expand_threshold 为核心 chunk 不扩展阈值，context_token_budget 为每个核心上下文组上限；
    返回对象不负责 Qdrant 查询或 embedding，只负责编排候选和预算。
    """

    def __init__(self, no_expand_threshold: int, context_token_budget: int) -> None:
        """初始化 P1 编排器并校验阈值。

        参数 no_expand_threshold 为达到该值即不扩展的 token 数，context_token_budget 为核心加扩展总上限；
        阈值非法时抛出 ValueError。
        """

        if no_expand_threshold <= 0:
            raise ValueError("p1_no_expand_threshold must be positive")
        if context_token_budget < no_expand_threshold:
            raise ValueError("p1_context_token_budget must be at least no_expand_threshold")
        self._no_expand_threshold = no_expand_threshold
        self._context_token_budget = context_token_budget

    def assemble(
        self,
        core_chunks: list[RetrievedChunk],
        fetch_adjacent_chunks: Callable[[RetrievedChunk], list[RetrievedChunk]],
    ) -> list[ContextGroup]:
        """为每个 Dense 核心命中构建受预算限制的 P1 上下文组。

        参数 core_chunks 按 Dense 分数排序，fetch_adjacent_chunks 只读取文档顺序中确实存在的前后候选；
        返回按核心命中顺序排列的 ContextGroup。
        """

        core_ids = {chunk.chunk_id for chunk in core_chunks}
        used_extensions: set[str] = set()
        groups: list[ContextGroup] = []
        for core in core_chunks:
            selected = [core]
            if core.token_count < self._no_expand_threshold:
                selected.extend(
                    self._select_extensions(
                        core=core,
                        core_ids=core_ids,
                        used_extensions=used_extensions,
                        fetch_adjacent_chunks=fetch_adjacent_chunks,
                    )
                )
            groups.append(
                ContextGroup(
                    core_chunk_id=core.chunk_id,
                    chunks=sorted(selected, key=lambda item: (item.chunk_order, item.chunk_id)),
                )
            )
        return groups

    def _select_extensions(
        self,
        core: RetrievedChunk,
        core_ids: set[str],
        used_extensions: set[str],
        fetch_adjacent_chunks: Callable[[RetrievedChunk], list[RetrievedChunk]],
    ) -> list[RetrievedChunk]:
        """按文档顺序的直接邻居和 token 预算选择 P1 扩展 chunk。

        参数 core 为当前核心命中，core_ids 为全部核心命中 ID，used_extensions 为已加入其他组的扩展，
        fetch_adjacent_chunks 只返回已存在的前后块；返回不重复且符合 P1 关系的扩展列表。
        """

        ordered_candidates = self._ordered_adjacent_candidates(core, fetch_adjacent_chunks(core))
        selected: list[RetrievedChunk] = []
        seen_ids = {core.chunk_id}
        used_tokens = core.token_count
        for candidate in ordered_candidates:
            if candidate.chunk_id in seen_ids or candidate.chunk_id in core_ids or candidate.chunk_id in used_extensions:
                continue
            if not _is_p1_related(core, candidate):
                continue
            if used_tokens + candidate.token_count > self._context_token_budget:
                continue
            selected.append(candidate)
            seen_ids.add(candidate.chunk_id)
            used_extensions.add(candidate.chunk_id)
            used_tokens += candidate.token_count
        return selected

    @staticmethod
    def _ordered_adjacent_candidates(core: RetrievedChunk, candidates: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
        """将文档前后候选按前一块、后一块的顺序排列。

        参数 core 为核心命中，candidates 为文档顺序上实际存在的相邻 chunk；返回稳定的前后顺序列表。
        """

        return sorted(
            candidates,
            key=lambda candidate: (
                0 if candidate.chunk_order == core.chunk_order - 1 else 1,
                candidate.chunk_order,
                candidate.chunk_id,
            ),
        )


class DenseRetrievalService:
    """协调 Dense、BM25、RRF、MMR 和 P1 上下文组装。

    参数 embedding_client 为 OpenAI-compatible 客户端，qdrant 为 Qdrant 客户端，bm25_index 为已验证本地索引，
    settings 为一次运行固定配置，logger 为只记录元数据的日志器。
    """

    def __init__(
        self,
        embedding_client: Any,
        qdrant: Any,
        bm25_index: Bm25Index,
        settings: RetrievalSettings,
        logger: logging.Logger,
    ) -> None:
        """初始化服务并验证已明确指定的索引版本可用。

        参数同类说明；无返回值。collection、BM25 artifact 或 index_version 不匹配时抛出 RetrievalUnavailableError。
        """

        self._embedding_client = embedding_client
        self._qdrant = qdrant
        self._bm25_index = bm25_index
        self._settings = settings
        self._logger = logger
        self._document_chunk_orders: dict[str, dict[int, str]] = {}
        self._assembler = P1ContextAssembler(
            no_expand_threshold=settings.p1_no_expand_threshold,
            context_token_budget=settings.p1_context_token_budget,
        )
        self._validate_active_index()

    def search(self, query: str, top_k: int) -> list[str]:
        """执行一次完整混合检索并返回兼容旧 HTTP 合同的参考资料字符串列表。

        参数 query 为已校验的用户问题，top_k 为 RRF/MMR 后最终核心命中数；
        返回每个核心命中对应的 P1 上下文组文本，外部不可用时抛出 RetrievalUnavailableError。
        """

        if top_k <= 0 or top_k > self._settings.dense_top_k_max:
            raise RetrievalRequestError("top_k is outside the configured range")
        started_at = time.perf_counter()
        try:
            vector = self._embed_query(query)
            core_chunks, trace = self._query_hybrid_core_chunks(query, vector, top_k)
            contexts = self._assembler.assemble(
                core_chunks=core_chunks,
                fetch_adjacent_chunks=self._fetch_adjacent_chunks,
            )
        except (RetrievalConfigurationError, RetrievalDataError, RetrievalRequestError, RetrievalUnavailableError):
            raise
        except Exception as error:
            raise RetrievalUnavailableError("RAG retrieval dependency failed") from error

        expanded_chunk_count = sum(len(context.chunks) - 1 for context in contexts)
        self._logger.info(
            "event=rag_search_completed collection=%s index_version=%s top_k=%d dense_candidate_count=%d bm25_candidate_count=%d fused_candidate_count=%d mmr_discarded_count=%d core_hit_count=%d expanded_chunk_count=%d result_count=%d duration_ms=%d",
            self._settings.collection_name,
            self._settings.index_version,
            top_k,
            trace.dense_candidate_count,
            trace.bm25_candidate_count,
            trace.fused_candidate_count,
            trace.mmr_discarded_count,
            len(core_chunks),
            expanded_chunk_count,
            len(contexts),
            int((time.perf_counter() - started_at) * 1000),
        )
        return [context.render() for context in contexts]

    def evaluate_core_chunks(self, query: str, top_k: int) -> dict[str, list[RetrievedChunk]]:
        """使用一次 query embedding 产出 Dense 基线与完整混合链路的核心结果。

        参数 query 为评测问题，top_k 为每种链路最终返回的核心数；返回包含 dense 与 hybrid 的 chunk 列表。
        本方法仅供离线评测使用，不渲染正文、不执行 P1，也不向日志写入问题原文。
        """

        if top_k <= 0 or top_k > self._settings.dense_top_k_max:
            raise RetrievalRequestError("top_k is outside the configured range")
        vector = self._embed_query(query)
        dense_chunks = self._query_dense_chunks(vector, top_k)
        hybrid_chunks, _ = self._query_hybrid_core_chunks(query, vector, top_k)
        return {"dense": dense_chunks, "hybrid": hybrid_chunks}

    def _validate_active_index(self) -> None:
        """校验 collection 维度及指定 index_version 至少包含一个 point。

        无参数；无返回值。版本未写入、collection 不存在或维度错误时抛出 RetrievalUnavailableError。
        """

        try:
            collection = self._qdrant.get_collection(self._settings.collection_name)
            vectors = collection.config.params.vectors
            dimensions = getattr(vectors, "size", None)
            if dimensions != self._settings.embedding_dimensions:
                raise RetrievalUnavailableError(
                    f"collection vector size mismatch: expected {self._settings.embedding_dimensions}, got {dimensions}"
                )
            count = self._qdrant.count(
                collection_name=self._settings.collection_name,
                count_filter=_index_version_filter(self._settings.index_version),
                exact=True,
            ).count
        except RetrievalUnavailableError:
            raise
        except Exception as error:
            raise RetrievalUnavailableError("RAG index is unavailable") from error
        if count <= 0:
            raise RetrievalUnavailableError("configured RAG_INDEX_VERSION has no indexed points")
        self._document_chunk_orders = self._build_document_chunk_orders()
        qdrant_chunk_ids = {
            chunk_id
            for document_order in self._document_chunk_orders.values()
            for chunk_id in document_order.values()
        }
        if self._bm25_index.index_version != self._settings.index_version:
            raise RetrievalUnavailableError("BM25 index_version does not match RAG_INDEX_VERSION")
        if self._bm25_index.collection_name != self._settings.collection_name:
            raise RetrievalUnavailableError("BM25 collection_name does not match current collection")
        if self._bm25_index.chunk_ids != qdrant_chunk_ids:
            raise RetrievalUnavailableError("BM25 chunk IDs do not match Qdrant index_version")
        if self._settings.candidate_top_k < self._settings.dense_top_k_max:
            raise RetrievalConfigurationError("candidate_top_k must be at least RAG_DENSE_TOP_K_MAX")

    def _build_document_chunk_orders(self) -> dict[str, dict[int, str]]:
        """一次性读取当前版本的顺序元数据，建立文档内 chunk 邻接映射。

        无参数；返回 document_id 到 chunk_order/chunk_id 的映射。只读取标识字段，
        不读取正文或向量；顺序重复或不连续时抛出 RetrievalDataError。
        """

        document_orders: dict[str, dict[int, str]] = {}
        offset: Any = None
        while True:
            records, offset = self._qdrant.scroll(
                collection_name=self._settings.collection_name,
                scroll_filter=_index_version_filter(self._settings.index_version),
                limit=256,
                offset=offset,
                with_payload=["document_id", "chunk_order", "chunk_id", "index_version"],
                with_vectors=False,
            )
            for record in records:
                payload = getattr(record, "payload", None)
                if not isinstance(payload, dict):
                    raise RetrievalDataError("Qdrant order payload is missing")
                document_id = payload.get("document_id")
                chunk_order = payload.get("chunk_order")
                chunk_id = payload.get("chunk_id")
                index_version = payload.get("index_version")
                if (
                    not isinstance(document_id, str)
                    or not isinstance(chunk_order, int)
                    or chunk_order < 0
                    or not isinstance(chunk_id, str)
                    or not chunk_id
                    or index_version != self._settings.index_version
                ):
                    raise RetrievalDataError("Qdrant order payload is invalid")
                document_order = document_orders.setdefault(document_id, {})
                if chunk_order in document_order:
                    raise RetrievalDataError("duplicate chunk_order within document")
                document_order[chunk_order] = chunk_id
            if offset is None:
                break

        for document_order in document_orders.values():
            if sorted(document_order) != list(range(len(document_order))):
                raise RetrievalDataError("chunk_order is not continuous within document")
        return document_orders

    def _embed_query(self, query: str) -> list[float]:
        """调用 embedding provider 生成并校验用户查询向量。

        参数 query 为原始用户问题；返回与 collection 维度一致的浮点向量，响应异常时抛出 RetrievalUnavailableError。
        """

        try:
            response = self._embedding_client.embeddings.create(
                model=self._settings.embedding_model,
                input=[query],
                dimensions=self._settings.embedding_dimensions,
            )
            if len(response.data) != 1 or response.data[0].index != 0:
                raise RetrievalUnavailableError("embedding response does not match query")
            vector = response.data[0].embedding
        except RetrievalUnavailableError:
            raise
        except Exception as error:
            raise RetrievalUnavailableError("query embedding failed") from error
        if not isinstance(vector, list) or len(vector) != self._settings.embedding_dimensions:
            raise RetrievalUnavailableError("query embedding dimension does not match collection")
        return vector

    def _query_dense_chunks(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """按当前 index_version 查询 Qdrant 的 Dense 核心命中。

        参数 vector 为已校验查询向量，top_k 为核心命中数；返回按 Qdrant 分数顺序排列的 chunk 列表。
        """

        result = self._qdrant.query_points(
            collection_name=self._settings.collection_name,
            query=vector,
            query_filter=_index_version_filter(self._settings.index_version),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        chunks = [RetrievedChunk.from_point(point) for point in result.points]
        return self._validate_version(chunks)

    def _query_core_chunks(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """兼容现有内部调用，按向量返回 Dense 核心命中。

        参数 vector 为已校验查询向量，top_k 为核心命中数；返回当前 index_version 的 Dense 排名结果。
        新的线上完整链路应使用 _query_hybrid_core_chunks。
        """

        return self._query_dense_chunks(vector, top_k)

    def _query_hybrid_core_chunks(
        self,
        query: str,
        vector: list[float],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        """执行 Dense、BM25、RRF 与 MMR，返回最终核心命中和可观测元数据。

        参数 query 为原始问题，vector 为该问题的已校验 embedding，top_k 为最终核心数；
        返回 MMR 选出的 chunk 和不含原文的 RetrievalTrace。Qdrant 向量或 BM25 候选不一致时抛出 RetrievalDataError。
        """

        dense_chunks = self._query_dense_chunks(vector, self._settings.candidate_top_k)
        bm25_candidates = self._bm25_index.search(query, self._settings.candidate_top_k)
        bm25_chunks = self._fetch_by_chunk_ids([candidate.chunk_id for candidate in bm25_candidates])
        if {chunk.chunk_id for chunk in bm25_chunks} != {candidate.chunk_id for candidate in bm25_candidates}:
            raise RetrievalDataError("Qdrant BM25 candidates do not match BM25 index")
        fused_candidates = _fuse_rrf(
            dense_chunks=dense_chunks,
            bm25_chunks=bm25_chunks,
            bm25_ranks={candidate.chunk_id: candidate.rank for candidate in bm25_candidates},
            rank_constant=self._settings.rrf_rank_constant,
            dense_weight=self._settings.rrf_dense_weight,
            bm25_weight=self._settings.rrf_bm25_weight,
        )
        vectors = self._fetch_candidate_vectors(fused_candidates)
        selected = _select_mmr(
            candidates=fused_candidates,
            candidate_vectors=vectors,
            top_k=top_k,
            mmr_lambda=self._settings.mmr_lambda,
        )
        trace = RetrievalTrace(
            dense_candidate_count=len(dense_chunks),
            bm25_candidate_count=len(bm25_candidates),
            fused_candidate_count=len(fused_candidates),
            mmr_discarded_count=len(fused_candidates) - len(selected),
        )
        return selected, trace

    def _fetch_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """按业务 chunk ID 读取前后邻居，并限制为当前 index_version。

        参数 chunk_ids 为 previous/next 业务 ID；返回读取到的合法 chunk 列表，空列表时不报错。
        """

        if not chunk_ids:
            return []
        records, _ = self._qdrant.scroll(
            collection_name=self._settings.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="index_version", match=MatchValue(value=self._settings.index_version)),
                    FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids)),
                ]
            ),
            limit=len(chunk_ids),
            with_payload=True,
            with_vectors=False,
        )
        chunks = self._validate_version([RetrievedChunk.from_point(record) for record in records])
        chunks_by_id: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            if chunk.chunk_id in chunks_by_id:
                raise RetrievalDataError("Qdrant returned duplicate chunk_id")
            chunks_by_id[chunk.chunk_id] = chunk
        return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]

    def _fetch_candidate_vectors(self, candidates: list[FusionCandidate]) -> dict[str, list[float]]:
        """读取 RRF 候选的 Qdrant 向量，用于 MMR 相互相似度计算。

        参数 candidates 为已融合且带 point ID 的候选；返回以业务 chunk ID 为键的向量字典。
        任一候选缺向量、重复或维度不匹配时抛出 RetrievalDataError。
        """

        if not candidates:
            return {}
        point_to_chunk_id = {candidate.chunk.point_id: candidate.chunk.chunk_id for candidate in candidates}
        if len(point_to_chunk_id) != len(candidates):
            raise RetrievalDataError("RRF candidates contain duplicate Qdrant point ID")
        try:
            records = self._qdrant.retrieve(
                collection_name=self._settings.collection_name,
                ids=list(point_to_chunk_id),
                with_payload=False,
                with_vectors=True,
            )
        except Exception as error:
            raise RetrievalUnavailableError("Qdrant candidate vectors are unavailable") from error
        vectors: dict[str, list[float]] = {}
        for record in records:
            point_id = str(getattr(record, "id", ""))
            chunk_id = point_to_chunk_id.get(point_id)
            vector = getattr(record, "vector", None)
            if chunk_id is None or not isinstance(vector, list) or len(vector) != self._settings.embedding_dimensions:
                raise RetrievalDataError("Qdrant candidate vector is invalid")
            if not all(isinstance(value, (int, float)) for value in vector):
                raise RetrievalDataError("Qdrant candidate vector has invalid values")
            if chunk_id in vectors:
                raise RetrievalDataError("Qdrant returned duplicate candidate vector")
            vectors[chunk_id] = [float(value) for value in vector]
        expected_chunk_ids = {candidate.chunk.chunk_id for candidate in candidates}
        if set(vectors) != expected_chunk_ids:
            raise RetrievalDataError("Qdrant candidate vectors do not match fused candidates")
        return vectors

    def _fetch_adjacent_chunks(self, core: RetrievedChunk) -> list[RetrievedChunk]:
        """读取当前核心命中文档顺序中确实存在的前后两个 chunk。

        参数 core 为 Dense 核心命中；返回最多两个相邻 chunk。文档开头或结尾不存在的顺序不会产生 Qdrant 查询。
        """

        chunk_ids = _adjacent_chunk_ids(self._document_chunk_orders, core)
        return self._fetch_by_chunk_ids(chunk_ids)

    def _validate_version(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """确认 Qdrant 返回 point 没有越过当前 index_version。

        参数 chunks 为已解析的 Qdrant 结果；返回原列表。发现越版本 payload 时抛出 RetrievalDataError。
        """

        for chunk in chunks:
            if chunk.index_version != self._settings.index_version:
                raise RetrievalDataError("Qdrant returned a point from a different index_version")
        return chunks


def _fuse_rrf(
    dense_chunks: list[RetrievedChunk],
    bm25_chunks: list[RetrievedChunk],
    bm25_ranks: dict[str, int],
    rank_constant: int,
    dense_weight: float,
    bm25_weight: float,
) -> list[FusionCandidate]:
    """按 Reciprocal Rank Fusion 合并 Dense 与 BM25 的业务 chunk 排名。

    参数 dense_chunks、bm25_chunks 为各自从高到低的候选，bm25_ranks 为 BM25 原始排名，
    rank_constant 为 YAML 配置的正整数，dense_weight 与 bm25_weight 为两路正权重；
    返回去重且稳定排序的 FusionCandidate 列表。
    """

    if rank_constant <= 0 or dense_weight <= 0 or bm25_weight <= 0:
        raise RetrievalDataError("RRF parameters must be positive")
    candidates: dict[str, FusionCandidate] = {}
    for rank, chunk in enumerate(dense_chunks, start=1):
        existing = candidates.get(chunk.chunk_id)
        candidates[chunk.chunk_id] = FusionCandidate(
            chunk=chunk,
            rrf_score=(existing.rrf_score if existing else 0.0) + dense_weight / (rank_constant + rank),
            dense_rank=rank,
            bm25_rank=existing.bm25_rank if existing else None,
        )
    for fallback_rank, chunk in enumerate(bm25_chunks, start=1):
        rank = bm25_ranks.get(chunk.chunk_id, fallback_rank)
        if not isinstance(rank, int) or rank <= 0:
            raise RetrievalDataError("BM25 candidate rank is invalid")
        existing = candidates.get(chunk.chunk_id)
        candidates[chunk.chunk_id] = FusionCandidate(
            chunk=chunk if existing is None else existing.chunk,
            rrf_score=(existing.rrf_score if existing else 0.0) + bm25_weight / (rank_constant + rank),
            dense_rank=existing.dense_rank if existing else None,
            bm25_rank=rank,
        )
    return sorted(candidates.values(), key=_fusion_sort_key)


def fuse_rrf(
    dense_chunks: list[RetrievedChunk],
    bm25_chunks: list[RetrievedChunk],
    bm25_ranks: dict[str, int],
    rank_constant: int,
    dense_weight: float,
    bm25_weight: float,
) -> list[FusionCandidate]:
    """公开复用 Dense 与 BM25 的稳定 RRF 融合实现。

    参数与内部 _fuse_rrf 相同，分别为两路候选、BM25 排名及 RRF 配置；返回按稳定规则排序的融合候选。
    OPERA paragraph Retriever 复用此函数，避免复制并分叉现有 Hybrid 排名逻辑。
    """

    return _fuse_rrf(
        dense_chunks=dense_chunks,
        bm25_chunks=bm25_chunks,
        bm25_ranks=bm25_ranks,
        rank_constant=rank_constant,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )


def _select_mmr(
    candidates: list[FusionCandidate],
    candidate_vectors: dict[str, list[float]],
    top_k: int,
    mmr_lambda: float,
) -> list[RetrievedChunk]:
    """以 RRF 相关性和候选间余弦相似度执行 Maximum Marginal Relevance 选择。

    参数 candidates 为 RRF 排序候选，candidate_vectors 为对应 Qdrant 向量，top_k 为最终核心数，
    mmr_lambda 为相关性权重；返回按 MMR 选择顺序排列的 chunk。输入缺向量或参数非法时抛出 RetrievalDataError。
    """

    if top_k <= 0:
        raise RetrievalDataError("MMR top_k must be positive")
    if not 0 < mmr_lambda <= 1:
        raise RetrievalDataError("MMR lambda must be within (0, 1]")
    if not candidates:
        return []
    expected_ids = {candidate.chunk.chunk_id for candidate in candidates}
    if set(candidate_vectors) != expected_ids:
        raise RetrievalDataError("MMR candidate vectors do not match fused candidates")

    ordered_candidates = sorted(candidates, key=_fusion_sort_key)
    relevance = _normalize_rrf_scores(ordered_candidates)
    selected: list[FusionCandidate] = []
    remaining = list(ordered_candidates)
    while remaining and len(selected) < top_k:
        scored_candidates: list[tuple[float, FusionCandidate]] = []
        for candidate in remaining:
            novelty_penalty = max(
                (
                    _cosine_similarity(candidate_vectors[candidate.chunk.chunk_id], candidate_vectors[chosen.chunk.chunk_id])
                    for chosen in selected
                ),
                default=0.0,
            )
            score = mmr_lambda * relevance[candidate.chunk.chunk_id] - (1.0 - mmr_lambda) * novelty_penalty
            scored_candidates.append((score, candidate))
        _, selected_candidate = min(scored_candidates, key=lambda item: (-item[0], _fusion_sort_key(item[1])))
        selected.append(selected_candidate)
        remaining = [candidate for candidate in remaining if candidate.chunk.chunk_id != selected_candidate.chunk.chunk_id]
    return [replace(candidate.chunk, score=candidate.rrf_score) for candidate in selected]


def select_mmr(
    candidates: list[FusionCandidate],
    candidate_vectors: dict[str, list[float]],
    top_k: int,
    mmr_lambda: float,
) -> list[RetrievedChunk]:
    """公开复用稳定的 MMR 去重实现。

    参数 candidates 为 RRF 结果，candidate_vectors 为候选向量，top_k 为保留数，mmr_lambda 为相关性权重；
    返回 MMR 选中的候选。OPERA 仅替换 paragraph 数据模型，不修改 V2 的排序语义。
    """

    return _select_mmr(
        candidates=candidates,
        candidate_vectors=candidate_vectors,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
    )


def _normalize_rrf_scores(candidates: list[FusionCandidate]) -> dict[str, float]:
    """将同一次 RRF 候选分数缩放到零到一的 MMR 相关性区间。

    参数 candidates 为至少一个融合候选；返回按 chunk ID 查找的归一化相关性。
    单个候选或分数完全相同时统一返回一，避免 MMR 因除零失效。
    """

    scores = [candidate.rrf_score for candidate in candidates]
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return {candidate.chunk.chunk_id: 1.0 for candidate in candidates}
    return {candidate.chunk.chunk_id: (candidate.rrf_score - low) / (high - low) for candidate in candidates}


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    """计算两个同维候选向量的余弦相似度。

    参数 first 与 second 为来自同一 Qdrant collection 的向量；返回有限的余弦值。
    维度不同、空向量或非有限数值时抛出 RetrievalDataError。
    """

    if not first or len(first) != len(second):
        raise RetrievalDataError("MMR vectors have inconsistent dimensions")
    dot_product = sum(left * right for left, right in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not math.isfinite(dot_product) or not math.isfinite(first_norm) or not math.isfinite(second_norm):
        raise RetrievalDataError("MMR vectors contain non-finite values")
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return dot_product / (first_norm * second_norm)


def _fusion_sort_key(candidate: FusionCandidate) -> tuple[float, int, int, int, str]:
    """构造 RRF 和 MMR 平分时使用的稳定候选排序键。

    参数 candidate 为融合候选；返回按 RRF 分数、最佳来源排名和 chunk ID 排序的元组。
    """

    ranks = [rank for rank in (candidate.dense_rank, candidate.bm25_rank) if rank is not None]
    best_rank = min(ranks) if ranks else math.inf
    return (
        -candidate.rrf_score,
        best_rank,
        candidate.dense_rank if candidate.dense_rank is not None else math.inf,
        candidate.bm25_rank if candidate.bm25_rank is not None else math.inf,
        candidate.chunk.chunk_id,
    )


def _index_version_filter(index_version: str) -> Filter:
    """构建当前 index_version 的 Qdrant 硬过滤器。

    参数 index_version 为显式指定的索引批次；返回仅匹配该批次的 Filter。
    """

    return Filter(must=[FieldCondition(key="index_version", match=MatchValue(value=index_version))])


def _is_p1_related(core: RetrievedChunk, candidate: RetrievedChunk) -> bool:
    """判断候选是否与核心命中满足已确认的 P1 关系。

    参数 core 为核心命中，candidate 为扩展候选；返回同文档同标题或同文档同非空直接父标题时为真。
    """

    if candidate.document_id != core.document_id:
        return False
    if candidate.section_id == core.section_id:
        return True
    return core.parent_section_id is not None and candidate.parent_section_id == core.parent_section_id


def _adjacent_chunk_ids(document_chunk_orders: dict[str, dict[int, str]], core: RetrievedChunk) -> list[str]:
    """从顺序映射中取得文档内真实存在的前后 chunk ID。

    参数 document_chunk_orders 为 document_id 到顺序映射，core 为核心命中；
    返回前一块、后一块的业务 ID。开头和结尾不存在的顺序不会出现在返回值中。
    """

    document_order = document_chunk_orders.get(core.document_id, {})
    return [
        chunk_id
        for order in (core.chunk_order - 1, core.chunk_order + 1)
        if (chunk_id := document_order.get(order)) is not None
    ]
