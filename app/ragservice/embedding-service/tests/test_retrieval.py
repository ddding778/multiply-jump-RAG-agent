"""RAG V2 在线 Dense 检索与 P1 上下文组装测试。"""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_index.retrieval import (
    FusionCandidate,
    P1ContextAssembler,
    RetrievedChunk,
    RetrievalDataError,
    _adjacent_chunk_ids,
    _fuse_rrf,
    _select_mmr,
)


def make_chunk(
    chunk_id: str,
    *,
    document_id: str = "document-a",
    section_id: str = "section-c",
    parent_section_id: str | None = "section-b",
    chunk_order: int = 10,
    token_count: int = 300,
    previous_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
    index_version: str = "version-a",
) -> RetrievedChunk:
    """构造 P1 测试所需的最小 chunk。

    参数可覆盖文档、标题关系、顺序、预算和邻居；返回携带完整 payload 语义的 RetrievedChunk。
    """

    return RetrievedChunk(
        point_id=f"point-{chunk_id}",
        score=0.9,
        document_id=document_id,
        section_id=section_id,
        parent_section_id=parent_section_id,
        chunk_id=chunk_id,
        chunk_order=chunk_order,
        heading_path=[{"level": 1, "text": "A"}, {"level": 2, "text": "B"}],
        text=f"body-{chunk_id}",
        line_start=chunk_order,
        line_end=chunk_order,
        token_count=token_count,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
        index_version=index_version,
    )


class P1ContextAssemblerTest(unittest.TestCase):
    """验证同标题、同父标题和 token 预算的 P1 合同。"""

    def setUp(self) -> None:
        """创建 800 不扩展、1200 总预算的 P1 编排器。

        无参数；无返回值。
        """

        self.assembler = P1ContextAssembler(no_expand_threshold=800, context_token_budget=1200)

    def test_adjacent_chunks_fill_budget_and_keep_same_parent_sibling(self) -> None:
        """验证仅相邻 chunk 参与 P1，且同父标题的相邻块可在预算内加入。"""

        core = make_chunk("core", token_count=400, chunk_order=10)
        previous = make_chunk("previous", chunk_order=9, token_count=300)
        next_chunk = make_chunk("next", section_id="section-d", chunk_order=11, token_count=300)

        groups = self.assembler.assemble(
            core_chunks=[core],
            fetch_adjacent_chunks=lambda _: [next_chunk, previous],
        )

        self.assertEqual(1, len(groups))
        self.assertEqual(["previous", "core", "next"], [chunk.chunk_id for chunk in groups[0].chunks])
        self.assertEqual(1000, sum(chunk.token_count for chunk in groups[0].chunks))

    def test_core_at_threshold_does_not_fetch_extensions(self) -> None:
        """验证 token_count 等于 800 的核心命中不触发任何 P1 候选读取。"""

        core = make_chunk("core", token_count=800)

        groups = self.assembler.assemble(
            core_chunks=[core],
            fetch_adjacent_chunks=lambda _: self.fail("threshold core must not fetch adjacent candidates"),
        )

        self.assertEqual(["core"], [chunk.chunk_id for chunk in groups[0].chunks])

    def test_cross_document_and_cross_parent_candidates_are_rejected(self) -> None:
        """验证 P1 不扩展到其他文档或不同直接父标题。"""

        core = make_chunk("core", token_count=300)
        other_document = make_chunk("other-document", document_id="document-b", token_count=100)
        other_parent = make_chunk("other-parent", section_id="section-x", parent_section_id="section-y", token_count=100)

        groups = self.assembler.assemble(
            core_chunks=[core],
            fetch_adjacent_chunks=lambda _: [other_document, other_parent],
        )

        self.assertEqual(["core"], [chunk.chunk_id for chunk in groups[0].chunks])

    def test_another_dense_core_is_not_added_as_an_extension(self) -> None:
        """验证已独立 Dense 命中的 chunk 不会被重复塞入其他核心的 P1 扩展。"""

        first = make_chunk("first", token_count=300, chunk_order=10)
        second = make_chunk("second", section_id="section-d", token_count=300, chunk_order=11)

        groups = self.assembler.assemble(
            core_chunks=[first, second],
            fetch_adjacent_chunks=lambda _: [second],
        )

        self.assertEqual(["first"], [chunk.chunk_id for chunk in groups[0].chunks])
        self.assertEqual(["second"], [chunk.chunk_id for chunk in groups[1].chunks])

    def test_context_render_keeps_heading_path_and_source_lines(self) -> None:
        """验证送入 Chat 的参考资料会保留标题路径和原始行区间。"""

        rendered = make_chunk("core", chunk_order=17).render_for_context()

        self.assertIn("标题路径：A > B", rendered)
        self.assertIn("来源行：17-17", rendered)
        self.assertIn("正文：\nbody-core", rendered)

    def test_document_boundaries_do_not_create_nonexistent_adjacent_ids(self) -> None:
        """验证文档开头和结尾只读取实际存在的一个相邻 chunk。"""

        document_order = {"document-a": {0: "first", 1: "last"}}
        first = make_chunk("first", chunk_order=0)
        last = make_chunk("last", chunk_order=1)

        self.assertEqual(["last"], _adjacent_chunk_ids(document_order, first))
        self.assertEqual(["first"], _adjacent_chunk_ids(document_order, last))


class RetrievedChunkTest(unittest.TestCase):
    """验证 Qdrant payload 解析的完整性保护。"""

    def test_invalid_payload_is_rejected(self) -> None:
        """验证缺失 token_count 的 point 不会静默进入检索结果。"""

        point = SimpleNamespace(
            id="point-invalid",
            score=0.8,
            payload={
                "document_id": "document-a",
                "section_id": "section-a",
                "chunk_id": "chunk-a",
                "text": "body",
                "index_version": "version-a",
                "chunk_order": 0,
                "line_start": 1,
                "line_end": 1,
                "heading_path": [],
            },
        )

        with self.assertRaises(RetrievalDataError):
            RetrievedChunk.from_point(point)


class HybridRankingTest(unittest.TestCase):
    """验证 RRF 融合与 MMR 去重的纯计算合同。"""

    def test_rrf_merges_same_chunk_and_keeps_stable_order(self) -> None:
        """验证两路都命中的 chunk 会累加 RRF 分数并排在仅单路命中项之前。"""

        dense_first = make_chunk("dense-first", chunk_order=1)
        shared = make_chunk("shared", chunk_order=2)
        bm25_only = make_chunk("bm25-only", chunk_order=3)

        fused = _fuse_rrf(
            dense_chunks=[dense_first, shared],
            bm25_chunks=[shared, bm25_only],
            bm25_ranks={"shared": 1, "bm25-only": 2},
            rank_constant=60,
            dense_weight=1.0,
            bm25_weight=1.0,
        )

        self.assertEqual(["shared", "dense-first", "bm25-only"], [candidate.chunk.chunk_id for candidate in fused])

    def test_mmr_prefers_less_redundant_candidate_after_best_result(self) -> None:
        """验证 MMR 在保留最高 RRF 命中的同时，会优先选择不重复的候选。"""

        first = FusionCandidate(make_chunk("first", chunk_order=1), rrf_score=0.9, dense_rank=1, bm25_rank=1)
        redundant = FusionCandidate(make_chunk("redundant", chunk_order=2), rrf_score=0.8, dense_rank=2, bm25_rank=2)
        diverse = FusionCandidate(make_chunk("diverse", chunk_order=3), rrf_score=0.7, dense_rank=3, bm25_rank=3)

        selected = _select_mmr(
            candidates=[first, redundant, diverse],
            candidate_vectors={
                "first": [1.0, 0.0],
                "redundant": [0.99, 0.01],
                "diverse": [0.0, 1.0],
            },
            top_k=2,
            mmr_lambda=0.5,
        )

        self.assertEqual(["first", "diverse"], [chunk.chunk_id for chunk in selected])


if __name__ == "__main__":
    unittest.main()
