"""MarkdownChunker 的结构与 token 边界测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_index.markdown_chunker import MarkdownChunker


class CharacterTokenCounter:
    """以字符数代替 token 数的确定性测试计数器。"""

    def count(self, text: str) -> int:
        """返回文本字符数，便于构造精确边界测试。

        参数 text 为待计数文本；返回字符数量。
        """

        return len(text)


class MarkdownChunkerTest(unittest.TestCase):
    """验证结构化切分器的核心合同。"""

    def setUp(self) -> None:
        """创建使用确定性计数器的切分器。

        无参数；无返回值。
        """

        self.chunker = MarkdownChunker(CharacterTokenCounter(), soft_limit_tokens=80, hard_limit_tokens=120)

    def test_heading_tree_and_next_heading_force_section_boundary(self) -> None:
        """验证跳级标题保留完整路径，且新标题即使未满也断开。"""

        markdown = "# Root\nroot intro\n\n### Child\nchild content\n\n## Sibling\nsibling content"
        chunks = self.chunker.chunk_markdown(markdown, "guide/structure.md")

        self.assertEqual(3, len(chunks))
        self.assertEqual(["Root"], [item.text for item in chunks[0].heading_path])
        self.assertEqual(["Root", "Child"], [item.text for item in chunks[1].heading_path])
        self.assertEqual(["Root", "Sibling"], [item.text for item in chunks[2].heading_path])
        self.assertEqual(chunks[0].section_id, chunks[1].parent_section_id)
        self.assertEqual(chunks[0].section_id, chunks[2].parent_section_id)

    def test_fenced_code_hash_is_not_a_heading(self) -> None:
        """验证 fenced code 中的井号不会改变标题路径。"""

        markdown = "# API\n说明。\n\n```go\n# not a heading\nfunc main() {}\n```"
        chunks = self.chunker.chunk_markdown(markdown, "guide/api.md")

        self.assertEqual(1, len(chunks))
        self.assertEqual(["API"], [item.text for item in chunks[0].heading_path])
        self.assertTrue(chunks[0].has_code)
        self.assertIn("# not a heading", chunks[0].text)

    def test_toc_control_blocks_are_not_indexed(self) -> None:
        """验证独立 TOC 控制标记会跳过，但短正文仍保留。"""

        markdown = "[TOC]\n\n<!-- TOC -->\n\n# 结论\nOK"
        chunks = self.chunker.chunk_markdown(markdown, "guide/toc.md")

        self.assertEqual(1, len(chunks))
        self.assertEqual("OK", chunks[0].text)
        self.assertEqual(["结论"], [item.text for item in chunks[0].heading_path])

    def test_sentence_boundary_precedes_hard_limit(self) -> None:
        """验证超长正文优先在句末而不是硬阈值中间切开。"""

        markdown = "# Topic\n" + "甲" * 65 + "。" + "乙" * 65 + "。"
        chunks = self.chunker.chunk_markdown(markdown, "guide/limit.md")

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(item.token_count <= 120 for item in chunks))
        self.assertTrue(chunks[0].text.endswith("。"))
        self.assertEqual(["Topic"], [item.text for item in chunks[1].heading_path])

    def test_single_sentence_over_hard_limit_has_safe_fallback(self) -> None:
        """验证单句本身超限时仍不会生成超过硬上限的 embedding 文本。"""

        markdown = "# Topic\n" + "a" * 300
        chunks = self.chunker.chunk_markdown(markdown, "guide/fallback.md")

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(item.token_count <= 120 for item in chunks))
        self.assertEqual("".join(item.text for item in chunks), "a" * 300)

    def test_neighbors_only_link_inside_same_section(self) -> None:
        """验证前后邻居不会跨越标题区间。"""

        markdown = "# First\n" + ("甲" * 65 + "。") * 2 + "\n\n# Second\nsecond"
        chunks = self.chunker.chunk_markdown(markdown, "guide/neighbors.md")
        first_section = [item for item in chunks if item.heading_path[-1].text == "First"]
        second_section = [item for item in chunks if item.heading_path[-1].text == "Second"]

        self.assertGreaterEqual(len(first_section), 2)
        self.assertEqual(first_section[1].chunk_id, first_section[0].next_chunk_id)
        self.assertEqual(first_section[0].chunk_id, first_section[1].previous_chunk_id)
        self.assertIsNone(second_section[0].previous_chunk_id)

    def test_chunk_order_is_monotonic_within_a_document(self) -> None:
        """验证 chunk_order 跨标题区间仍按文档顺序连续递增。"""

        markdown = "# First\n" + ("甲" * 65 + "。") * 2 + "\n\n# Second\nsecond"
        chunks = self.chunker.chunk_markdown(markdown, "guide/order.md")

        self.assertEqual(list(range(len(chunks))), [item.chunk_order for item in chunks])

    def test_manifest_fields_are_serializable(self) -> None:
        """验证公开字段包含后续 manifest 所需的切分元数据。"""

        chunk = self.chunker.chunk_markdown("# Title\nbody", "guide/manifest.md")[0]
        payload = chunk.as_dict()

        self.assertTrue(
            {
                "document_id",
                "section_id",
                "parent_section_id",
                "chunk_id",
                "chunk_order",
                "heading_path",
                "line_start",
                "line_end",
                "content_hash",
                "previous_chunk_id",
                "next_chunk_id",
                "has_code",
                "embedding_text",
            }.issubset(payload)
        )
        self.assertRegex(str(payload["content_hash"]), r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
