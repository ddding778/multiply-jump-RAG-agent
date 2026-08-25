"""RAG V2 索引计划、payload 和 manifest 测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_index.indexer import IndexConfig, build_index_plan, build_payload, write_manifest


class CharacterTokenCounter:
    """以字符数代替 token 数的确定性测试计数器。"""

    def count(self, text: str) -> int:
        """返回文本字符数。

        参数 text 为待计数文本；返回字符数。
        """

        return len(text)


class IndexerTest(unittest.TestCase):
    """验证离线索引构建不依赖第三方网络服务。"""

    def setUp(self) -> None:
        """创建默认测试索引合同。

        无参数；无返回值。
        """

        self.config = IndexConfig(
            collection_prefix="tech_docs_v2",
            embedding_model="qwen3.7-text-embedding",
            embedding_dimensions=1024,
            tokenizer_encoding="cl100k_base",
            soft_limit_tokens=80,
            hard_limit_tokens=120,
            embedding_batch_size=10,
            bm25_tokenizer_version="jieba-identifier-v1",
        )

    def test_plan_payload_contains_retrieval_and_traceability_fields(self) -> None:
        """验证 payload 包含 P1 扩展和版本追溯所需字段。"""

        with tempfile.TemporaryDirectory() as directory:
            docs_dir = Path(directory)
            (docs_dir / "guide.md").write_text("# A\nintro\n\n## B\nbody", encoding="utf-8")
            plan = build_index_plan(docs_dir, self.config, CharacterTokenCounter())

        self.assertEqual(2, len(plan.records))
        record = plan.records[1]
        payload = build_payload(record, plan.index_version)
        self.assertEqual("guide.md", payload["source_path"])
        self.assertEqual(plan.records[0].chunk.section_id, payload["parent_section_id"])
        self.assertEqual(record.chunk.chunk_id, payload["chunk_id"])
        self.assertEqual(plan.index_version, payload["index_version"])
        self.assertEqual(1024, self.config.embedding_dimensions)
        self.assertEqual("tech_docs_v2_" + plan.index_version, plan.collection_name)
        self.assertRegex(record.point_id, r"^[0-9a-f-]{36}$")
        self.assertRegex(record.embedding_input_hash, r"^[0-9a-f]{64}$")

    def test_heading_change_reembeds_even_when_body_is_unchanged(self) -> None:
        """验证相同正文在标题路径变化时具有不同 embedding 输入版本。"""

        with tempfile.TemporaryDirectory() as directory:
            docs_dir = Path(directory)
            target = docs_dir / "guide.md"
            target.write_text("# Original\nsame body", encoding="utf-8")
            first = build_index_plan(docs_dir, self.config, CharacterTokenCounter())
            target.write_text("# Renamed\nsame body", encoding="utf-8")
            second = build_index_plan(docs_dir, self.config, CharacterTokenCounter())

        self.assertEqual(first.records[0].chunk.content_hash, second.records[0].chunk.content_hash)
        self.assertNotEqual(first.records[0].embedding_input_hash, second.records[0].embedding_input_hash)
        self.assertNotEqual(first.index_version, second.index_version)

    def test_manifest_has_metadata_then_qdrant_point_records(self) -> None:
        """验证 manifest 先写完整合同元数据，再写可复核的 point payload。"""

        with tempfile.TemporaryDirectory() as directory:
            docs_dir = Path(directory) / "docs"
            output_dir = Path(directory) / "out" / "rag-index" / "tech_docs_v2"
            docs_dir.mkdir()
            (docs_dir / "guide.md").write_text("# A\nbody", encoding="utf-8")
            plan = build_index_plan(docs_dir, self.config, CharacterTokenCounter())
            manifest_path = write_manifest(plan, output_dir)
            lines = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(plan.index_version, manifest_path.parent.name)
        self.assertEqual("manifest_metadata", lines[0]["record_type"])
        self.assertEqual(plan.index_version, lines[0]["index_version"])
        self.assertEqual(plan.collection_name, lines[0]["collection_name"])
        self.assertEqual("jieba-identifier-v1", lines[0]["bm25_tokenizer_version"])
        self.assertEqual("chunk", lines[1]["record_type"])
        self.assertEqual(plan.records[0].point_id, lines[1]["point_id"])
        self.assertNotIn("embedding_text", lines[1]["payload"])


if __name__ == "__main__":
    unittest.main()
