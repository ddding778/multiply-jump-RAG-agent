"""RAG V2 BM25 分词、artifact 与运行时加载测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_index.bm25_index import Bm25Document, Bm25IndexError, load_bm25_index, tokenize_for_bm25, write_bm25_index


class Bm25IndexTest(unittest.TestCase):
    """验证中文、标识符和版本绑定的 BM25 artifact 合同。"""

    def test_tokenizer_keeps_identifier_and_splits_components(self) -> None:
        """验证 camelCase、snake_case、路径和中文词都能进入 token 序列。"""

        tokens = tokenize_for_bm25("解释 runtime.Gosched 与 goroutine_scheduler 的调度", "jieba-identifier-v1")

        self.assertIn("runtime.gosched", tokens)
        self.assertIn("runtime", tokens)
        self.assertIn("gosched", tokens)
        self.assertIn("goroutine_scheduler", tokens)
        self.assertIn("goroutine", tokens)
        self.assertIn("scheduler", tokens)
        self.assertTrue(any("调度" in token for token in tokens))

    def test_artifact_requires_matching_version_collection_and_tokenizer(self) -> None:
        """验证 artifact 只会在完整索引合同一致时加载。"""

        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "bm25_index.json"
            write_bm25_index(
                documents=[
                    Bm25Document(chunk_id="chunk-a", text="标题路径 A\nGoroutine 调度"),
                    Bm25Document(chunk_id="chunk-b", text="标题路径 B\nTCP 协议"),
                    Bm25Document(chunk_id="chunk-c", text="标题路径 C\n内存分配"),
                ],
                output_path=artifact_path,
                index_version="a" * 64,
                collection_name="tech_docs_v2_" + "a" * 64,
                tokenizer_version="jieba-identifier-v1",
            )
            index = load_bm25_index(
                artifact_path,
                expected_index_version="a" * 64,
                expected_collection_name="tech_docs_v2_" + "a" * 64,
                expected_tokenizer_version="jieba-identifier-v1",
            )

            self.assertEqual(["chunk-a"], [candidate.chunk_id for candidate in index.search("goroutine", 3)])
            with self.assertRaises(Bm25IndexError):
                load_bm25_index(
                    artifact_path,
                    expected_index_version="b" * 64,
                    expected_collection_name="tech_docs_v2_" + "a" * 64,
                    expected_tokenizer_version="jieba-identifier-v1",
                )

    def test_corrupt_artifact_chunk_count_is_rejected(self) -> None:
        """验证 artifact 声明数量和实际 entries 不一致时不会静默加载。"""

        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "bm25_index.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "record_type": "bm25_index",
                        "schema_version": 1,
                        "index_version": "a" * 64,
                        "collection_name": "tech_docs_v2_" + "a" * 64,
                        "tokenizer_version": "jieba-identifier-v1",
                        "chunk_count": 2,
                        "entries": [{"chunk_id": "chunk-a", "tokens": ["a"]}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(Bm25IndexError):
                load_bm25_index(
                    artifact_path,
                    expected_index_version="a" * 64,
                    expected_collection_name="tech_docs_v2_" + "a" * 64,
                    expected_tokenizer_version="jieba-identifier-v1",
                )


if __name__ == "__main__":
    unittest.main()
