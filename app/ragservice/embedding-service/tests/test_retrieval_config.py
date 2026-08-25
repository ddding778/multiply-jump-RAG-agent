"""RAG V2 YAML 检索配置测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_index.retrieval_config import RetrievalConfigError, collection_name_for_version, load_retrieval_algorithm_config


class RetrievalConfigTest(unittest.TestCase):
    """验证 collection 命名和 YAML 配置的严格校验。"""

    def test_config_loads_required_values(self) -> None:
        """验证完整 YAML 可转换为明确的运行时配置对象。"""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rag-retrieval.yaml"
            path.write_text(
                "collection_prefix: tech_docs_v2\n"
                "bm25:\n  candidate_top_k: 12\n  tokenizer_version: jieba-identifier-v1\n"
                "rrf:\n  rank_constant: 60\n  dense_weight: 1.0\n  bm25_weight: 0.5\n"
                "mmr:\n  lambda: 0.7\n",
                encoding="utf-8",
            )
            config = load_retrieval_algorithm_config(path)

        self.assertEqual("tech_docs_v2", config.collection_prefix)
        self.assertEqual(12, config.bm25_candidate_top_k)
        self.assertEqual(0.5, config.rrf_bm25_weight)
        self.assertEqual(0.7, config.mmr_lambda)

    def test_invalid_mmr_lambda_is_rejected(self) -> None:
        """验证 MMR 权重超出合法区间时配置加载失败。"""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rag-retrieval.yaml"
            path.write_text(
                "collection_prefix: tech_docs_v2\n"
                "bm25:\n  candidate_top_k: 10\n  tokenizer_version: jieba-identifier-v1\n"
                "rrf:\n  rank_constant: 60\n  dense_weight: 1.0\n  bm25_weight: 0.5\n"
                "mmr:\n  lambda: 0\n",
                encoding="utf-8",
            )

            with self.assertRaises(RetrievalConfigError):
                load_retrieval_algorithm_config(path)

    def test_collection_name_is_version_scoped(self) -> None:
        """验证 collection 名称含完整版本，避免不同批次覆盖。"""

        index_version = "a" * 64
        self.assertEqual("tech_docs_v2_" + index_version, collection_name_for_version("tech_docs_v2", index_version))
        with self.assertRaises(RetrievalConfigError):
            collection_name_for_version("tech docs", index_version)


if __name__ == "__main__":
    unittest.main()
