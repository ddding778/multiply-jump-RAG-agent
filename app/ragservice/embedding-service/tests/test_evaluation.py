"""RAG 双基线评测 schema 与指标计算测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_index.evaluation import EvaluationCase, _rank_metrics, load_evaluation_cases


class EvaluationTest(unittest.TestCase):
    """验证评测集的两个 schema 与基础指标计算。"""

    def test_loads_single_and_multiple_relevant_chunk_schemas(self) -> None:
        """验证旧单目标和融合多目标 schema 都能提取正确 chunk ID。"""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            path.write_text(
                json.dumps({"case_id": "a", "query": "one", "target_chunk_id": "chunk-a"})
                + "\n"
                + json.dumps(
                    {
                        "case_id": "b",
                        "query": "two",
                        "stress_type": "lexical_anchor",
                        "relevant_chunks": [{"chunk_id": "chunk-b"}, {"chunk_id": "chunk-c"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cases = load_evaluation_cases(path)

        self.assertEqual({"chunk-a"}, cases[0].relevant_chunk_ids)
        self.assertEqual({"chunk-b", "chunk-c"}, cases[1].relevant_chunk_ids)
        self.assertEqual("lexical_anchor", cases[1].stress_type)

    def test_rank_metrics_use_first_relevant_result(self) -> None:
        """验证 HitRate 和 MRR 只按每题第一个相关 chunk 的排名计分。"""

        metrics = _rank_metrics(
            rows=[
                {"rank": 1},
                {"rank": 2},
                {"rank": None},
            ],
            cutoffs=[1, 3],
            top_k_max=3,
        )

        self.assertEqual(0.3333, metrics["HitRate@1"])
        self.assertEqual(0.6667, metrics["HitRate@3"])
        self.assertEqual(0.5, metrics["MRR@3"])


if __name__ == "__main__":
    unittest.main()
