"""HotpotQA paragraph 导入、artifact 与 case/all BM25 检索测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opera.hotpot_bm25 import load_hotpot_bm25_index
from opera.hotpot_data import HotpotIndexConfig, build_hotpot_index_plan, load_hotpot_cases, write_hotpot_artifacts


class HotpotIndexTest(unittest.TestCase):
    """验证 HotpotQA 运行时语料不会带入答案标注，且 BM25 严格遵守 scope。"""

    def test_import_excludes_offline_labels_and_keeps_paragraph_boundary(self) -> None:
        """验证 context 的每个元素保留为一个 paragraph，answer 标注不会进入运行时数据模型。"""

        with tempfile.TemporaryDirectory() as directory:
            dataset_path = self._write_dataset(Path(directory))
            cases = load_hotpot_cases(dataset_path)

        self.assertEqual(2, len(cases))
        self.assertEqual(3, len(cases[0].paragraphs))
        self.assertEqual("case-a:0", cases[0].paragraphs[0].chunk_id)
        self.assertIn("Title: Apple", cases[0].paragraphs[0].embedding_text)
        self.assertEqual("", cases[0].paragraphs[0].sentences[1])
        self.assertNotIn("  ", cases[0].paragraphs[0].text)
        self.assertFalse(hasattr(cases[0], "answer"))
        self.assertFalse(hasattr(cases[0], "supporting_facts"))

    def test_scope_bm25_uses_shared_tokens_and_lru_cache(self) -> None:
        """验证 all 使用全量语料，而 case 只返回当前题候选且第二次查询命中缓存。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_hotpot_index_plan(self._write_dataset(root), self._config())
            _, bm25_path = write_hotpot_artifacts(plan, root / "out")
            index = load_hotpot_bm25_index(
                bm25_path,
                expected_index_version=plan.index_version,
                expected_collection_name=plan.collection_name,
                expected_tokenizer_version=plan.config.bm25_tokenizer_version,
                case_cache_size=2,
            )
            first_case = index.search("apple orchard", scope="case", case_id="case-a", limit=3)
            second_case = index.search("apple orchard", scope="case", case_id="case-a", limit=3)
            all_scope = index.search("apple orchard", scope="all", case_id=None, limit=4)

        self.assertFalse(first_case.cache_hit)
        self.assertTrue(second_case.cache_hit)
        self.assertTrue(first_case.candidates)
        self.assertTrue(all(candidate.chunk_id.startswith("case-a:") for candidate in first_case.candidates))
        self.assertIn("case-b:0", {candidate.chunk_id for candidate in all_scope.candidates})

    def test_plan_and_artifact_are_versioned_by_dataset_contract(self) -> None:
        """验证索引计划包含独立 collection、64 位版本号和可追溯 artifact。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_hotpot_index_plan(self._write_dataset(root), self._config())
            manifest_path, bm25_path = write_hotpot_artifacts(plan, root / "out")
            manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
            bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))

        self.assertTrue(plan.collection_name.startswith("hotpot_distractor_v1_"))
        self.assertEqual(64, len(plan.index_version))
        self.assertEqual(6, len(plan.records))
        self.assertEqual("hotpot_manifest_metadata", json.loads(manifest_lines[0])["record_type"])
        self.assertEqual("hotpot_bm25_index", bm25["record_type"])
        self.assertEqual({"case-a", "case-b"}, set(bm25["case_chunk_ids"]))

    def _config(self) -> HotpotIndexConfig:
        """构造测试数据集共用的独立 OPERA 索引配置。"""

        return HotpotIndexConfig(
            collection_prefix="hotpot_distractor_v1",
            embedding_model="test-embedding",
            embedding_dimensions=4,
            embedding_batch_size=2,
            bm25_tokenizer_version="jieba-identifier-v1",
        )

    def _write_dataset(self, root: Path) -> Path:
        """写入最小 HotpotQA JSON fixture 并返回文件路径。"""

        dataset_path = root / "hotpot.json"
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "_id": "case-a",
                        "question": "Which fruit grows in the orchard?",
                        "answer": "apple",
                        "supporting_facts": [["Apple", 0]],
                        "context": [
                            ["Apple", ["Apple trees grow in an orchard.", "", "The fruit can be red."]],
                            ["River", ["Water flows to the sea."]],
                            ["Forest", ["Pine trees grow in a forest."]],
                        ],
                    },
                    {
                        "_id": "case-b",
                        "question": "Which fruit is sold?",
                        "answer": "apple",
                        "supporting_facts": [["Market", 0]],
                        "context": [
                            ["Market", ["An apple is sold at the market."]],
                            ["Mountain", ["A mountain has snow."]],
                            ["Library", ["A library stores books."]],
                        ],
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return dataset_path


if __name__ == "__main__":
    unittest.main()
