"""HotpotQA embedding checkpoint 与批次恢复测试。"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opera.embedding_checkpoint import (
    checkpoint_path_for_plan,
    completed_record_count,
    load_or_initialize_checkpoint,
    mark_batch_completed,
    next_uncompleted_start,
    save_checkpoint,
)
from opera.hotpot_data import HotpotIndexConfig, build_hotpot_index_plan
from opera.hotpot_runner import DEFAULT_EMBEDDING_BATCH_SIZE, batch_is_already_upserted, build_config, embed_and_upsert


@dataclass(frozen=True)
class FakePoint:
    """表示测试时由 Qdrant 回读的最小 point。"""

    id: str
    payload: dict[str, object]


class FakeQdrant:
    """记录测试中的 Qdrant 回读和写入调用。"""

    def __init__(self, recovered_points: list[FakePoint]) -> None:
        """使用预置回读结果构造测试客户端。"""

        self.recovered_points = recovered_points
        self.retrieve_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> list[FakePoint]:
        """记录精确 ID 回读请求并返回预置 point。"""

        self.retrieve_calls.append(kwargs)
        return self.recovered_points

    def upsert(self, **kwargs: object) -> None:
        """记录一次同步 Qdrant upsert 请求。"""

        self.upsert_calls.append(kwargs)


class FakeEmbeddings:
    """记录 embedding 文本数量并构造固定维度响应。"""

    def __init__(self, dimensions: int) -> None:
        """使用指定向量维度初始化测试响应生成器。"""

        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        """记录输入并返回与输入顺序一致的 embedding 响应。"""

        inputs = kwargs["input"]
        if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
            raise AssertionError("embedding input must be a list of strings")
        self.calls.append(inputs)
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=[float(index)] * self.dimensions) for index in range(len(inputs))]
        )


class FakeEmbeddingClient:
    """暴露 OpenAI-compatible embeddings 属性的最小测试客户端。"""

    def __init__(self, dimensions: int) -> None:
        """使用指定向量维度构造内部 FakeEmbeddings。"""

        self.embeddings = FakeEmbeddings(dimensions)


class HotpotEmbeddingCheckpointTest(unittest.TestCase):
    """验证 checkpoint 能绑定索引合同并在中断后避免重复 embedding。"""

    def test_checkpoint_round_trip_rejects_different_batch_contract(self) -> None:
        """验证 checkpoint 记录 batch size，避免用不同恢复粒度继续旧任务。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._build_plan(root, batch_size=2)
            checkpoint_path = checkpoint_path_for_plan(root / "out", plan)
            checkpoint, existed = load_or_initialize_checkpoint(checkpoint_path, plan)
            self.assertFalse(existed)
            save_checkpoint(checkpoint_path, checkpoint)
            checkpoint = mark_batch_completed(checkpoint, 0, plan.records[:2])
            save_checkpoint(checkpoint_path, checkpoint)

            restored, restored_existed = load_or_initialize_checkpoint(checkpoint_path, plan)
            self.assertTrue(restored_existed)
            self.assertEqual(2, completed_record_count(restored))
            self.assertEqual(2, next_uncompleted_start(restored))

            different_plan = self._build_plan(root, batch_size=1)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_or_initialize_checkpoint(checkpoint_path, different_plan)

    def test_existing_checkpoint_reconciles_first_unconfirmed_batch(self) -> None:
        """验证 Qdrant 已写而 checkpoint 未推进时不重复调用该批 embedding。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._build_plan(root, batch_size=2)
            checkpoint_path = checkpoint_path_for_plan(root / "out", plan)
            checkpoint, _ = load_or_initialize_checkpoint(checkpoint_path, plan)
            save_checkpoint(checkpoint_path, checkpoint)
            recovered_points = [
                FakePoint(
                    id=record.point_id,
                    payload={
                        "index_version": plan.index_version,
                        "embedding_input_hash": record.embedding_input_hash,
                    },
                )
                for record in plan.records[:2]
            ]
            qdrant = FakeQdrant(recovered_points)
            embedding_client = FakeEmbeddingClient(plan.config.embedding_dimensions)

            embed_and_upsert(
                qdrant,
                embedding_client,
                plan,
                checkpoint_path,
                logging.getLogger("hotpot_embedding_checkpoint_test"),
            )
            completed, existed = load_or_initialize_checkpoint(checkpoint_path, plan)

        self.assertTrue(existed)
        self.assertEqual(3, completed_record_count(completed))
        self.assertEqual(1, len(qdrant.retrieve_calls))
        self.assertEqual(1, len(qdrant.upsert_calls))
        self.assertEqual(1, len(embedding_client.embeddings.calls))
        self.assertEqual(1, len(embedding_client.embeddings.calls[0]))

    def test_batch_readback_requires_matching_point_payload(self) -> None:
        """验证恢复回读会拒绝索引版本或输入摘要不一致的 point。"""

        with tempfile.TemporaryDirectory() as directory:
            plan = self._build_plan(Path(directory), batch_size=2)
            mismatched = FakeQdrant(
                [
                    FakePoint(
                        id=plan.records[0].point_id,
                        payload={"index_version": "old-version", "embedding_input_hash": plan.records[0].embedding_input_hash},
                    )
                ]
            )

            result = batch_is_already_upserted(mismatched, plan, plan.records[:2])

        self.assertFalse(result)
        self.assertEqual([record.point_id for record in plan.records[:2]], mismatched.retrieve_calls[0]["ids"])

    def test_default_batch_size_and_known_provider_limit_are_enforced(self) -> None:
        """验证默认 batch 为官方上限 20，超出 qwen3.7 限制时在请求前失败。"""

        args = SimpleNamespace(
            retrieval_config=Path("unused.yaml"),
            embedding_dimensions=1024,
            embedding_batch_size=21,
            embedding_model="qwen3.7-text-embedding",
        )
        with patch("opera.hotpot_runner.load_opera_retrieval_config", return_value=SimpleNamespace(collection_prefix="hotpot")):
            with patch("opera.hotpot_runner.load_retrieval_algorithm_config", return_value=SimpleNamespace(bm25_tokenizer_version="jieba")):
                with self.assertRaisesRegex(ValueError, "limit of 20"):
                    build_config(args)

        self.assertEqual(20, DEFAULT_EMBEDDING_BATCH_SIZE)

    def _build_plan(self, root: Path, batch_size: int):
        """创建包含三个 paragraph 的确定性 HotpotQA 索引计划。"""

        dataset_path = root / "hotpot.json"
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "_id": "case-a",
                        "question": "Which fruit grows in the orchard?",
                        "context": [
                            ["Apple", ["Apple trees grow in an orchard."]],
                            ["River", ["Water flows to the sea."]],
                            ["Forest", ["Pine trees grow in a forest."]],
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        return build_hotpot_index_plan(
            dataset_path,
            HotpotIndexConfig(
                collection_prefix="hotpot_distractor_v1",
                embedding_model="qwen3.7-text-embedding",
                embedding_dimensions=4,
                embedding_batch_size=batch_size,
                bm25_tokenizer_version="jieba-identifier-v1",
            ),
        )


if __name__ == "__main__":
    unittest.main()
