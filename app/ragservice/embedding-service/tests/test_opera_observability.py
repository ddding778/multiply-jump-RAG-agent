"""OPERA Langfuse prompt、隐私摘要与 token 观测的无外部依赖测试。"""

from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from langfuse.api.commons.errors.not_found_error import NotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opera.llm import PromptDefinition, PromptResolver, ResponseOutputTextError, ResponsesClient
from opera.observability import LangfuseSettings, OperaObservability
from opera.prompt_sync import sync_prompt_definitions
from opera.schemas import PlanResult


class FakeObservation:
    """记录 Langfuse SDK observation 的初始化参数与更新内容。"""

    def __init__(self, payload: dict[str, object]) -> None:
        """保存 observation 初始化参数并初始化 update 列表。

        参数 payload 为 start_as_current_observation 收到的参数；无返回值。
        """

        self.payload = payload
        self.updates: list[dict[str, object]] = []

    def __enter__(self) -> "FakeObservation":
        """模拟 Langfuse context manager 的进入操作。

        无参数；返回自身，供被测代码调用 update。
        """

        return self

    def __exit__(self, *_: object) -> None:
        """模拟 Langfuse context manager 的退出操作。

        参数为 context manager 异常信息；无返回值，不吞掉异常。
        """

        return None

    def update(self, **payload: object) -> None:
        """记录一次 observation 更新。

        参数 payload 为 Langfuse update 关键字参数；无返回值。
        """

        self.updates.append(payload)


class FakeRemotePrompt:
    """模拟可编译的 Langfuse text prompt。"""

    version = 7

    def compile(self) -> str:
        """返回模拟的远端纯文本 prompt。

        无参数；返回可发送到模型的非空文本。
        """

        return "remote prompt"


class FakeLangfuse:
    """同时模拟 prompt 读取和 observation 创建的 Langfuse 客户端。"""

    def __init__(self) -> None:
        """初始化请求记录和远端 prompt。

        无参数；无返回值。
        """

        self.observations: list[FakeObservation] = []
        self.get_prompt_calls: list[dict[str, object]] = []
        self.remote_prompt = FakeRemotePrompt()

    def get_prompt(self, name: str, **kwargs: object) -> FakeRemotePrompt:
        """记录 prompt 查询并返回固定远端 prompt。

        参数 name 为 prompt 名称，kwargs 为标签、类型和缓存参数；返回 FakeRemotePrompt。
        """

        self.get_prompt_calls.append({"name": name, **kwargs})
        return self.remote_prompt

    def start_as_current_observation(self, **payload: object) -> FakeObservation:
        """创建并记录一个模拟 observation。

        参数 payload 为 observation 属性；返回支持 context manager 的 FakeObservation。
        """

        observation = FakeObservation(payload)
        self.observations.append(observation)
        return observation

    def flush(self) -> None:
        """模拟 SDK 刷新操作。

        无参数；无返回值。
        """


class FakePromptPublisher:
    """记录 Langfuse prompt 创建请求的替身。"""

    def __init__(self) -> None:
        """初始化创建请求列表。"""

        self.requests: list[dict[str, object]] = []
        self.prompts: dict[tuple[str, str], str] = {}

    def get_prompt(self, name: str, **kwargs: object) -> SimpleNamespace:
        """按名称和标签返回已创建的模拟远端 prompt。

        参数 name 为 prompt 名称，kwargs 包含目标标签；不存在时模拟 Langfuse 404，返回可 compile 的对象。
        """

        label = str(kwargs["label"])
        content = self.prompts.get((name, label))
        if content is None:
            raise NotFoundError({})
        return SimpleNamespace(compile=lambda: content)

    def create_prompt(self, **request: object) -> None:
        """记录一次 prompt 创建请求。

        参数 request 为同步模块传给 Langfuse SDK 的创建参数；无返回值。
        """

        self.requests.append(request)
        self.prompts[(str(request["name"]), str(request["labels"][0]))] = str(request["prompt"])


class FakeResponses:
    """返回带 OpenAI-compatible usage 的固定 JSON Schema 响应。"""

    def __init__(self) -> None:
        """暴露与 OpenAI 客户端兼容的 responses 属性。

        无参数；无返回值。
        """

        self.responses = self
        self.last_request: dict[str, object] = {}

    def create(self, **request: object) -> SimpleNamespace:
        """生成一个合法计划及包含缓存 token 的 usage。

        参数为被测客户端传入的 Responses 参数；返回模拟响应对象。
        """

        self.last_request = request
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "steps": [
                        {
                            "step_id": "step_1",
                            "subgoal": "find fact",
                            "depends_on": [],
                            "is_final": True,
                        }
                    ]
                }
            ),
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=3,
                input_tokens_details=SimpleNamespace(cached_tokens=4),
            ),
        )


class EmptyOutputResponses:
    """返回没有 output_text 的 provider 响应，用于验证错误观测。"""

    def __init__(self) -> None:
        """暴露与 OpenAI 客户端兼容的 responses 属性。"""

        self.responses = self

    def create(self, **_: object) -> SimpleNamespace:
        """返回带失败状态和非文本 output 项的模拟响应。"""

        return SimpleNamespace(
            id="response-test",
            status="failed",
            output_text="",
            output=[SimpleNamespace(type="reasoning")],
            error=SimpleNamespace(type="server_error", code="provider_error", message="hidden"),
            incomplete_details=None,
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=0,
                input_tokens_details=SimpleNamespace(cached_tokens=4),
            ),
        )


class OperaObservabilityTest(unittest.TestCase):
    """验证远端 prompt、嵌套 generation 与默认隐私策略。"""

    def test_prompt_resolver_reads_production_text_prompt_with_cache(self) -> None:
        """验证远端 prompt 使用指定 label、text 类型和 YAML 缓存秒数。"""

        fake_langfuse = FakeLangfuse()
        resolver = PromptResolver(fake_langfuse, "production", 300, logging.getLogger("test"))
        local_path = Path(__file__).resolve().parents[1] / "opera" / "prompts" / "planner_system.md"

        prompt = resolver.resolve("opera-planner-system", local_path)

        self.assertEqual("langfuse", prompt.source)
        self.assertEqual("remote prompt", prompt.content)
        self.assertEqual(7, prompt.version)
        self.assertEqual(
            [{"name": "opera-planner-system", "type": "text", "label": "production", "cache_ttl_seconds": 300}],
            fake_langfuse.get_prompt_calls,
        )

    def test_generation_masks_raw_input_and_tracks_exclusive_usage(self) -> None:
        """验证默认不开原文采集且缓存 token 不会与 input 重复计费。"""

        fake_langfuse = FakeLangfuse()
        observability = OperaObservability(
            fake_langfuse,
            LangfuseSettings(True, "production", 300, False),
            logging.getLogger("test"),
        )
        prompt = PromptDefinition(
            "remote prompt",
            "opera-planner-system",
            "langfuse",
            "production",
            7,
            fake_langfuse.remote_prompt,
        )
        fake_responses = FakeResponses()
        client = ResponsesClient(
            fake_responses,
            "deepseek-chat",
            400,
            timeout_seconds=20,
            observability=observability,
        )

        with observability.agent("planner"):
            result = client.complete(prompt, "Question:\nsecret question", PlanResult, "opera_plan")

        self.assertEqual("step_1", result.steps[0].step_id)
        self.assertEqual(20, fake_responses.last_request["timeout"])
        self.assertEqual({"effort": "high"}, fake_responses.last_request["reasoning"])
        generation = fake_langfuse.observations[-1]
        self.assertEqual("generation", generation.payload["as_type"])
        self.assertEqual("deepseek-chat", generation.payload["model"])
        self.assertIs(fake_langfuse.remote_prompt, generation.payload["prompt"])
        self.assertEqual({"captured": False, "character_count": len("Question:\nsecret question")}, generation.payload["input"])
        self.assertNotIn("secret question", str(generation.payload))
        self.assertEqual(
            {"input": 6, "cache_read_input_tokens": 4, "output": 3},
            generation.updates[-1]["usage_details"],
        )

    def test_generation_records_safe_metadata_for_empty_provider_output(self) -> None:
        """验证空文本响应会保留状态和 output 类型，但不保留 provider 错误正文。"""

        fake_langfuse = FakeLangfuse()
        observability = OperaObservability(
            fake_langfuse,
            LangfuseSettings(True, "production", 300, True),
            logging.getLogger("test"),
        )
        client = ResponsesClient(EmptyOutputResponses(), "deepseek-v4-flash", 800, observability=observability)

        with self.assertRaises(ResponseOutputTextError):
            client.complete("prompt", "Question:\nq", PlanResult, "opera_plan")

        generation = fake_langfuse.observations[-1]
        metadata = generation.updates[-1]["metadata"]
        self.assertEqual("provider_empty_output", metadata["result"])
        self.assertEqual("failed", metadata["provider_response_status"])
        self.assertEqual(["reasoning"], metadata["provider_output_item_types"])
        self.assertEqual("provider_error", metadata["provider_error_code"])
        self.assertNotIn("hidden", str(metadata))

    def test_prompt_sync_accepts_candidate_label_without_changing_runtime_config(self) -> None:
        """验证候选同步显式使用 candidate，而不改写 YAML 的 production 标签。"""

        publisher = FakePromptPublisher()
        config = SimpleNamespace(
            prompts={
                "planner": "opera-planner-system",
                "analysis_answer": "opera-analysis-answer-system",
                "rewriter": "opera-rewrite-system",
            },
            prompt_label="production",
        )

        synced = sync_prompt_definitions(publisher, config, "candidate")

        self.assertEqual(3, synced)
        self.assertEqual(["candidate", "candidate", "candidate"], [request["labels"][0] for request in publisher.requests])
        self.assertTrue(all(request["type"] == "text" and request["prompt"] for request in publisher.requests))
        self.assertEqual(0, sync_prompt_definitions(publisher, config, "candidate"))
        self.assertEqual(3, len(publisher.requests))


if __name__ == "__main__":
    unittest.main()
