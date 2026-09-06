"""DeepSeek Responses 严格 JSON Schema 调用与 prompt fallback 封装。"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from uuid import uuid4
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .observability import LangfuseSettings, OperaObservability
from .events import emit


T = TypeVar("T", bound=BaseModel)


class ResponseOutputTextError(ValueError):
    """表示 provider 返回了没有可用文本输出的 Responses 对象。"""

    def __init__(self, schema_name: str, provider_metadata: dict[str, object]) -> None:
        """保存调用的 Schema 名称及不含正文的 provider 响应元信息。

        参数 schema_name 标识发生异常的 Agent Schema，provider_metadata 包含状态、错误类别和 output 项类型；无返回值。
        """

        super().__init__("Responses output_text must be non-empty text")
        self.schema_name = schema_name
        self.provider_metadata = provider_metadata


@dataclass(frozen=True)
class PromptDefinition:
    """表示已解析的 Agent system prompt。

    参数 content 为实际发送给模型的纯文本，name 为稳定 prompt 名称，source 表示远端或本地来源，
    label 与 version 用于观测，remote_prompt 为可链接 Langfuse generation 的 SDK 对象；返回对象不可变。
    """

    content: str
    name: str
    source: str
    label: str | None
    version: int | None
    remote_prompt: object | None

    def trace_metadata(self) -> dict[str, object]:
        """返回不含 prompt 正文的 Langfuse generation 元数据。

        无参数；返回 prompt 名称、来源、标签及可选版本，供 generation 过滤和审计使用。
        """

        metadata: dict[str, object] = {
            "prompt_name": self.name,
            "prompt_source": self.source,
        }
        if self.label is not None:
            metadata["prompt_label"] = self.label
        if self.version is not None:
            metadata["prompt_version"] = self.version
        return metadata


class PromptResolver:
    """优先读取 Langfuse text prompt，失败时回退本地 Markdown 文件。"""

    def __init__(
        self,
        langfuse_client: object | None,
        prompt_label: str,
        cache_ttl_seconds: int,
        logger: logging.Logger,
    ) -> None:
        """保存可选 SDK 客户端与本地 fallback 配置。

        参数 langfuse_client 为 Langfuse 客户端或 None，prompt_label 为版本标签，
        cache_ttl_seconds 为 prompt 缓存秒数，logger 只记录名称和异常类型；无返回值。
        """

        self._langfuse_client = langfuse_client
        self._prompt_label = prompt_label
        self._cache_ttl_seconds = cache_ttl_seconds
        self._logger = logger

    def resolve(self, name: str, local_path: Path) -> PromptDefinition:
        """解析一个 Agent 的生产 prompt，并保证本地文件始终可用。

        参数 name 为 Langfuse prompt 名，local_path 为本地 Markdown fallback；返回 PromptDefinition。
        远端不可达、类型错误或未配置客户端时返回本地内容，不会使 OPERA 服务不可用。
        """

        local_content = load_local_prompt(local_path)
        local_prompt = PromptDefinition(
            content=local_content,
            name=name,
            source="local_fallback",
            label=None,
            version=None,
            remote_prompt=None,
        )
        if self._langfuse_client is None:
            emit("initialization", stage="prompt", message=f"{name} 使用本地提示词。")
            return local_prompt

        emit("initialization", stage="prompt", message=f"正在读取 {name}，远端超时将回退本地提示词…")
        try:
            remote_prompt = self._langfuse_client.get_prompt(
                name,
                type="text",
                label=self._prompt_label,
                cache_ttl_seconds=self._cache_ttl_seconds,
                max_retries=0,
                fetch_timeout_seconds=3,
            )
            content = remote_prompt.compile()
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Langfuse prompt must compile to non-empty text")
            version = getattr(remote_prompt, "version", None)
            if not isinstance(version, int):
                version = None
            return PromptDefinition(
                content=content.strip(),
                name=name,
                source="langfuse",
                label=self._prompt_label,
                version=version,
                remote_prompt=remote_prompt,
            )
        except Exception as error:
            self._logger.warning(
                "event=opera_prompt_fallback prompt_name=%s error_type=%s",
                name,
                type(error).__name__,
            )
        emit("dependency_warning", category="langfuse", message=f"{name} 远端读取失败，已回退本地提示词，继续执行。")
        return local_prompt


class ResponsesClient:
    """封装不持久化对话状态的 Responses API 调用。"""

    def __init__(
        self,
        client: object,
        model: str,
        max_output_tokens: int,
        max_repair_attempts: int = 1,
        timeout_seconds: int | None = None,
        observability: OperaObservability | None = None,
        reasoning_effort: str = "high",
    ) -> None:
        """保存 provider 客户端、模型、输出限制、思考强度、Schema 修复次数与可选观测器。

        参数 client 为 OpenAI-compatible Responses 客户端，model 为模型名，max_output_tokens 为软输出上限，
        max_repair_attempts 为最大修复次数，timeout_seconds 为可选单请求超时，observability 为可选 Langfuse 观测器，
        reasoning_effort 为 DeepSeek Responses API 的思考强度；无返回值。
        """

        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._max_repair_attempts = max_repair_attempts
        self._timeout_seconds = timeout_seconds
        self._observability = observability or OperaObservability.create(
            settings=LangfuseSettings(False, "production", 0, False),
            logger=logging.getLogger("rag_service"),
        )
        self._reasoning_effort = reasoning_effort

    def complete(
        self,
        prompt: PromptDefinition | str,
        input_text: str,
        schema_type: type[T],
        schema_name: str,
    ) -> T:
        """使用远端 JSON Schema 并用本地 Pydantic 再次校验返回。

        参数 prompt 为已解析 prompt 或测试使用的字符串，input_text 为业务输入，schema_type 为本地 Pydantic 类型，
        schema_name 为远端 Schema 名称；返回已验证对象。校验失败时最多追加固定修复指令并重试指定次数。
        """

        resolved_prompt = _coerce_prompt(prompt)
        current_input = input_text
        for attempt in range(self._max_repair_attempts + 1):
            call_id = str(uuid4())
            started = time.monotonic()
            emit("agent_started", call_id=call_id, agent=schema_name, attempt=attempt + 1,
                 model=self._model, instructions=resolved_prompt.content, input=current_input,
                 prompt_metadata=resolved_prompt.trace_metadata())
            with self._observability.generation(
                name=f"{schema_name}-generation",
                model=self._model,
                schema_name=schema_name,
                attempt=attempt + 1,
                prompt=resolved_prompt.remote_prompt,
                prompt_metadata={
                    **resolved_prompt.trace_metadata(),
                    "max_output_tokens": self._max_output_tokens,
                    "reasoning_effort": self._reasoning_effort,
                    "timeout_seconds": self._timeout_seconds,
                },
                input_data=current_input,
            ) as generation:
                try:
                    request: dict[str, object] = {
                        "model": self._model,
                        "instructions": resolved_prompt.content,
                        "input": current_input,
                        "max_output_tokens": self._max_output_tokens,
                        "reasoning": {"effort": self._reasoning_effort},
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "strict": True,
                                "schema": schema_type.model_json_schema(),
                            }
                        },
                    }
                    if self._timeout_seconds is not None:
                        request["timeout"] = self._timeout_seconds
                    response = self._client.responses.create(**request)
                except Exception as error:
                    emit("agent_failed", call_id=call_id, error_type=type(error).__name__)
                    generation.update(metadata={"result": "provider_error", "error_type": type(error).__name__})
                    raise

                usage_details = _extract_usage_details(response)
                provider_metadata = _provider_response_metadata(response)
                try:
                    output_text = _response_output_text(response)
                except ValueError as error:
                    emit("agent_failed", call_id=call_id, error_type="ResponseOutputTextError",
                         provider_metadata=provider_metadata)
                    generation.update(
                        metadata={
                            "result": "provider_empty_output",
                            "schema_validation": "skipped",
                            **provider_metadata,
                        },
                        usage_details=usage_details,
                    )
                    raise ResponseOutputTextError(schema_name, provider_metadata) from error
                emit("agent_output", call_id=call_id, output=output_text,
                     duration_ms=round((time.monotonic() - started) * 1000), usage=usage_details)
                try:
                    parsed = schema_type.model_validate_json(output_text)
                except ValidationError:
                    emit("agent_validated", call_id=call_id, validation="failed",
                         will_retry=attempt < self._max_repair_attempts)
                    generation.update(
                        output_data=output_text,
                        metadata={"schema_validation": "failed"},
                        usage_details=usage_details,
                    )
                    if attempt == self._max_repair_attempts:
                        raise
                else:
                    emit("agent_validated", call_id=call_id, validation="passed", will_retry=False)
                    generation.update(
                        output_data=output_text,
                        metadata={"schema_validation": "passed"},
                        usage_details=usage_details,
                    )
                    return parsed

            current_input = f"{input_text}\n\nReturn only a response that exactly satisfies the JSON Schema."
        raise RuntimeError("unreachable Responses schema retry state")


def load_local_prompt(path: Path) -> str:
    """读取本地 fallback system prompt，空文件或不可读时抛出 ValueError。

    参数 path 为 UTF-8 prompt 文件路径；返回去除首尾空白后的非空 prompt 文本。
    """

    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError("local prompt cannot be read") from error
    if not prompt:
        raise ValueError("local prompt must not be empty")
    return prompt


def _coerce_prompt(prompt: PromptDefinition | str) -> PromptDefinition:
    """将兼容测试的纯字符串转换为本地 prompt 定义。

    参数 prompt 为 PromptDefinition 或纯文本；返回完整 PromptDefinition，纯文本仅用于本地测试，
    不会链接任何 Langfuse 远端 prompt。
    """

    if isinstance(prompt, PromptDefinition):
        return prompt
    return PromptDefinition(prompt, "local-test-prompt", "local_fallback", None, None, None)


def _response_output_text(response: object) -> str:
    """提取 OpenAI-compatible Responses 的文本输出。

    参数 response 为 provider 返回对象；返回非空字符串。输出字段缺失或类型错误时抛出 ValueError，
    避免将非文本或空结果交给 Pydantic 解析。
    """

    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Responses output_text must be non-empty text")
    return output_text


def _provider_response_metadata(response: object) -> dict[str, object]:
    """提取不含模型正文的 Responses 状态元信息。

    参数 response 为 provider 返回对象；返回 response 状态、错误码、不完整原因和 output 项类型，
    用于区分空文本、失败和截断，不保存模型正文或用户输入。
    """

    output_text = _read_string_field(response, "output_text")
    output = _read_field(response, "output")
    metadata: dict[str, object] = {
        "provider_response_id": _read_string_field(response, "id"),
        "provider_response_status": _read_string_field(response, "status"),
        "provider_output_text_state": "nonempty" if output_text and output_text.strip() else "empty_or_missing",
        "provider_output_count": len(output) if isinstance(output, list) else None,
    }

    error = _read_field(response, "error")
    if error is not None:
        metadata["provider_error_type"] = _read_string_field(error, "type")
        metadata["provider_error_code"] = _read_string_field(error, "code")

    incomplete_details = _read_field(response, "incomplete_details")
    if incomplete_details is not None:
        metadata["provider_incomplete_reason"] = _read_string_field(incomplete_details, "reason")

    if isinstance(output, list):
        item_types: list[str] = []
        content_types: list[str] = []
        for item in output:
            item_type = _read_string_field(item, "type")
            if item_type is not None:
                item_types.append(item_type)
            content = _read_field(item, "content")
            if isinstance(content, list):
                for content_item in content:
                    content_type = _read_string_field(content_item, "type")
                    if content_type is not None:
                        content_types.append(content_type)
        metadata["provider_output_item_types"] = item_types
        metadata["provider_output_content_types"] = content_types

    return {name: value for name, value in metadata.items() if value is not None}


def _read_field(source: object | None, name: str) -> object | None:
    """从对象或字典读取一个字段，不对 provider 返回结构作额外假设。

    参数 source 为 provider 返回对象或字典，name 为字段名；返回字段值或 None。
    """

    if source is None:
        return None
    return source.get(name) if isinstance(source, dict) else getattr(source, name, None)


def _read_string_field(source: object | None, name: str) -> str | None:
    """读取非空字符串字段，避免把任意 provider 正文写入观测元数据。

    参数 source 为 provider 返回对象或字典，name 为字段名；返回去除首尾空白后的字符串或 None。
    """

    value = _read_field(source, name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _extract_usage_details(response: object) -> dict[str, int] | None:
    """将兼容 Responses 的 usage 转换为 Langfuse 互斥 token 桶。

    参数 response 为 provider 返回对象；返回 input、cache_read_input_tokens、output 的非负整数映射，
    usage 缺失或无法可靠解析时返回 None。缓存 token 会从 input 中扣除，避免成本被重复统计。
    """

    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = _read_nonnegative_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _read_nonnegative_int(usage, "output_tokens", "completion_tokens")
    input_details = getattr(usage, "input_tokens_details", None) or getattr(usage, "prompt_tokens_details", None)
    cached_tokens = _read_nonnegative_int(
        input_details,
        "cached_tokens",
        "cache_read_input_tokens",
        "prompt_cache_hit_tokens",
    )
    if input_tokens is None and output_tokens is None:
        return None

    details: dict[str, int] = {}
    if input_tokens is not None:
        details["input"] = max(0, input_tokens - (cached_tokens or 0))
    if cached_tokens:
        details["cache_read_input_tokens"] = cached_tokens
    if output_tokens is not None:
        details["output"] = output_tokens
    return details or None


def _read_nonnegative_int(source: object | None, *names: str) -> int | None:
    """从对象或字典读取首个非负整数属性。

    参数 source 为 provider usage 对象或字典，names 为按优先级查找的字段名；返回整数或 None。
    布尔值与负数均不被视为合法 token 数。
    """

    if source is None:
        return None
    for name in names:
        value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None
