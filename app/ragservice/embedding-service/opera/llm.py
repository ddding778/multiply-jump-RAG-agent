"""DeepSeek Responses 严格 JSON Schema 调用与 prompt fallback 封装。"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .observability import LangfuseSettings, OperaObservability


T = TypeVar("T", bound=BaseModel)


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
            return local_prompt

        try:
            remote_prompt = self._langfuse_client.get_prompt(
                name,
                type="text",
                label=self._prompt_label,
                cache_ttl_seconds=self._cache_ttl_seconds,
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
    ) -> None:
        """保存 provider 客户端、模型、输出限制、Schema 修复次数与可选观测器。

        参数 client 为 OpenAI-compatible Responses 客户端，model 为模型名，max_output_tokens 为软输出上限，
        max_repair_attempts 为最大修复次数，timeout_seconds 为可选单请求超时，observability 为可选 Langfuse 观测器；无返回值。
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
            with self._observability.generation(
                name=f"{schema_name}-generation",
                model=self._model,
                schema_name=schema_name,
                attempt=attempt + 1,
                prompt=resolved_prompt.remote_prompt,
                prompt_metadata={
                    **resolved_prompt.trace_metadata(),
                    "max_output_tokens": self._max_output_tokens,
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
                    generation.update(metadata={"result": "provider_error", "error_type": type(error).__name__})
                    raise

                output_text = _response_output_text(response)
                usage_details = _extract_usage_details(response)
                try:
                    parsed = schema_type.model_validate_json(output_text)
                except ValidationError:
                    generation.update(
                        output_data=output_text,
                        metadata={"schema_validation": "failed"},
                        usage_details=usage_details,
                    )
                    if attempt == self._max_repair_attempts:
                        raise
                else:
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
