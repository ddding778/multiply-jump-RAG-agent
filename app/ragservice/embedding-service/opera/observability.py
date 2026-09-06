"""OPERA 的 Langfuse 可选观测封装。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from typing import Iterator


@dataclass(frozen=True)
class LangfuseSettings:
    """保存 OPERA 使用的非敏感 Langfuse 运行参数。

    参数 enabled 表示是否尝试启用远端观测，prompt_label 为读取 prompt 的版本标签，
    prompt_cache_ttl_seconds 为客户端缓存时长，capture_input_output 控制是否采集原始业务文本；
    返回对象仅描述配置，不保存凭据。
    """

    enabled: bool
    prompt_label: str
    prompt_cache_ttl_seconds: int
    capture_input_output: bool


class _Observation:
    """统一处理真实与空 Langfuse observation 的更新接口。"""

    def __init__(self, raw_observation: object | None, capture_input_output: bool) -> None:
        """保存原始 observation 与原文采集开关。

        参数 raw_observation 为 Langfuse SDK 创建的 observation 或 None，
        capture_input_output 决定 input/output 是否可以包含业务原文；无返回值。
        """

        self._raw_observation = raw_observation
        self._capture_input_output = capture_input_output

    def update(
        self,
        *,
        input_data: object | None = None,
        output_data: object | None = None,
        metadata: dict[str, object] | None = None,
        usage_details: dict[str, int] | None = None,
    ) -> None:
        """安全更新 observation，不启用原文采集时仅提交长度摘要。

        参数 input_data 与 output_data 为可选业务内容，metadata 为非敏感运行标签，
        usage_details 为已归一化 token 桶；无返回值。SDK 不可用时本方法静默跳过。
        """

        if self._raw_observation is None:
            return

        payload: dict[str, object] = {}
        if input_data is not None:
            payload["input"] = self._safe_payload(input_data)
        if output_data is not None:
            payload["output"] = self._safe_payload(output_data)
        if metadata:
            payload["metadata"] = metadata
        if usage_details:
            payload["usage_details"] = usage_details
        if payload:
            self._raw_observation.update(**payload)

    def _safe_payload(self, value: object) -> object:
        """按采集开关返回原文或不可逆的长度摘要。

        参数 value 为待写入 Langfuse 的输入或输出；返回原始对象或仅含字符数的摘要，
        不会在关闭采集时保留原始业务内容。
        """

        if self._capture_input_output:
            return value
        return {"captured": False, "character_count": len(str(value))}


class OperaObservability:
    """为 OPERA 请求、Agent、Retriever 与 Generation 建立嵌套 Langfuse 观测。"""

    def __init__(
        self,
        client: object | None,
        settings: LangfuseSettings,
        logger: logging.Logger,
    ) -> None:
        """保存 Langfuse 客户端和运行开关。

        参数 client 为已初始化的 Langfuse SDK 客户端或 None，settings 为非敏感配置，
        logger 仅记录事件和异常类型；无返回值。
        """

        self._client = client
        self._settings = settings
        self._logger = logger

    @property
    def prompt_client(self) -> object | None:
        """返回可读取 Langfuse prompt 的客户端。

        无参数；返回 SDK 客户端或 None，调用方据此决定是否使用本地 fallback。
        """

        return self._client

    @classmethod
    def create(cls, settings: LangfuseSettings, logger: logging.Logger) -> "OperaObservability":
        """按配置和环境变量创建可降级的 Langfuse 观测器。

        参数 settings 为 YAML 中的开关和采集策略，logger 为事件日志；返回可用观测器。
        关闭开关、缺少凭据、依赖未安装或 SDK 初始化失败时均返回本地 no-op 观测器。
        """

        if not settings.enabled:
            return cls(None, settings, logger)

        missing = [
            name
            for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
            if not os.getenv(name, "").strip()
        ]
        if missing:
            logger.warning(
                "event=opera_langfuse_disabled reason=credentials_missing missing_count=%s",
                len(missing),
            )
            return cls(None, settings, logger)

        try:
            from langfuse import get_client

            client = get_client()
        except Exception as error:
            logger.warning(
                "event=opera_langfuse_disabled reason=client_initialization_failed error_type=%s",
                type(error).__name__,
            )
            return cls(None, settings, logger)

        logger.info("event=opera_langfuse_enabled capture_input_output=%s", settings.capture_input_output)
        return cls(client, settings, logger)

    @contextmanager
    def request(
        self,
        *,
        run_id: str,
        question: str,
        retrieval_scope: str,
        has_case_id: bool,
        max_steps: int,
        max_rewrites: int,
    ) -> Iterator[_Observation]:
        """创建一条 OPERA 请求根 Agent observation。

        参数包含运行 ID、问题、范围和执行上限；返回可更新的 observation 上下文。
        问题只有在显式开启 capture_input_output 时才会被发送到 Langfuse。
        """

        metadata = {
            "feature": "opera",
            "run_id": run_id,
            "retrieval_scope": retrieval_scope,
            "has_case_id": has_case_id,
            "max_steps": max_steps,
            "max_rewrites": max_rewrites,
        }
        with self._open(
            name="opera-ask",
            observation_type="agent",
            metadata=metadata,
            input_data={"question": question},
        ) as observation:
            yield observation

    @contextmanager
    def agent(self, role: str, metadata: dict[str, object] | None = None) -> Iterator[_Observation]:
        """创建一个嵌套的 OPERA 子 Agent observation。

        参数 role 为稳定的 Agent 角色名，metadata 为不含业务原文的补充标签；
        返回该 Agent 的 observation 上下文。
        """

        with self._open(
            name=f"opera-{role}",
            observation_type="agent",
            metadata=metadata or {},
        ) as observation:
            yield observation

    @contextmanager
    def retriever(self, metadata: dict[str, object] | None = None, input_data: object | None = None) -> Iterator[_Observation]:
        """创建一个嵌套的 Hybrid Retriever observation。

        参数 metadata 为检索范围、候选数等标签，input_data 为查询内容；返回检索 observation。
        查询原文遵循全局 capture_input_output 开关。
        """

        with self._open(
            name="opera-hybrid-retrieval",
            observation_type="retriever",
            metadata=metadata or {},
            input_data=input_data,
        ) as observation:
            yield observation

    @contextmanager
    def generation(
        self,
        *,
        name: str,
        model: str,
        schema_name: str,
        attempt: int,
        prompt: object | None,
        prompt_metadata: dict[str, object],
        input_data: object,
    ) -> Iterator[_Observation]:
        """创建一个带模型、prompt 与 Schema 元数据的 Generation observation。

        参数 name、model、schema_name 与 attempt 标识模型调用，prompt 为远端 Langfuse prompt，
        prompt_metadata 标识 fallback 或远端版本，input_data 为本次模型输入；返回 generation 上下文。
        """

        metadata = {
            "provider": "deepseek",
            "schema_name": schema_name,
            "attempt": attempt,
            **prompt_metadata,
        }
        with self._open(
            name=name,
            observation_type="generation",
            metadata=metadata,
            input_data=input_data,
            model=model,
            prompt=prompt,
        ) as observation:
            yield observation

    def flush(self) -> None:
        """在服务退出时尝试刷出异步 Langfuse 事件。

        无参数；无返回值。刷新失败只记录异常类型，不影响 RAG 服务的退出流程。
        """

        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as error:
            self._logger.warning("event=opera_langfuse_flush_failed error_type=%s", type(error).__name__)

    @contextmanager
    def _open(
        self,
        *,
        name: str,
        observation_type: str,
        metadata: dict[str, object],
        input_data: object | None = None,
        model: str | None = None,
        prompt: object | None = None,
    ) -> Iterator[_Observation]:
        """按需打开一个 SDK observation，未启用时提供空实现。

        参数为 observation 的名称、类型、元数据、可选输入、模型与 prompt；返回统一更新接口。
        SDK 调用失败时不影响业务执行，并记录不含业务原文的异常类型。
        """

        if self._client is None:
            yield _Observation(None, self._settings.capture_input_output)
            return

        payload: dict[str, object] = {
            "as_type": observation_type,
            "name": name,
            "metadata": metadata,
        }
        if input_data is not None:
            payload["input"] = self._payload_for_start(input_data)
        if model is not None:
            payload["model"] = model
        if prompt is not None:
            payload["prompt"] = prompt

        try:
            observation_context = self._client.start_as_current_observation(**payload)
        except Exception as error:
            self._logger.warning(
                "event=opera_langfuse_observation_failed name=%s error_type=%s",
                name,
                type(error).__name__,
            )
            yield _Observation(None, self._settings.capture_input_output)
            return

        try:
            raw_observation = observation_context.__enter__()
        except Exception as error:
            self._logger.warning(
                "event=opera_langfuse_observation_failed name=%s error_type=%s",
                name,
                type(error).__name__,
            )
            yield _Observation(None, self._settings.capture_input_output)
            return

        try:
            yield _Observation(raw_observation, self._settings.capture_input_output)
        except BaseException as error:
            try:
                observation_context.__exit__(type(error), error, error.__traceback__)
            except Exception as exit_error:
                self._logger.warning(
                    "event=opera_langfuse_observation_exit_failed name=%s error_type=%s",
                    name,
                    type(exit_error).__name__,
                )
            raise
        else:
            try:
                observation_context.__exit__(None, None, None)
            except Exception as error:
                self._logger.warning(
                    "event=opera_langfuse_observation_exit_failed name=%s error_type=%s",
                    name,
                    type(error).__name__,
                )

    def _payload_for_start(self, value: object) -> object:
        """为 observation 创建初始 input，遵守原文采集开关。

        参数 value 为原始输入；返回原始对象或不可逆长度摘要。
        """

        if self._settings.capture_input_output:
            return value
        return {"captured": False, "character_count": len(str(value))}
