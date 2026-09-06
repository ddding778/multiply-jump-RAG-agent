"""OPERA 单请求本地调试轨迹写入器。"""

from __future__ import annotations

import json
import logging
from pathlib import Path


class OperaTraceWriter:
    """将已脱敏的 OPERA 执行轨迹写入本地 out 目录。"""

    def __init__(self, enabled: bool, output_directory: Path, logger: logging.Logger) -> None:
        """保存写入开关、输出目录和结构化日志器。

        参数 enabled 控制是否写入轨迹，output_directory 为已解析的本地 out 子目录，logger 只记录 run_id、
        路径与异常类型；无返回值。
        """

        self._enabled = enabled
        self._output_directory = output_directory
        self._logger = logger

    def write(self, run_id: str, trace: dict[str, object]) -> None:
        """将一条请求的轨迹作为独占 JSON 文件写入本地目录。

        参数 run_id 为请求唯一标识，trace 为不包含原始用户问题的执行轨迹；无返回值。
        目录创建、序列化和写入异常只记录类型，不影响 OPERA 的 HTTP 响应。
        """

        if not self._enabled:
            return

        output_path = self._output_directory / f"{run_id}.json"
        try:
            self._output_directory.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as file:
                json.dump(trace, file, ensure_ascii=False, indent=2)
        except Exception as error:
            self._logger.warning(
                "event=opera_trace_write_failed run_id=%s error_type=%s",
                run_id,
                type(error).__name__,
            )
            return

        self._logger.info("event=opera_trace_written run_id=%s path=%s", run_id, output_path)
