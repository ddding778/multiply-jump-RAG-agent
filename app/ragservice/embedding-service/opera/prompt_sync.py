"""将 OPERA 本地 system prompt 显式同步为 Langfuse production text prompt。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_index.retrieval_config import OperaLangfuseConfig, load_opera_runtime_config

from .llm import load_local_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[4]
logger = logging.getLogger("rag_service")


def sync_prompt_definitions(client: object, config: OperaLangfuseConfig) -> int:
    """将三份本地 system prompt 创建或更新为指定标签的 Langfuse text prompt。

    参数 client 为已认证的 Langfuse SDK 客户端，config 为 YAML Langfuse 配置；
    返回成功同步的 prompt 数量。同名 prompt 会由 Langfuse 创建新版本，不会修改本地 fallback 文件。
    """

    prompt_dir = Path(__file__).resolve().parent / "prompts"
    local_paths = {
        "planner": prompt_dir / "planner_system.md",
        "analysis_answer": prompt_dir / "analysis_answer_system.md",
        "rewriter": prompt_dir / "rewrite_system.md",
    }
    synced = 0
    for agent_name, local_path in local_paths.items():
        prompt_name = config.prompts[agent_name]
        content = load_local_prompt(local_path)
        client.create_prompt(
            name=prompt_name,
            type="text",
            prompt=content,
            labels=[config.prompt_label],
        )
        synced += 1
        logger.info("event=opera_prompt_synced prompt_name=%s agent=%s", prompt_name, agent_name)
    return synced


def main() -> None:
    """从 .env 加载凭据并执行一次显式的 OPERA prompt 同步。

    无参数；无返回值。缺少凭据时抛出 RuntimeError；成功后刷新 SDK 事件并记录数量，
    不打印 prompt 内容或任何密钥。
    """

    load_dotenv(PROJECT_ROOT / ".env")
    missing = [
        name
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL are required")

    from langfuse import get_client

    config = load_opera_runtime_config().langfuse
    client = get_client()
    synced = sync_prompt_definitions(client, config)
    client.flush()
    logger.info("event=opera_prompt_sync_completed prompt_count=%s", synced)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
