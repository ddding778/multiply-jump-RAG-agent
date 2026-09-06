"""将 OPERA 本地 system prompt 显式同步为指定 Langfuse text prompt 标签。"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse.api.commons.errors.not_found_error import NotFoundError

from rag_index.retrieval_config import OperaLangfuseConfig, load_opera_runtime_config

from .llm import load_local_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[4]
logger = logging.getLogger("rag_service")


def sync_prompt_definitions(client: object, config: OperaLangfuseConfig, label: str | None = None) -> int:
    """将三份本地 system prompt 创建或更新为指定标签的 Langfuse text prompt。

    参数 client 为已认证的 Langfuse SDK 客户端，config 为 YAML Langfuse 配置；
    label 为可选发布标签，缺省时使用 YAML 的生产标签；返回本次新建的 prompt 数量。
    同名且同标签内容已经一致时跳过；只有远端明确返回 404 时才创建新版本，网络异常会向上抛出，
    避免超时重试意外产生多余版本。本函数不会修改本地 fallback 文件。
    """

    selected_label = label or config.prompt_label
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
        if _remote_prompt_matches(client, prompt_name, selected_label, content):
            logger.info(
                "event=opera_prompt_sync_skipped prompt_name=%s agent=%s label=%s",
                prompt_name,
                agent_name,
                selected_label,
            )
            continue
        client.create_prompt(
            name=prompt_name,
            type="text",
            prompt=content,
            labels=[selected_label],
        )
        synced += 1
        logger.info(
            "event=opera_prompt_synced prompt_name=%s agent=%s label=%s",
            prompt_name,
            agent_name,
            selected_label,
        )
    return synced


def _remote_prompt_matches(client: object, name: str, label: str, content: str) -> bool:
    """检查指定标签的远端 text prompt 是否已与本地内容完全一致。

    参数 client 为已认证的 Langfuse SDK 客户端，name 为 prompt 名称，label 为目标标签，
    content 为本地非空 prompt 文本；远端明确不存在时返回 False，内容相同时返回 True。
    网络、鉴权或其他远端错误均向上抛出，调用方不得把不确定状态误判为可创建新版本。
    """

    try:
        remote_prompt = client.get_prompt(
            name,
            type="text",
            label=label,
            cache_ttl_seconds=0,
            max_retries=0,
            fetch_timeout_seconds=5000,
        )
    except NotFoundError:
        return False

    remote_content = remote_prompt.compile()
    if not isinstance(remote_content, str):
        raise ValueError("Langfuse text prompt must compile to text")
    return remote_content.strip() == content


def main() -> None:
    """从 .env 加载凭据并执行一次显式的 OPERA prompt 同步。

    可选命令行参数 `--label` 指定本次同步的 Langfuse 标签，默认使用 YAML 的生产标签；
    无返回值。缺少凭据时抛出 RuntimeError；成功后刷新 SDK 事件并记录数量，不打印 prompt 内容或任何密钥。
    """

    load_dotenv(PROJECT_ROOT / ".env")
    missing = [
        name
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL are required")

    config = load_opera_runtime_config().langfuse
    parser = argparse.ArgumentParser(description="Sync OPERA system prompts to Langfuse")
    parser.add_argument(
        "--label",
        default=config.prompt_label,
        help="Langfuse prompt label to assign; defaults to the YAML prompt_label",
    )
    args = parser.parse_args()

    from langfuse import get_client

    client = get_client()
    synced = sync_prompt_definitions(client, config, args.label)
    client.flush()
    logger.info("event=opera_prompt_sync_completed prompt_count=%s label=%s", synced, args.label)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
