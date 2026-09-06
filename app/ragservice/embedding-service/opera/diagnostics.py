"""演示请求的低成本依赖检查与安全故障分类，不调用模型。"""

import os
from pathlib import Path

from qdrant_client import QdrantClient
from rag_index.bm25_index import Bm25IndexError
from rag_index.retrieval_config import load_opera_runtime_config, collection_name_for_version
from .events import emit


class DemoDependencyError(RuntimeError):
    """携带可直接展示的依赖错误，不包含底层异常原文。"""

    def __init__(self, category: str, message: str) -> None:
        """参数 category 为稳定分类，message 为安全中文文案；初始化异常，无返回值。"""
        super().__init__(message)
        self.category = category
        self.message = message


def check_demo_dependencies(project_root: Path) -> None:
    """预检配置、artifact 与 Qdrant；参数为仓库根目录，无返回值，失败抛出分类错误。"""
    emit("initialization", stage="configuration", message="正在检查模型配置与 HotpotQA 索引文件…")
    index_version = os.getenv("OPERA_INDEX_VERSION", "").strip()
    if not index_version or not os.getenv("DEEPSEEK_API_KEY") or not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")):
        raise DemoDependencyError("configuration", "后端配置不完整：请检查 OPERA_INDEX_VERSION、DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY。")
    try:
        config = load_opera_runtime_config()
        collection_name = collection_name_for_version(config.retrieval.collection_prefix, index_version)
    except ValueError as error:
        raise DemoDependencyError("configuration", "OPERA 配置无效，请检查索引版本与 rag-retrieval.yaml。") from error
    artifact = project_root / "out" / "opera-index" / config.retrieval.collection_prefix / index_version / "bm25_index.json"
    if not artifact.is_file():
        raise DemoDependencyError("index", "HotpotQA BM25 索引文件不存在，请检查 OPERA_INDEX_VERSION 与对应导入产物。")
    emit("initialization", stage="qdrant", message="正在检查 Qdrant 连接与 HotpotQA collection…")
    try:
        port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    except ValueError as error:
        raise DemoDependencyError("configuration", "QDRANT_GRPC_PORT 必须是有效端口。") from error
    client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), grpc_port=port,
                          prefer_grpc=True, timeout=3, check_compatibility=False)
    try:
        # 先区别连接失败，再检查 collection，避免将索引缺失报成服务没启动。
        client.get_collections()
    except Exception as error:
        raise DemoDependencyError("qdrant", "无法连接 Qdrant。请启动 Docker Desktop 和 Qdrant 容器，并检查 6334 端口。") from error
    else:
        try:
            client.get_collection(collection_name)
        except Exception as error:
            raise DemoDependencyError("index", "Qdrant 已连接，但当前 HotpotQA collection 不可用。请检查 OPERA_INDEX_VERSION 与索引导入状态。") from error
    finally:
        client.close()


def describe_failure(error: Exception) -> tuple[int, str, str]:
    """分类运行失败；参数为异常，返回 HTTP 语义码、稳定分类和安全中文文案。"""
    if isinstance(error, DemoDependencyError):
        return 503, error.category, error.message
    if isinstance(error, Bm25IndexError):
        return 503, "index", "BM25 索引读取或校验失败，请检查索引文件及 OPERA_INDEX_VERSION。"
    if isinstance(error, ValueError):
        return 400, "agent_output", "本次 Agent 输出或证据校验失败，请查看调用结果后手动重试。"
    if type(error).__name__ == "HotpotRetrievalError":
        cause = error.__cause__
        if cause is not None and type(cause).__module__.startswith("openai"):
            return 503, "embedding", "DashScope embedding 请求失败或超时，请检查网络、服务及密钥配置。"
        return 503, "retrieval", "HotpotQA 检索失败，请检查 Qdrant 是否仍在运行、索引是否匹配。"
    if type(error).__module__.startswith("openai"):
        return 503, "model", "DeepSeek 模型请求失败或超时，请检查网络、服务及密钥配置。"
    return 503, "service", "RAG 服务暂不可用，请检查后端日志与依赖。"
