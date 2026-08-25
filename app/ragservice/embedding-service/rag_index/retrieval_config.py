"""加载 RAG V2 离线索引和在线检索共用的 YAML 配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_RETRIEVAL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag-retrieval.yaml"


class RetrievalConfigError(ValueError):
    """表示 RAG 检索 YAML 配置缺失、格式错误或不满足约束。"""


@dataclass(frozen=True)
class RetrievalAlgorithmConfig:
    """保存 BM25、RRF、MMR 和 collection 命名共用的非敏感配置。

    参数 collection_prefix 为每批 Qdrant collection 的名称前缀，bm25_candidate_top_k 为两路候选数，
    bm25_tokenizer_version 为索引分词合同，rrf_rank_constant 与两路权重为融合参数，mmr_lambda 为相关性权重；
    返回对象只描述配置值，不负责读取索引或执行检索。
    """

    collection_prefix: str
    bm25_candidate_top_k: int
    bm25_tokenizer_version: str
    rrf_rank_constant: int
    rrf_dense_weight: float
    rrf_bm25_weight: float
    mmr_lambda: float


def collection_name_for_version(collection_prefix: str, index_version: str) -> str:
    """根据 collection 前缀和完整索引版本生成唯一 Qdrant collection 名称。

    参数 collection_prefix 为 YAML 中声明的前缀，index_version 为 SHA-256 十六进制版本；
    返回由二者组成的稳定 collection 名称。任一值非法时抛出 RetrievalConfigError。
    """

    if not _is_qdrant_name(collection_prefix):
        raise RetrievalConfigError("collection_prefix contains unsupported characters")
    if len(index_version) != 64 or any(character not in "0123456789abcdef" for character in index_version):
        raise RetrievalConfigError("index_version must be a lowercase SHA-256 string")
    return f"{collection_prefix}_{index_version}"


def load_retrieval_algorithm_config(path: Path = DEFAULT_RETRIEVAL_CONFIG_PATH) -> RetrievalAlgorithmConfig:
    """读取并严格校验 RAG 检索 YAML 文件。

    参数 path 为 YAML 文件路径；返回 RetrievalAlgorithmConfig。文件缺失、依赖缺失或字段非法时抛出
    RetrievalConfigError，调用方不得以代码内隐藏默认值替代。
    """

    try:
        import yaml
    except ImportError as error:
        raise RetrievalConfigError("PyYAML is required to load RAG retrieval config") from error

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RetrievalConfigError("RAG retrieval config cannot be read") from error
    except yaml.YAMLError as error:
        raise RetrievalConfigError("RAG retrieval config is invalid YAML") from error

    if not isinstance(loaded, dict):
        raise RetrievalConfigError("RAG retrieval config must be a mapping")
    bm25 = loaded.get("bm25")
    rrf = loaded.get("rrf")
    mmr = loaded.get("mmr")
    if not isinstance(bm25, dict) or not isinstance(rrf, dict) or not isinstance(mmr, dict):
        raise RetrievalConfigError("RAG retrieval config requires bm25, rrf and mmr mappings")

    collection_prefix = loaded.get("collection_prefix")
    candidate_top_k = bm25.get("candidate_top_k")
    tokenizer_version = bm25.get("tokenizer_version")
    rank_constant = rrf.get("rank_constant")
    dense_weight = rrf.get("dense_weight")
    bm25_weight = rrf.get("bm25_weight")
    mmr_lambda = mmr.get("lambda")
    if not isinstance(collection_prefix, str) or not _is_qdrant_name(collection_prefix):
        raise RetrievalConfigError("collection_prefix contains unsupported characters")
    if not isinstance(candidate_top_k, int) or candidate_top_k <= 0:
        raise RetrievalConfigError("bm25.candidate_top_k must be a positive integer")
    if not isinstance(tokenizer_version, str) or not tokenizer_version.strip():
        raise RetrievalConfigError("bm25.tokenizer_version must be a non-empty string")
    if not isinstance(rank_constant, int) or rank_constant <= 0:
        raise RetrievalConfigError("rrf.rank_constant must be a positive integer")
    if not isinstance(dense_weight, (int, float)) or isinstance(dense_weight, bool) or float(dense_weight) <= 0:
        raise RetrievalConfigError("rrf.dense_weight must be positive")
    if not isinstance(bm25_weight, (int, float)) or isinstance(bm25_weight, bool) or float(bm25_weight) <= 0:
        raise RetrievalConfigError("rrf.bm25_weight must be positive")
    if not isinstance(mmr_lambda, (int, float)) or isinstance(mmr_lambda, bool) or not 0 < float(mmr_lambda) <= 1:
        raise RetrievalConfigError("mmr.lambda must be within (0, 1]")
    return RetrievalAlgorithmConfig(
        collection_prefix=collection_prefix,
        bm25_candidate_top_k=candidate_top_k,
        bm25_tokenizer_version=tokenizer_version.strip(),
        rrf_rank_constant=rank_constant,
        rrf_dense_weight=float(dense_weight),
        rrf_bm25_weight=float(bm25_weight),
        mmr_lambda=float(mmr_lambda),
    )


def _is_qdrant_name(value: str) -> bool:
    """判断 collection 前缀是否只含 Qdrant collection 名称允许的安全字符。

    参数 value 为待校验名称；返回仅由 ASCII 字母、数字、下划线和连字符组成时为真。
    """

    return bool(value) and all(character.isascii() and (character.isalnum() or character in "_-") for character in value)
