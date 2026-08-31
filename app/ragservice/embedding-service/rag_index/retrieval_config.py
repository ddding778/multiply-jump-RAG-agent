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


@dataclass(frozen=True)
class OperaRetrievalConfig:
    """保存 OPERA HotpotQA 检索的独立 collection 和范围配置。

    参数 collection_prefix 为 OPERA collection 名称前缀，default_scope 为缺省检索范围，top_k 为默认核心命中数，
    case_bm25_cache_size 为按题构建的 BM25 模型 LRU 缓存上限；返回对象不包含密钥或运行时连接。
    """

    collection_prefix: str
    default_scope: str
    top_k: int
    case_bm25_cache_size: int


@dataclass(frozen=True)
class OperaAgentConfig:
    """保存单个 OPERA Agent 的模型输出和超时限制。

    参数 max_output_tokens 为单次调用的软输出上限，timeout_seconds 为 provider 请求超时配置；
    返回对象只描述 YAML 值，不创建模型客户端。
    """

    max_output_tokens: int
    timeout_seconds: int


@dataclass(frozen=True)
class OperaLangfuseConfig:
    """保存 OPERA 的 Langfuse prompt 与隐私配置。

    参数 enabled 控制是否建立 Langfuse 客户端，prompt_label 为读取的版本标签，
    prompt_cache_ttl_seconds 为 SDK 缓存秒数，capture_input_output 控制业务原文采集，
    prompts 保存各 Agent 的稳定 prompt 名称；返回对象不包含密钥。
    """

    enabled: bool
    prompt_label: str
    prompt_cache_ttl_seconds: int
    capture_input_output: bool
    prompts: dict[str, str]


@dataclass(frozen=True)
class OperaRuntimeConfig:
    """保存 OPERA 执行器的完整非敏感运行配置。

    参数包含检索、模型、Agent、Schema 与 Langfuse 配置；返回对象仅用于构造服务，
    不读取环境变量中的密钥或索引版本。
    """

    retrieval: OperaRetrievalConfig
    max_steps: int
    max_rewrites: int
    model_name: str
    agents: dict[str, OperaAgentConfig]
    max_repair_attempts: int
    langfuse: OperaLangfuseConfig


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


def load_opera_retrieval_config(path: Path = DEFAULT_RETRIEVAL_CONFIG_PATH) -> OperaRetrievalConfig:
    """读取并校验 OPERA HotpotQA 的独立检索配置。

    参数 path 为共用 YAML 路径；返回 OperaRetrievalConfig。字段缺失或非法时抛出 RetrievalConfigError，
    不使用隐藏默认值，避免 OPERA 与 RAG V2 意外共用 collection。
    """

    loaded = _load_config_mapping(path)
    opera = loaded.get("opera")
    if not isinstance(opera, dict):
        raise RetrievalConfigError("RAG retrieval config requires opera mapping")
    retrieval = opera.get("retrieval")
    if not isinstance(retrieval, dict):
        raise RetrievalConfigError("opera.retrieval must be a mapping")

    collection_prefix = retrieval.get("collection_prefix")
    default_scope = retrieval.get("default_scope")
    top_k = retrieval.get("top_k")
    case_bm25_cache_size = retrieval.get("case_bm25_cache_size")
    if not isinstance(collection_prefix, str) or not _is_qdrant_name(collection_prefix):
        raise RetrievalConfigError("opera.retrieval.collection_prefix contains unsupported characters")
    if default_scope not in {"case", "all"}:
        raise RetrievalConfigError("opera.retrieval.default_scope must be case or all")
    if not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalConfigError("opera.retrieval.top_k must be a positive integer")
    if not isinstance(case_bm25_cache_size, int) or case_bm25_cache_size <= 0:
        raise RetrievalConfigError("opera.retrieval.case_bm25_cache_size must be a positive integer")
    return OperaRetrievalConfig(
        collection_prefix=collection_prefix,
        default_scope=default_scope,
        top_k=top_k,
        case_bm25_cache_size=case_bm25_cache_size,
    )


def load_opera_runtime_config(path: Path = DEFAULT_RETRIEVAL_CONFIG_PATH) -> OperaRuntimeConfig:
    """读取并校验 OPERA Agent、Schema 与 Langfuse 的完整运行配置。

    参数 path 为共用 YAML 路径；返回 OperaRuntimeConfig。字段缺失、类型错误或超出范围时抛出
    RetrievalConfigError，避免服务以隐藏的模型、prompt 或观测默认值运行。
    """

    loaded = _load_config_mapping(path)
    opera = loaded.get("opera")
    if not isinstance(opera, dict):
        raise RetrievalConfigError("RAG retrieval config requires opera mapping")

    max_steps = opera.get("max_steps")
    max_rewrites = opera.get("max_rewrites")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise RetrievalConfigError("opera.max_steps must be a positive integer")
    if not isinstance(max_rewrites, int) or isinstance(max_rewrites, bool) or max_rewrites < 0:
        raise RetrievalConfigError("opera.max_rewrites must be a non-negative integer")

    model = opera.get("model")
    if not isinstance(model, dict):
        raise RetrievalConfigError("opera.model must be a mapping")
    model_name = model.get("name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise RetrievalConfigError("opera.model.name must be a non-empty string")

    agents = _parse_opera_agents(opera.get("agents"))
    schema_validation = opera.get("schema_validation")
    if not isinstance(schema_validation, dict):
        raise RetrievalConfigError("opera.schema_validation must be a mapping")
    max_repair_attempts = schema_validation.get("max_repair_attempts")
    if (
        not isinstance(max_repair_attempts, int)
        or isinstance(max_repair_attempts, bool)
        or max_repair_attempts < 0
    ):
        raise RetrievalConfigError("opera.schema_validation.max_repair_attempts must be a non-negative integer")

    return OperaRuntimeConfig(
        retrieval=load_opera_retrieval_config(path),
        max_steps=max_steps,
        max_rewrites=max_rewrites,
        model_name=model_name.strip(),
        agents=agents,
        max_repair_attempts=max_repair_attempts,
        langfuse=_parse_opera_langfuse_config(opera.get("langfuse")),
    )


def _parse_opera_agents(value: object) -> dict[str, OperaAgentConfig]:
    """校验固定三类 OPERA Agent 的运行限制。

    参数 value 为 YAML 的 opera.agents 节；返回按 Agent 名称索引的配置字典。
    缺失、额外 Agent 或非法限制时抛出 RetrievalConfigError。
    """

    expected_names = {"planner", "analysis_answer", "rewriter"}
    if not isinstance(value, dict) or set(value) != expected_names:
        raise RetrievalConfigError("opera.agents must contain planner, analysis_answer and rewriter")

    parsed: dict[str, OperaAgentConfig] = {}
    for name in expected_names:
        raw = value[name]
        if not isinstance(raw, dict):
            raise RetrievalConfigError(f"opera.agents.{name} must be a mapping")
        max_output_tokens = raw.get("max_output_tokens")
        timeout_seconds = raw.get("timeout_seconds")
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            raise RetrievalConfigError(f"opera.agents.{name}.max_output_tokens must be a positive integer")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise RetrievalConfigError(f"opera.agents.{name}.timeout_seconds must be a positive integer")
        parsed[name] = OperaAgentConfig(max_output_tokens=max_output_tokens, timeout_seconds=timeout_seconds)
    return parsed


def _parse_opera_langfuse_config(value: object) -> OperaLangfuseConfig:
    """校验 OPERA Langfuse 开关、缓存、隐私策略与 prompt 名称。

    参数 value 为 YAML 的 opera.langfuse 节；返回不含凭据的 OperaLangfuseConfig。
    类型错误、缺少固定 Agent prompt 或缓存非法时抛出 RetrievalConfigError。
    """

    expected_prompt_keys = {"planner", "analysis_answer", "rewriter"}
    if not isinstance(value, dict):
        raise RetrievalConfigError("opera.langfuse must be a mapping")
    enabled = value.get("enabled")
    prompt_label = value.get("prompt_label")
    cache_ttl_seconds = value.get("prompt_cache_ttl_seconds")
    capture_input_output = value.get("capture_input_output")
    prompts = value.get("prompts")
    if not isinstance(enabled, bool):
        raise RetrievalConfigError("opera.langfuse.enabled must be a boolean")
    if not isinstance(prompt_label, str) or not prompt_label.strip():
        raise RetrievalConfigError("opera.langfuse.prompt_label must be a non-empty string")
    if (
        not isinstance(cache_ttl_seconds, int)
        or isinstance(cache_ttl_seconds, bool)
        or cache_ttl_seconds < 0
    ):
        raise RetrievalConfigError("opera.langfuse.prompt_cache_ttl_seconds must be a non-negative integer")
    if not isinstance(capture_input_output, bool):
        raise RetrievalConfigError("opera.langfuse.capture_input_output must be a boolean")
    if not isinstance(prompts, dict) or set(prompts) != expected_prompt_keys:
        raise RetrievalConfigError("opera.langfuse.prompts must contain planner, analysis_answer and rewriter")
    if any(not isinstance(name, str) or not name.strip() for name in prompts.values()):
        raise RetrievalConfigError("opera.langfuse prompt names must be non-empty strings")
    return OperaLangfuseConfig(
        enabled=enabled,
        prompt_label=prompt_label.strip(),
        prompt_cache_ttl_seconds=cache_ttl_seconds,
        capture_input_output=capture_input_output,
        prompts={key: value.strip() for key, value in prompts.items()},
    )


def _load_config_mapping(path: Path) -> dict[str, object]:
    """读取 YAML 配置根对象并统一转换为字典。

    参数 path 为 YAML 文件路径；返回已反序列化的字典。文件、依赖或 YAML 格式异常时抛出 RetrievalConfigError。
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
    return loaded


def _is_qdrant_name(value: str) -> bool:
    """判断 collection 前缀是否只含 Qdrant collection 名称允许的安全字符。

    参数 value 为待校验名称；返回仅由 ASCII 字母、数字、下划线和连字符组成时为真。
    """

    return bool(value) and all(character.isascii() and (character.isalnum() or character in "_-") for character in value)
