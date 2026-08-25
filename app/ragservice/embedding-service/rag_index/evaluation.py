"""比较 Dense 基线与完整混合 RAG 链路的离线评测入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CUTOFFS = (1, 3, 5, 10)


@dataclass(frozen=True)
class EvaluationCase:
    """表示一条仅含评测所需信息的已校验检索样本。

    参数 case_id 为样本标识，query 为仅在内存中送入检索的提问，relevant_chunk_ids 为正确 chunk 集合，
    stress_type 为可选分组标签；返回对象不保留评测文件中的正文、备注或其他不可信字段。
    """

    case_id: str
    query: str
    relevant_chunk_ids: set[str]
    stress_type: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 RAG 双基线评测命令行参数。

    参数 argv 为可选参数列表；返回包含评测集路径、输出目录和最大 Top-K 的命名空间。
    """

    parser = argparse.ArgumentParser(description="Evaluate Dense baseline versus RRF+MMR RAG retrieval.")
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "out" / "rag-eval")
    parser.add_argument("--top-k-max", type=int, default=10)
    return parser.parse_args(argv)


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """读取 JSONL 评测集并提取单目标或多目标相关 chunk 集合。

    参数 path 为 JSONL 文件；返回已校验的 EvaluationCase 列表。格式错误、空问题或无相关 chunk 时抛出 ValueError。
    文件中的 note、comparison 等字段均不作为指令执行。
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("evaluation set cannot be read") from error
    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"evaluation set line {line_number} is not valid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"evaluation set line {line_number} must be an object")
        case_id = row.get("case_id")
        query = row.get("query")
        if not isinstance(case_id, str) or not case_id or case_id in seen_case_ids:
            raise ValueError(f"evaluation set line {line_number} has invalid case_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"evaluation set line {line_number} has empty query")
        relevant_chunk_ids = _relevant_chunk_ids(row)
        if not relevant_chunk_ids:
            raise ValueError(f"evaluation set line {line_number} has no relevant chunk")
        seen_case_ids.add(case_id)
        stress_type = row.get("stress_type")
        cases.append(
            EvaluationCase(
                case_id=case_id,
                query=query.strip(),
                relevant_chunk_ids=relevant_chunk_ids,
                stress_type=stress_type if isinstance(stress_type, str) and stress_type else "unspecified",
            )
        )
    if not cases:
        raise ValueError("evaluation set has no cases")
    return cases


def evaluate_cases(service: Any, cases: list[EvaluationCase], top_k_max: int) -> dict[str, object]:
    """运行每题一次 embedding 的 Dense 与混合链路对比评测。

    参数 service 为已启动且校验过索引的检索服务，cases 为评测样本，top_k_max 为最大排名截断；
    返回不含 query/正文的指标、分组汇总和逐 case 排名。某题依赖失败时抛出异常，避免混入部分成功结果。
    """

    if top_k_max <= 0:
        raise ValueError("top_k_max must be positive")
    mode_ranks: dict[str, list[dict[str, object]]] = {"dense": [], "hybrid": []}
    for case in cases:
        ranked_results = service.evaluate_core_chunks(case.query, top_k_max)
        for mode in mode_ranks:
            chunks = ranked_results.get(mode)
            if not isinstance(chunks, list):
                raise ValueError(f"retrieval service did not return {mode} results")
            candidate_ids = [chunk.chunk_id for chunk in chunks]
            first_rank = next(
                (rank for rank, chunk_id in enumerate(candidate_ids, start=1) if chunk_id in case.relevant_chunk_ids),
                None,
            )
            mode_ranks[mode].append(
                {
                    "case_id": case.case_id,
                    "stress_type": case.stress_type,
                    "rank": first_rank,
                    "returned_count": len(candidate_ids),
                }
            )
    return {
        "case_count": len(cases),
        "dense": _summarize_mode(mode_ranks["dense"], top_k_max),
        "hybrid": _summarize_mode(mode_ranks["hybrid"], top_k_max),
    }


def write_evaluation_result(
    output_dir: Path,
    eval_set_path: Path,
    service: Any,
    top_k_max: int,
    results: dict[str, object],
) -> Path:
    """将双基线评测结果原子写入版本化 out 目录。

    参数 output_dir 为 `out/rag-eval` 根目录，eval_set_path 为输入文件，service 为检索服务，
    top_k_max 为评测截断，results 为已汇总指标；返回结果 JSON 的绝对路径。
    输出不包含 query、相关正文、向量或任何密钥。
    """

    index_version = service._settings.index_version
    output_path = output_dir / index_version / f"{eval_set_path.stem}-dense-vs-hybrid.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256(eval_set_path.read_bytes()).hexdigest()
    artifact = {
        "record_type": "rag_retrieval_evaluation",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "eval_set_name": eval_set_path.name,
        "eval_set_sha256": source_digest,
        "index_version": index_version,
        "collection_name": service._settings.collection_name,
        "embedding_model": service._settings.embedding_model,
        "top_k_max": top_k_max,
        "retrieval_config": {
            "candidate_top_k": service._settings.candidate_top_k,
            "rrf_rank_constant": service._settings.rrf_rank_constant,
            "rrf_dense_weight": service._settings.rrf_dense_weight,
            "rrf_bm25_weight": service._settings.rrf_bm25_weight,
            "mmr_lambda": service._settings.mmr_lambda,
        },
        "modes": ["dense", "hybrid"],
        "results": results,
    }
    temporary_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output_path.parent,
        prefix="evaluation-",
        suffix=".tmp",
        delete=False,
    )
    with temporary_file as stream:
        json.dump(artifact, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    os.replace(temporary_file.name, output_path)
    return output_path.resolve()


def configure_logging() -> logging.Logger:
    """创建只记录评测元数据而不记录问题或文档正文的日志器。

    无参数；返回标准输出日志器，日志包含总数、版本、输出路径和耗时等安全字段。
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("jieba").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logging.getLogger("rag_evaluation")


def main(argv: list[str] | None = None) -> int:
    """执行 Dense 与完整混合链路的真实检索评测。

    参数 argv 为可选参数列表；成功时返回 0 并写入 out 结果，失败时返回 1 且日志不输出评测问题或正文。
    """

    args = parse_args(argv)
    logger = configure_logging()
    try:
        if args.top_k_max <= 0:
            raise ValueError("top_k_max must be positive")
        from dotenv import load_dotenv
        import main as service_entry

        load_dotenv(PROJECT_ROOT / ".env")
        cases = load_evaluation_cases(args.eval_set)
        service = service_entry._create_service()
        started_at = datetime.now(UTC)
        results = evaluate_cases(service, cases, args.top_k_max)
        output_path = write_evaluation_result(args.output_dir, args.eval_set, service, args.top_k_max, results)
        duration_ms = int((datetime.now(UTC) - started_at).total_seconds() * 1000)
        logger.info(
            "event=rag_evaluation_completed index_version=%s collection=%s case_count=%d output=%s duration_ms=%d",
            service._settings.index_version,
            service._settings.collection_name,
            len(cases),
            output_path,
            duration_ms,
        )
        return 0
    except Exception as error:
        logger.error("event=rag_evaluation_failed error_type=%s", type(error).__name__)
        return 1


def _relevant_chunk_ids(row: dict[str, object]) -> set[str]:
    """从两种已知评测 schema 中提取正确 chunk ID 集合。

    参数 row 为 JSONL 中一条已解析记录；返回 `target_chunk_id` 或 `relevant_chunks` 内的有效业务 ID 集合。
    无法识别时返回空集合，由调用方报告格式错误。
    """

    target_chunk_id = row.get("target_chunk_id")
    if isinstance(target_chunk_id, str) and target_chunk_id:
        return {target_chunk_id}
    relevant_chunks = row.get("relevant_chunks")
    if not isinstance(relevant_chunks, list):
        return set()
    return {
        item["chunk_id"]
        for item in relevant_chunks
        if isinstance(item, dict) and isinstance(item.get("chunk_id"), str) and item["chunk_id"]
    }


def _summarize_mode(rows: list[dict[str, object]], top_k_max: int) -> dict[str, object]:
    """按整体和 stress_type 计算单目标或多目标首个命中的排名指标。

    参数 rows 为不含原文的逐 case 排名，top_k_max 为评测最大截断；返回 HitRate、MRR 和分组汇总。
    """

    if not rows:
        raise ValueError("evaluation has no ranking rows")
    cutoffs = [cutoff for cutoff in DEFAULT_CUTOFFS if cutoff <= top_k_max]
    summary = _rank_metrics(rows, cutoffs, top_k_max)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stress_type"])].append(row)
    summary["by_stress_type"] = {
        stress_type: _rank_metrics(group_rows, cutoffs, top_k_max)
        for stress_type, group_rows in sorted(grouped.items())
    }
    summary["cases"] = rows
    return summary


def _rank_metrics(rows: list[dict[str, object]], cutoffs: list[int], top_k_max: int) -> dict[str, float | int]:
    """根据每题的首个相关排名计算 HitRate 和截断 MRR。

    参数 rows 为逐题排名，cutoffs 为需统计的 K，top_k_max 为 MRR 截断；返回纯数值指标字典。
    """

    ranks = [row["rank"] if isinstance(row.get("rank"), int) else None for row in rows]
    metrics: dict[str, float | int] = {"case_count": len(rows)}
    for cutoff in cutoffs:
        metrics[f"HitRate@{cutoff}"] = round(sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks), 4)
    metrics[f"MRR@{top_k_max}"] = round(
        sum(1.0 / rank for rank in ranks if rank is not None and rank <= top_k_max) / len(ranks),
        4,
    )
    return metrics


if __name__ == "__main__":
    sys.exit(main())
