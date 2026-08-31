"""OPERA ExecutionState 与串行多跳执行器。"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from uuid import uuid4

from .agents.analysis_answer import analyze
from .agents.planner import plan
from .agents.rewriter import rewrite
from .observability import LangfuseSettings, OperaObservability
from .schemas import AnalysisStatus, EvidenceRef, OperaAskResponse, PlanResult, PlanStep, RetrievalScope, StepResult


@dataclass
class ExecutionState:
    """保存单请求短生命周期的已验证执行状态。"""

    run_id: str
    question: str
    retrieval_scope: RetrievalScope
    case_id: str | None
    plan: PlanResult | None = None
    step_results: dict[str, StepResult] = field(default_factory=dict)
    rewrite_count_by_step: dict[str, int] = field(default_factory=dict)


class OperaExecutor:
    """协调 Planner、Retriever、Analysis-Answer 与 Rewrite 的串行闭环。"""

    def __init__(
        self,
        planner_client: object,
        analysis_client: object,
        rewrite_client: object,
        retriever: object,
        prompts: dict[str, object],
        max_steps: int,
        max_rewrites: int,
        top_k: int,
        observability: OperaObservability | None = None,
    ) -> None:
        """保存 Agent 客户端、检索器、prompt、YAML 限制与可选观测器。

        参数三类 client 分别执行规划、分析和改写，retriever 执行混合检索，prompts 为已解析 prompt，
        max_steps、max_rewrites、top_k 为 YAML 上限，observability 为可选 Langfuse 观测器；无返回值。
        """

        self._planner_client = planner_client
        self._analysis_client = analysis_client
        self._rewrite_client = rewrite_client
        self._retriever = retriever
        self._prompts = prompts
        self._max_steps = max_steps
        self._max_rewrites = max_rewrites
        self._top_k = top_k
        self._observability = observability or OperaObservability.create(
            LangfuseSettings(False, "production", 0, False),
            logging.getLogger("rag_service"),
        )

    def execute(
        self,
        question: str,
        scope: RetrievalScope,
        case_id: str | None,
        top_k: int | None = None,
    ) -> OperaAskResponse:
        """执行一次多跳请求，并仅返回 final 步的已验证答案。

        参数 question 为用户问题，scope 与 case_id 限定检索范围，top_k 可覆盖默认候选数；
        返回完成或证据不足的 OperaAskResponse。模型、检索和 Agent 输出均被嵌套记录到同一 trace。
        """

        state = ExecutionState(str(uuid4()), question, scope, case_id)
        with self._observability.request(
            run_id=state.run_id,
            question=question,
            retrieval_scope=scope.value,
            has_case_id=case_id is not None,
            max_steps=self._max_steps,
            max_rewrites=self._max_rewrites,
        ) as request_observation:
            try:
                response = self._execute_state(state, top_k)
            except Exception as error:
                request_observation.update(metadata={"result": "failed", "error_type": type(error).__name__})
                raise
            request_observation.update(
                output_data=response.model_dump(),
                metadata={
                    "result": response.status,
                    "completed_step_count": response.completed_step_count,
                },
            )
            return response

    def flush_observability(self) -> None:
        """在服务关闭时刷出本执行器已排队的 Langfuse 事件。

        无参数；无返回值。观测器自身会吞掉刷新异常，避免影响服务正常关闭。
        """

        self._observability.flush()

    def _execute_state(self, state: ExecutionState, top_k: int | None) -> OperaAskResponse:
        """在已创建的请求 trace 内推进 ExecutionState。

        参数 state 为当前请求独占运行态，top_k 为可选候选数覆盖；返回最终 HTTP 响应。
        任何越界计划、伪造证据或模型错误均向上抛出，由 HTTP 层统一处理。
        """

        with self._observability.agent("planner") as planner_observation:
            state.plan = plan(self._planner_client, self._prompts["planner"], state.question)
            planner_observation.update(metadata={"planned_step_count": len(state.plan.steps)})
        if len(state.plan.steps) > self._max_steps:
            raise ValueError("plan exceeds configured max_steps")

        for step in state.plan.steps:
            response = self._execute_step(state, step, top_k)
            if response is not None:
                return response

        final = state.step_results[state.plan.steps[-1].step_id]
        return OperaAskResponse(
            run_id=state.run_id,
            status="completed",
            answer=final.answer,
            final_evidence=final.evidence,
            completed_step_count=len(state.step_results),
        )

    def _execute_step(self, state: ExecutionState, step: PlanStep, top_k: int | None) -> OperaAskResponse | None:
        """检索、分析并按需改写一个已排序子目标。

        参数 state 为当前运行态，step 为通过 Plan Schema 校验的步骤，top_k 为可选候选数覆盖；
        返回证据不足时的响应，成功完成步骤时返回 None。
        """

        query = step.subgoal
        rewrites = 0
        while True:
            result = self._retrieve(state, step.step_id, query, top_k)
            evidence = [
                {
                    "chunk_id": item.chunk_id,
                    "title": item.title,
                    "sentences": list(item.sentences),
                }
                for item in result.hybrid
            ]
            dependencies = {dependency: state.step_results[dependency].answer or "" for dependency in step.depends_on}
            with self._observability.agent(
                "analysis-answer",
                {"step_id": step.step_id, "dependency_count": len(dependencies), "rewrite_count": rewrites},
            ) as analysis_observation:
                analysis = analyze(
                    self._analysis_client,
                    self._prompts["analysis"],
                    step.subgoal,
                    dependencies,
                    evidence,
                )
                analysis_observation.update(metadata={"analysis_status": analysis.status.value})

            if analysis.status == AnalysisStatus.SUPPORTED:
                self._validate_evidence(analysis.evidence, evidence)
                state.step_results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    status="completed",
                    answer=analysis.answer,
                    evidence=analysis.evidence,
                    retrieval_query=query,
                    rewrite_count=rewrites,
                )
                return None

            if rewrites >= self._max_rewrites:
                state.step_results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    status="insufficient",
                    answer=None,
                    evidence=[],
                    retrieval_query=query,
                    rewrite_count=rewrites,
                )
                return OperaAskResponse(
                    run_id=state.run_id,
                    status="insufficient",
                    answer=None,
                    final_evidence=[],
                    completed_step_count=len(state.step_results),
                )

            with self._observability.agent(
                "rewrite",
                {"step_id": step.step_id, "rewrite_count": rewrites + 1},
            ):
                rewritten = rewrite(
                    self._rewrite_client,
                    self._prompts["rewrite"],
                    step.subgoal,
                    analysis.failure_reason,
                )
            query = rewritten.rewritten_query
            rewrites += 1
            state.rewrite_count_by_step[step.step_id] = rewrites

    def _retrieve(self, state: ExecutionState, step_id: str, query: str, top_k: int | None) -> object:
        """执行一次 Hybrid Retriever 并记录不含正文的检索观测。

        参数 state 提供 scope 和 case 限制，step_id 标识执行步骤，query 为当前查询，top_k 为候选数覆盖；
        返回 Retriever 原始结果。默认不上传 query 与 paragraph 内容，仅记录数量和范围标签。
        """

        selected_top_k = top_k or self._top_k
        with self._observability.retriever(
            {
                "step_id": step_id,
                "retrieval_scope": state.retrieval_scope.value,
                "top_k": selected_top_k,
            },
            {"query": query},
        ) as retrieval_observation:
            result = self._retriever.search(query, state.retrieval_scope.value, state.case_id, selected_top_k)
            retrieval_observation.update(metadata={"hybrid_candidate_count": len(result.hybrid)})
            return result

    def _validate_evidence(self, refs: list[EvidenceRef], evidence: list[dict[str, object]]) -> None:
        """确认 Agent 指向的 chunk 与句子索引属于本次 Retriever 候选。

        参数 refs 为 Analysis-Answer 返回的证据引用，evidence 为本轮检索候选；无返回值。
        任一 chunk 或 sentence_index 越界时抛出 ValueError，拒绝把伪造证据写入 ExecutionState。
        """

        allowed = {str(item["chunk_id"]): len(item["sentences"]) for item in evidence}
        if any(ref.chunk_id not in allowed or ref.sentence_index >= allowed[ref.chunk_id] for ref in refs):
            raise ValueError("analysis evidence is outside retrieved paragraphs")
