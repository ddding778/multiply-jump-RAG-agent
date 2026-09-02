"""OPERA schema、修复重试与多跳执行器的无外部依赖测试。"""
from __future__ import annotations
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opera.executor import OperaExecutor
from opera.debug_trace import OperaTraceWriter
from opera.llm import ResponseOutputTextError, ResponsesClient
from opera.schemas import AnalysisResult, OperaAskRequest, PlanResult, RetrievalScope
import main

class FakeResponses:
    """按调用顺序返回预设 Responses 文本。"""
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        """保存模拟模型输出列表。"""
        self.outputs = outputs; self.calls = 0; self.requests: list[dict[str, object]] = []; self.responses = self
    def create(self, **request: object) -> SimpleNamespace:
        """返回下一条 JSON 输出。"""
        self.requests.append(request)
        value = self.outputs[self.calls]; self.calls += 1
        return SimpleNamespace(output_text=json.dumps(value))

class FakeRetriever:
    """返回固定 paragraph 的检索器替身。"""
    def __init__(self) -> None:
        """初始化记录检索 query 的列表。"""
        self.queries: list[str] = []

    def search(self, query: str, *_: object) -> SimpleNamespace:
        """返回一个合法的 Hybrid 命中。"""
        self.queries.append(query)
        item = SimpleNamespace(chunk_id="case-a:0", title="A", sentences=("fact",))
        trace = SimpleNamespace(
            scope="case",
            case_id="case-a",
            dense_candidate_count=3,
            bm25_candidate_count=3,
            fused_candidate_count=1,
            bm25_case_cache_hit=False,
            duration_ms=12,
        )
        return SimpleNamespace(hybrid=[item], trace=trace)


class EmptyOutputResponses:
    """返回缺少文本内容的 provider 响应替身。"""

    def __init__(self) -> None:
        """暴露与 OpenAI 客户端兼容的 responses 属性。"""

        self.responses = self

    def create(self, **_: object) -> SimpleNamespace:
        """返回带失败状态和空 output_text 的响应。"""

        return SimpleNamespace(id="response-test", status="failed", output_text="", output=[], error=None, incomplete_details=None)

class OperaExecutionTest(unittest.TestCase):
    """验证 Agent 结构化边界与 ExecutionState 串行规则。"""
    def test_request_scope_contract(self) -> None:
        """验证 case/all 的 case_id 条件约束。"""
        with self.assertRaises(ValueError): OperaAskRequest(question="q", retrieval_scope="case")
        with self.assertRaises(ValueError): OperaAskRequest(question="q", retrieval_scope="all", case_id="a")

    def test_responses_client_repairs_invalid_json_schema_once(self) -> None:
        """验证本地校验失败后只额外调用一次 provider。"""
        client = ResponsesClient(FakeResponses([{"steps": []}, {"steps": [{"step_id":"step_1","subgoal":"q","depends_on":[],"is_final":True}]}]), "test", 10)
        result = client.complete("p", "q", __import__("opera.schemas", fromlist=["PlanResult"]).PlanResult, "plan")
        self.assertEqual("step_1", result.steps[0].step_id)
        self.assertEqual(2, client._client.calls)

    def test_analysis_schema_requires_explicit_nullable_fields(self) -> None:
        """验证 DeepSeek strict Schema 所需的字段均被声明为 required，且可显式传空值。"""
        schema = AnalysisResult.model_json_schema()
        self.assertEqual(set(schema["properties"]), set(schema["required"]))
        with self.assertRaises(ValueError):
            AnalysisResult.model_validate({"status": "insufficient", "needs_rewrite": True})
        result = AnalysisResult.model_validate(
            {
                "status": "insufficient",
                "answer": None,
                "evidence": [],
                "needs_rewrite": True,
                "failure_reason": "missing evidence",
            }
        )
        self.assertIsNone(result.answer)
        self.assertEqual([], result.evidence)

    def test_executor_returns_final_supported_answer(self) -> None:
        """验证最终答案只来自计划的 is_final 步并通过证据校验。"""
        planner = ResponsesClient(FakeResponses([{"steps":[{"step_id":"step_1","subgoal":"find fact","depends_on":[],"is_final":False},{"step_id":"step_2","subgoal":"derive","depends_on":["step_1"],"is_final":True}]}]), "test", 10)
        analysis = ResponsesClient(FakeResponses([{"status":"supported","answer":"a","evidence":[{"chunk_id":"case-a:0","sentence_index":0}],"needs_rewrite":False,"failure_reason":None},{"status":"supported","answer":"final","evidence":[{"chunk_id":"case-a:0","sentence_index":0}],"needs_rewrite":False,"failure_reason":None}]), "test", 10)
        executor = OperaExecutor(planner, analysis, ResponsesClient(FakeResponses([]),"test",10), FakeRetriever(), {"planner":"p","analysis":"a","rewrite":"r"}, 4, 1, 3)
        result = executor.execute("question", RetrievalScope.CASE, "case-a")
        self.assertEqual("completed", result.status); self.assertEqual("final", result.answer); self.assertEqual(2, result.completed_step_count)

    def test_executor_rejects_evidence_outside_retrieval(self) -> None:
        """验证模型伪造 chunk_id 时 Executor 拒绝写入状态。"""
        planner = ResponsesClient(FakeResponses([{"steps":[{"step_id":"step_1","subgoal":"q","depends_on":[],"is_final":True}]}]), "test", 10)
        analysis = ResponsesClient(FakeResponses([{"status":"supported","answer":"a","evidence":[{"chunk_id":"forged","sentence_index":0}],"needs_rewrite":False,"failure_reason":None}]), "test", 10)
        executor = OperaExecutor(planner, analysis, ResponsesClient(FakeResponses([]),"test",10), FakeRetriever(), {"planner":"p","analysis":"a","rewrite":"r"}, 4, 1, 3)
        with self.assertRaises(ValueError): executor.execute("q", RetrievalScope.CASE, "case-a")

    def test_executor_appends_dependency_answer_before_retrieval(self) -> None:
        """验证后续检索由 Executor 自动追加已完成步骤的答案。"""
        planner = ResponsesClient(FakeResponses([{"steps":[{"step_id":"step_1","subgoal":"identify company","depends_on":[],"is_final":False},{"step_id":"step_2","subgoal":"where is the company headquartered?","depends_on":["step_1"],"is_final":True}]}]), "test", 10)
        analysis = ResponsesClient(FakeResponses([{"status":"supported","answer":"Facebook","evidence":[{"chunk_id":"case-a:0","sentence_index":0}],"needs_rewrite":False,"failure_reason":None},{"status":"supported","answer":"Menlo Park","evidence":[{"chunk_id":"case-a:0","sentence_index":0}],"needs_rewrite":False,"failure_reason":None}]), "test", 10)
        retriever = FakeRetriever()
        executor = OperaExecutor(planner, analysis, ResponsesClient(FakeResponses([]), "test", 10), retriever, {"planner":"p","analysis":"a","rewrite":"r"}, 4, 1, 3)

        executor.execute("question", RetrievalScope.CASE, "case-a")

        self.assertEqual(
            ["identify company", "where is the company headquartered?\nKnown results:\nstep_1: Facebook"],
            retriever.queries,
        )

    def test_plan_rejects_non_consecutive_step_ids(self) -> None:
        """验证两步计划不能跳过 step_2 而直接声明 step_3。"""
        with self.assertRaisesRegex(ValueError, "consecutive"):
            PlanResult.model_validate(
                {
                    "steps": [
                        {"step_id": "step_1", "subgoal": "find entity", "depends_on": [], "is_final": False},
                        {"step_id": "step_3", "subgoal": "final answer", "depends_on": ["step_1"], "is_final": True},
                    ]
                }
            )

    def test_rewrite_receives_current_evidence_preview(self) -> None:
        """验证 Rewrite 根据当前候选 paragraph 和失败原因生成下一轮 query。"""
        planner = ResponsesClient(FakeResponses([{"steps":[{"step_id":"step_1","subgoal":"find fact","depends_on":[],"is_final":True}]}]), "test", 10)
        analysis = ResponsesClient(FakeResponses([{"status":"insufficient","answer":None,"evidence":[],"needs_rewrite":True,"failure_reason":"missing year"},{"status":"supported","answer":"answer","evidence":[{"chunk_id":"case-a:0","sentence_index":0}],"needs_rewrite":False,"failure_reason":None}]), "test", 10)
        rewrite_responses = FakeResponses([{"rewritten_query":"find fact year","reason":"add year"}])
        executor = OperaExecutor(planner, analysis, ResponsesClient(rewrite_responses, "test", 10), FakeRetriever(), {"planner":"p","analysis":"a","rewrite":"r"}, 4, 1, 3)

        executor.execute("question", RetrievalScope.CASE, "case-a")

        rewrite_input = json.loads(str(rewrite_responses.requests[0]["input"]))
        self.assertEqual("find fact", rewrite_input["subgoal"])
        self.assertEqual("missing year", rewrite_input["failure_reason"])
        self.assertEqual([{"chunk_id":"case-a:0", "title":"A", "sentences":["fact"]}], rewrite_input["untrusted_evidence"])

    def test_rewrite_keeps_subgoal_query_and_dependencies_separate(self) -> None:
        """验证 Rewrite 输入不再将实际 query 错标为原子子目标。"""
        planner = ResponsesClient(FakeResponses([{"steps":[{"step_id":"step_1","subgoal":"identify actress","depends_on":[],"is_final":False},{"step_id":"step_2","subgoal":"find government position","depends_on":["step_1"],"is_final":True}]}]), "test", 10)
        analysis = ResponsesClient(FakeResponses([{"status":"supported","answer":"Shirley Temple","evidence":[{"chunk_id":"case-a:0","sentence_index":0}],"needs_rewrite":False,"failure_reason":None},{"status":"insufficient","answer":None,"evidence":[],"needs_rewrite":True,"failure_reason":"missing title"},{"status":"supported","answer":"answer","evidence":[{"chunk_id":"case-a:0","sentence_index":0}],"needs_rewrite":False,"failure_reason":None}]), "test", 10)
        rewrite_responses = FakeResponses([{"rewritten_query":"Shirley Temple government title","reason":"add entity"}])
        executor = OperaExecutor(planner, analysis, ResponsesClient(rewrite_responses, "test", 10), FakeRetriever(), {"planner":"p","analysis":"a","rewrite":"r"}, 4, 1, 3)

        executor.execute("question", RetrievalScope.CASE, "case-a")

        rewrite_input = json.loads(str(rewrite_responses.requests[0]["input"]))
        self.assertEqual("find government position", rewrite_input["subgoal"])
        self.assertEqual("find government position\nKnown results:\nstep_1: Shirley Temple", rewrite_input["current_query"])
        self.assertEqual({"step_1": "Shirley Temple"}, rewrite_input["accepted_dependencies"])
        self.assertEqual("missing title", rewrite_input["failure_reason"])
        self.assertEqual([{"chunk_id":"case-a:0", "title":"A", "sentences":["fact"]}], rewrite_input["untrusted_evidence"])

    def test_executor_allows_two_rewrites(self) -> None:
        """验证 YAML 上限允许两次 Rewrite 时可正常写入最终 StepResult。"""
        planner = ResponsesClient(FakeResponses([{"steps":[{"step_id":"step_1","subgoal":"find fact","depends_on":[],"is_final":True}]}]), "test", 10)
        analysis = ResponsesClient(FakeResponses([{"status":"insufficient","answer":None,"evidence":[],"needs_rewrite":True,"failure_reason":"need alias"},{"status":"insufficient","answer":None,"evidence":[],"needs_rewrite":True,"failure_reason":"need year"},{"status":"supported","answer":"answer","evidence":[{"chunk_id":"case-a:0","sentence_index":0}],"needs_rewrite":False,"failure_reason":None}]), "test", 10)
        rewrite_responses = FakeResponses([{"rewritten_query":"find alias","reason":"add alias"},{"rewritten_query":"find alias year","reason":"add year"}])
        executor = OperaExecutor(planner, analysis, ResponsesClient(rewrite_responses, "test", 10), FakeRetriever(), {"planner":"p","analysis":"a","rewrite":"r"}, 4, 2, 3)

        result = executor.execute("question", RetrievalScope.CASE, "case-a")

        self.assertEqual("completed", result.status)
        self.assertEqual(2, rewrite_responses.calls)

    def test_executor_writes_local_trace_with_retrieved_paragraphs(self) -> None:
        """验证调试轨迹保留实际送入 Analysis 的 paragraph，但不保存原始问题。"""
        planner = ResponsesClient(FakeResponses([{"steps":[{"step_id":"step_1","subgoal":"find fact","depends_on":[],"is_final":True}]}]), "test", 10)
        analysis = ResponsesClient(FakeResponses([{"status":"insufficient","answer":None,"evidence":[],"needs_rewrite":True,"failure_reason":"missing year"},{"status":"supported","answer":"answer","evidence":[{"chunk_id":"case-a:0","sentence_index":0}],"needs_rewrite":False,"failure_reason":None}]), "test", 10)
        rewrite_responses = FakeResponses([{"rewritten_query":"find fact year","reason":"add year"}])
        with tempfile.TemporaryDirectory() as directory:
            executor = OperaExecutor(
                planner,
                analysis,
                ResponsesClient(rewrite_responses, "test", 10),
                FakeRetriever(),
                {"planner":"p","analysis":"a","rewrite":"r"},
                4,
                2,
                3,
                trace_writer=OperaTraceWriter(True, Path(directory), logging.getLogger("test")),
            )
            result = executor.execute("secret question", RetrievalScope.CASE, "case-a")
            trace = json.loads((Path(directory) / f"{result.run_id}.json").read_text(encoding="utf-8"))

        self.assertNotIn("question", trace)
        self.assertEqual("find fact", trace["plan"]["steps"][0]["subgoal"])
        first_attempt = trace["steps"][0]["retrieval_attempts"][0]
        self.assertEqual([{"chunk_id":"case-a:0", "title":"A", "sentences":["fact"]}], first_attempt["retrieved_paragraphs"])
        self.assertEqual(12, first_attempt["retrieval_summary"]["duration_ms"])
        self.assertEqual("insufficient", first_attempt["analysis"]["status"])
        self.assertEqual("find fact year", first_attempt["rewrite"]["rewritten_query"])
        self.assertEqual("completed", trace["final_response"]["status"])

    def test_executor_writes_provider_metadata_for_empty_output(self) -> None:
        """验证 Planner 空输出会把安全 provider 元信息写入本地 debug trace。"""

        with tempfile.TemporaryDirectory() as directory:
            executor = OperaExecutor(
                ResponsesClient(EmptyOutputResponses(), "test", 10),
                ResponsesClient(FakeResponses([]), "test", 10),
                ResponsesClient(FakeResponses([]), "test", 10),
                FakeRetriever(),
                {"planner": "p", "analysis": "a", "rewrite": "r"},
                4,
                2,
                3,
                trace_writer=OperaTraceWriter(True, Path(directory), logging.getLogger("test")),
            )
            with self.assertRaises(ResponseOutputTextError):
                executor.execute("secret question", RetrievalScope.CASE, "case-a")
            trace_path = next(Path(directory).glob("*.json"))
            trace = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual("opera_execution_trace_v2", trace["trace_version"])
        self.assertEqual("opera_plan", trace["provider_response"]["schema_name"])
        self.assertEqual("failed", trace["provider_response"]["provider_response_status"])
        self.assertEqual([], trace["provider_response"]["provider_output_item_types"])

    def test_opera_http_contract_rejects_invalid_scope_before_runtime(self) -> None:
        """验证 FastAPI 在未创建外部客户端前拒绝 all 与 case_id 的非法组合。"""
        from fastapi.testclient import TestClient
        with TestClient(main.app) as client:
            response = client.post("/opera/ask", json={"question":"q","retrieval_scope":"all","case_id":"case-a"})
        self.assertEqual(422, response.status_code)

if __name__ == "__main__": unittest.main()
