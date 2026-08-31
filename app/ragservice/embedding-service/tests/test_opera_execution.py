"""OPERA schema、修复重试与多跳执行器的无外部依赖测试。"""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opera.executor import OperaExecutor
from opera.llm import ResponsesClient
from opera.schemas import OperaAskRequest, RetrievalScope
import main

class FakeResponses:
    """按调用顺序返回预设 Responses 文本。"""
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        """保存模拟模型输出列表。"""
        self.outputs = outputs; self.calls = 0; self.responses = self
    def create(self, **_: object) -> SimpleNamespace:
        """返回下一条 JSON 输出。"""
        value = self.outputs[self.calls]; self.calls += 1
        return SimpleNamespace(output_text=json.dumps(value))

class FakeRetriever:
    """返回固定 paragraph 的检索器替身。"""
    def search(self, *_: object) -> SimpleNamespace:
        """返回一个合法的 Hybrid 命中。"""
        item = SimpleNamespace(chunk_id="case-a:0", title="A", sentences=("fact",))
        return SimpleNamespace(hybrid=[item])

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

    def test_opera_http_contract_rejects_invalid_scope_before_runtime(self) -> None:
        """验证 FastAPI 在未创建外部客户端前拒绝 all 与 case_id 的非法组合。"""
        from fastapi.testclient import TestClient
        with TestClient(main.app) as client:
            response = client.post("/opera/ask", json={"question":"q","retrieval_scope":"all","case_id":"case-a"})
        self.assertEqual(422, response.status_code)

if __name__ == "__main__": unittest.main()
