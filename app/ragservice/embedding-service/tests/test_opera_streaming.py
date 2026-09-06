"""本地实时演示的事件顺序、输入真实性、权限与中断回归测试。"""

import asyncio
import json
import os
import queue
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from test_opera_execution import FakeResponses, FakeRetriever
from opera.events import event_sink
from opera.executor import OperaExecutor
from opera.llm import ResponsesClient
from opera.schemas import OperaAskRequest, RetrievalScope
from opera.streaming import OperaStream, StreamDisconnected
from opera.diagnostics import check_demo_dependencies, DemoDependencyError
from pathlib import Path
import main

PLAN = {"steps": [{"step_id": "step_1", "subgoal": "find fact", "depends_on": [], "is_final": True}]}
ANSWER = {"status": "supported", "answer": "answer", "evidence": [{"chunk_id": "case-a:0", "sentence_index": 0}], "needs_rewrite": False, "failure_reason": None}
INSUFFICIENT = {"status": "insufficient", "answer": None, "evidence": [], "needs_rewrite": True, "failure_reason": "need evidence"}


def executor(outputs=None, planner=None, rewrites=0):
    """创建无外部调用的真实执行器；参数覆盖模型输出与改写次数，返回执行器和 provider 替身。"""
    provider = FakeResponses(outputs or [ANSWER])
    return OperaExecutor(ResponsesClient(planner or FakeResponses([PLAN]), "test", 100),
        ResponsesClient(provider, "test", 100),
        ResponsesClient(FakeResponses([{"rewritten_query": "new query", "reason": "more evidence"}]), "test", 100),
        FakeRetriever(), {"planner": "actual planner prompt", "analysis": "actual analysis prompt", "rewrite": "rewrite"},
        4, rewrites, 3), provider


class StreamTests(unittest.TestCase):
    """验证公开接口和真实调用处发出的事件合同。"""

    def test_dependency_failures_are_specific_and_do_not_leak_details(self):
        """无参数；验证 Qdrant 断开与 collection 缺失的分类，错误不含底层异常正文，无返回值。"""
        env = {"OPERA_INDEX_VERSION": "a" * 64, "DEEPSEEK_API_KEY": "test", "DASHSCOPE_API_KEY": "test"}
        with patch.dict(os.environ, env), patch.object(Path, "is_file", return_value=True), \
                patch("opera.diagnostics.QdrantClient") as factory:
            client = factory.return_value
            client.get_collections.side_effect = RuntimeError("private connection details")
            with self.assertRaises(DemoDependencyError) as caught:
                check_demo_dependencies(Path("."))
            self.assertEqual("qdrant", caught.exception.category)
            self.assertNotIn("private", str(caught.exception))
            client.close.assert_called_once()
            client.get_collections.side_effect = None
            client.get_collection.side_effect = RuntimeError("private collection details")
            with self.assertRaises(DemoDependencyError) as caught:
                check_demo_dependencies(Path("."))
            self.assertEqual("index", caught.exception.category)
        with patch.dict(os.environ, env), patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(DemoDependencyError) as caught:
                check_demo_dependencies(Path("."))
            self.assertEqual("index", caught.exception.category)

    def test_initialization_timeout_ends_stream_but_holds_slot_until_worker_exits(self):
        """无参数；验证初始化截止后发送终态，不提前释放工作线程的运行槽，无返回值。"""
        gate = threading.Event()
        released = threading.Event()
        instance, provider = executor()

        def factory():
            """模拟慢初始化；无参数，门闩打开后返回执行器。"""
            gate.wait(5)
            return instance

        stream = OperaStream(OperaAskRequest(question="q", retrieval_scope="all"), factory, released.set)

        async def consume():
            """消费首事件后推动超时；无参数和返回值，验证连续序号与终态。"""
            frames = stream.frames()
            try:
                self.assertIn("event: run_started", await anext(frames))
                stream.started -= 61
                frame = await anext(frames)
                self.assertIn("id: 2", frame)
                self.assertIn("initialization_timeout", frame)
                self.assertFalse(released.is_set())
            finally:
                gate.set()
                await frames.aclose()

        asyncio.run(consume())
        self.assertTrue(released.wait(5))
        self.assertEqual(0, provider.calls)

    def test_actual_input_output_and_validation_order(self):
        """无参数；验证实际请求输入和 prompt 原样展示，并在校验通过之后才接受子目标，无返回值。"""
        instance, provider = executor()
        events = []
        with event_sink(lambda kind, data: events.append((kind, data))):
            result = instance.execute("question", RetrievalScope.ALL, None)
        analysis = next(data for kind, data in events if kind == "agent_started" and data["agent"] == "opera_analysis")
        self.assertEqual(provider.requests[0]["input"], analysis["input"])
        self.assertEqual(provider.requests[0]["instructions"], analysis["instructions"])
        self.assertEqual("step_1", analysis["step_id"])
        kinds = [kind for kind, _ in events]
        self.assertLess(kinds.index("agent_output"), kinds.index("agent_validated"))
        self.assertEqual("step_completed", kinds[-1])
        self.assertEqual("completed", result.status)

    def test_repair_and_rewrite_are_separate_real_calls(self):
        """无参数；验证 Schema 修复保留原始无效输出与真实修复输入，Rewrite 另有节点，无返回值。"""
        planner = FakeResponses([{"steps": []}, PLAN])
        instance, _ = executor([INSUFFICIENT, ANSWER], planner, 1)
        events = []
        with event_sink(lambda kind, data: events.append((kind, data))):
            instance.execute("question", RetrievalScope.ALL, None)
        starts = [data for kind, data in events if kind == "agent_started"]
        self.assertEqual(5, len(starts))
        self.assertEqual(5, len({data["call_id"] for data in starts}))
        self.assertEqual(planner.requests[1]["input"], starts[1]["input"])
        self.assertTrue(any(kind == "agent_validated" and data["will_retry"] for kind, data in events))
        self.assertTrue(any(kind == "rewrite_accepted" for kind, _ in events))

    def test_forged_evidence_never_completes_step(self):
        """无参数；验证结构通过但证据伪造时不接受子目标，无返回值。"""
        instance, _ = executor([{**ANSWER, "evidence": [{"chunk_id": "forged", "sentence_index": 0}]}])
        events = []
        with event_sink(lambda kind, data: events.append(kind)), self.assertRaises(ValueError):
            instance.execute("question", RetrievalScope.ALL, None)
        self.assertNotIn("step_completed", events)

    def test_default_closed_and_local_origin_scope_boundaries(self):
        """无参数；验证默认关闭、远程来源与 case 请求被拒绝，非法参数无执行副作用，无返回值。"""
        body = {"question": "q", "retrieval_scope": "all", "top_k": 3}
        with TestClient(main.app, client=("127.0.0.1", 1234)) as client:
            with patch.dict(os.environ, {"OPERA_DEMO_STREAM_ENABLED": "0"}):
                self.assertEqual(404, client.post("/opera/ask/stream", json=body).status_code)
            with patch.dict(os.environ, {"OPERA_DEMO_STREAM_ENABLED": "1"}):
                self.assertEqual(403, client.post("/opera/ask/stream", json=body, headers={"Origin": "https://evil.example"}).status_code)
                self.assertEqual(422, client.post("/opera/ask/stream", json={**body, "retrieval_scope": "case", "case_id": "a"}).status_code)
                self.assertEqual(422, client.post("/opera/ask/stream", json={**body, "question": "  "}).status_code)
                self.assertEqual(422, client.post("/opera/ask/stream", json={**body, "question": "a" * 1001}).status_code)
        with TestClient(main.app, client=("192.0.2.1", 1234)) as client, patch.dict(os.environ, {"OPERA_DEMO_STREAM_ENABLED": "1"}):
            self.assertEqual(403, client.post("/opera/ask/stream", json=body).status_code)

    def test_http_stream_terminals_and_no_retry(self):
        """无参数；验证成功、证据不足、400 和 503 均有唯一终态、顺序 ID 与 run ID，无返回值。"""
        cases = [(executor()[0], None, "run_completed", "completed"),
                 (executor([INSUFFICIENT])[0], None, "run_completed", "insufficient"),
                 (None, ValueError("secret internal message"), "run_failed", 400),
                 (None, RuntimeError("secret internal message"), "run_failed", 503)]
        for instance, failure, terminal, value in cases:
            with self.subTest(value=value), TestClient(main.app, client=("127.0.0.1", 1234)) as client, \
                    patch.dict(os.environ, {"OPERA_DEMO_STREAM_ENABLED": "1"}), \
                    patch.object(main, "_get_demo_executor", return_value=instance, side_effect=failure) as factory:
                response = client.post("/opera/ask/stream", json={"question": "q", "retrieval_scope": "all", "top_k": 3})
                self.assertEqual(200, response.status_code)
                self.assertEqual("no-store", response.headers["cache-control"])
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                self.assertEqual("run_started", events[0]["type"])
                self.assertEqual(terminal, events[-1]["type"])
                self.assertEqual(value, events[-1]["data"].get("status", events[-1]["data"].get("code")))
                self.assertEqual(list(range(1, len(events) + 1)), [e["seq"] for e in events])
                self.assertEqual(1, len({e["run_id"] for e in events}))
                self.assertNotIn("secret internal message", response.text)
                factory.assert_called_once()

    def test_slot_rejects_duplicate_request(self):
        """无参数；验证正在运行时第二次请求立即拒绝，无返回值。"""
        with TestClient(main.app, client=("127.0.0.1", 1234)) as client, patch.dict(os.environ, {"OPERA_DEMO_STREAM_ENABLED": "1"}):
            main.stream_slot.acquire()
            try:
                response = client.post("/opera/ask/stream", json={"question": "q", "retrieval_scope": "all"})
                self.assertEqual(409, response.status_code)
            finally:
                main.stream_slot.release()

    def test_first_events_arrive_before_provider_finishes_and_disconnect_stops_next_call(self):
        """无参数；验证等待 provider 时已收到调用输入，断开后不再开始检索或 Analysis，无返回值。"""
        gate = threading.Event()
        entered = threading.Event()
        released = threading.Event()

        class BlockingProvider(FakeResponses):
            """通过门闩控制模型响应时间。"""
            def create(self, **request):
                """参数 request 为模型请求；等待测试释放后返回固定计划。"""
                entered.set()
                gate.wait(5)
                return super().create(**request)

        instance, analysis = executor(planner=BlockingProvider([PLAN]))
        stream = OperaStream(OperaAskRequest(question="q", retrieval_scope="all"), lambda: instance, released.set)
        worker = threading.Thread(target=stream.run)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            frames = []
            while not stream.queue.empty():
                frames.append(stream.queue.get_nowait())
            self.assertIn("event: agent_started", "".join(frames))
            self.assertNotIn("event: run_completed", "".join(frames))
            self.assertFalse(released.is_set())
            stream.closed.set()
        finally:
            gate.set()
            worker.join(5)
        self.assertTrue(released.is_set())
        self.assertEqual(0, analysis.calls)

    def test_event_context_does_not_leak_to_other_threads(self):
        """无参数；验证共享执行器客户端的订阅状态不会串入另一线程，无返回值。"""
        from opera.events import emit
        events = []
        with event_sink(lambda kind, data: events.append(kind)):
            thread = threading.Thread(target=lambda: emit("other_run"))
            thread.start()
            thread.join()
            emit("this_run")
        emit("unsubscribed")
        self.assertEqual(["this_run"], events)


if __name__ == "__main__":
    unittest.main()
