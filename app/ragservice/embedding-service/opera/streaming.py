"""本机演示的有界 SSE 通道：同步执行器在独立线程中运行。"""

import asyncio
from collections.abc import AsyncIterator, Callable
import json
import logging
import queue
import threading
import time
from uuid import uuid4

from .events import event_sink
from .schemas import OperaAskRequest
from .diagnostics import describe_failure

logger = logging.getLogger("rag_service")
# 演示服务每进程仅接收一个流式运行；连接断开后直到工作线程退出才释放。
stream_slot = threading.BoundedSemaphore(1)


class StreamDisconnected(Exception):
    """浏览器已经离开，在下一个事件边界停止后续模型或检索调用。"""


class OperaStream:
    """保存一次运行的有界事件队列与断开标记，不保留跨请求历史。"""

    def __init__(self, request: OperaAskRequest, executor_factory: Callable, release: Callable) -> None:
        """保存请求、执行器工厂和释放回调；创建运行 ID 与 64 条队列，无返回值。"""
        self.run_id = str(uuid4())
        self.request = request
        self.executor_factory = executor_factory
        self.release = release
        self.closed = threading.Event()
        self.queue: queue.Queue[str] = queue.Queue(maxsize=64)
        self.started = time.monotonic()
        self.sequence = 0
        self.initializing = True
        self.stage = "executor"

    def publish(self, event_type: str, data: dict[str, object]) -> None:
        """序列化并排入事件；参数为类型和业务数据，无返回值，断开或持续背压时停止执行。"""
        if self.closed.is_set():
            raise StreamDisconnected()
        if event_type == "initialization":
            self.stage = str(data.get("stage", "executor"))
        self.sequence += 1
        body = {"version": 1, "run_id": self.run_id, "seq": self.sequence,
                "type": event_type, "elapsed_ms": round((time.monotonic() - self.started) * 1000), "data": data}
        frame = f"id: {self.sequence}\nevent: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"
        deadline = time.monotonic() + 10
        while not self.closed.is_set() and time.monotonic() < deadline:
            try:
                self.queue.put(frame, timeout=0.2)
                return
            except queue.Full:
                continue
        self.closed.set()
        raise StreamDisconnected()

    def run(self) -> None:
        """在线程内运行真实执行器；无参数和返回值，失败发送安全错误事件，结束释放运行槽。"""
        try:
            with event_sink(self.publish):
                self.publish("run_started", {"retrieval_scope": "all", "top_k": self.request.top_k or 3})
                executor = self.executor_factory()
                self.initializing = False
                if self.closed.is_set():
                    raise StreamDisconnected()
                result = executor.execute(self.request.question.strip(), self.request.retrieval_scope,
                                          None, self.request.top_k, run_id=self.run_id)
                self.publish("run_completed", result.model_dump(mode="json"))
        except StreamDisconnected:
            logger.info("event=opera_stream_disconnected run_id=%s", self.run_id)
        except Exception as error:
            logger.warning("event=opera_stream_failed run_id=%s error_type=%s", self.run_id, type(error).__name__)
            code, category, message = describe_failure(error)
            try:
                self.publish("run_failed", {"code": code, "error_type": type(error).__name__,
                                           "category": category, "message": message})
            except StreamDisconnected:
                pass
        finally:
            self.release()

    async def frames(self) -> AsyncIterator[str]:
        """启动执行线程并异步读取事件；无参数，逐条返回 SSE 帧，取消时通知线程停止后续调用。"""
        worker = threading.Thread(target=self.run, name=f"opera-{self.run_id}", daemon=True)
        try:
            worker.start()
        except Exception:
            self.release()
            raise
        last_sequence = 0
        try:
            while True:
                if self.initializing and time.monotonic() - self.started > 60:
                    # HTTP 消费者结束等待，工作线程仍持有槽位，直到在途操作结束。
                    self.closed.set()
                    data = {"code": 504, "category": "initialization_timeout", "stage": self.stage,
                            "message": "执行器初始化超过 60 秒，请检查当前准备阶段和后端日志；在途初始化尚未结束时会拒绝重复运行。"}
                    body = {"version": 1, "run_id": self.run_id, "seq": last_sequence + 1,
                            "type": "run_failed", "elapsed_ms": round((time.monotonic() - self.started) * 1000), "data": data}
                    yield f"id: {last_sequence + 1}\nevent: run_failed\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"
                    break
                try:
                    frame = await asyncio.to_thread(self.queue.get, True, 1)
                except queue.Empty:
                    if self.closed.is_set() or not worker.is_alive():
                        break
                    yield ": keep-alive\n\n"
                    continue
                last_sequence = int(frame.split("\n", 1)[0].removeprefix("id: "))
                yield frame
                if "\nevent: run_completed\n" in frame or "\nevent: run_failed\n" in frame:
                    break
        finally:
            self.closed.set()
