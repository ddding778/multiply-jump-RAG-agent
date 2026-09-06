"""请求隔离的演示事件；不写日志、磁盘或远端观测。"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

EventSink = Callable[[str, dict[str, object]], None]
_sink: ContextVar[EventSink | None] = ContextVar("opera_event_sink", default=None)
_step: ContextVar[str | None] = ContextVar("opera_event_step", default=None)


@contextmanager
def event_sink(sink: EventSink) -> Iterator[None]:
    """绑定本请求的事件接收器；参数 sink 接收事件类型与数据，退出时恢复上下文，无返回值。"""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


@contextmanager
def event_step(step_id: str) -> Iterator[None]:
    """绑定当前子目标；参数 step_id 为计划步骤 ID，退出时恢复上下文，无返回值。"""
    token = _step.set(step_id)
    try:
        yield
    finally:
        _step.reset(token)


def emit(event_type: str, **data: object) -> None:
    """发送实际执行事件；参数为类型和白名单业务字段，无返回值，未订阅时不执行操作。"""
    sink = _sink.get()
    if sink is not None:
        sink(event_type, {"step_id": _step.get(), **data})
