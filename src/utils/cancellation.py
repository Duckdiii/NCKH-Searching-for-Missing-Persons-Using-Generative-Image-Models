"""Cooperative cancellation, scoped to the current thread/async task."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class TaskCancelled(Exception):
    pass


_current_event: ContextVar[Event | None] = ContextVar('cancellation_event', default=None)


def checkpoint() -> None:
    event = _current_event.get()
    if event is not None and event.is_set():
        raise TaskCancelled('Phiên đã được yêu cầu dừng.')


@contextmanager
def cancellation_scope(event: Event):
    token = _current_event.set(event)
    try:
        checkpoint()
        yield
    finally:
        _current_event.reset(token)
