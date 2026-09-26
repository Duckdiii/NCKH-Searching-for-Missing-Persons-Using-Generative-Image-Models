"""Track session work so deletion never races a worker still using its input."""
from dataclasses import dataclass, field
from functools import wraps
from threading import Condition, Event, Thread
import inspect

from fastapi import HTTPException
from src.utils.cancellation import TaskCancelled, cancellation_scope, checkpoint


@dataclass
class Lifecycle:
    tasks: set[Event] = field(default_factory=set)
    deleting: bool = False
    deleted: bool = False
    error: str | None = None
    removed: dict | None = None


_condition = Condition()
_states: dict[str, Lifecycle] = {}


def begin_task(session_id: str, event: Event | None = None) -> Event:
    with _condition:
        state = _states.setdefault(session_id, Lifecycle())
        if state.deleting or state.deleted or any(e.is_set() for e in state.tasks):
            raise HTTPException(409, 'Phiên đang dừng hoặc đã được xóa.')
        event = event if event is not None else Event()
        state.tasks.add(event)
        return event


def finish_task(session_id: str, event: Event) -> None:
    with _condition:
        state = _states.get(session_id)
        if state is not None:
            state.tasks.discard(event)
        _condition.notify_all()


def stop_tasks(session_id: str) -> dict:
    with _condition:
        state = _states.setdefault(session_id, Lifecycle())
        for event in state.tasks:
            event.set()
        return lifecycle_status(session_id)


def lifecycle_status(session_id: str) -> dict:
    with _condition:
        state = _states.get(session_id, Lifecycle())
        status = ('deleted' if state.deleted else 'delete_error' if state.error else
                  'deleting' if state.deleting else
                  'stopping' if any(e.is_set() for e in state.tasks) else
                  'running' if state.tasks else 'idle')
        return {'session_id': session_id, 'status': status,
                'active_tasks': len(state.tasks), 'deleted': state.removed,
                'error_message': state.error}


def delete_when_idle(session_id: str, cleanup) -> dict:
    with _condition:
        state = _states.setdefault(session_id, Lifecycle())
        if state.deleted or state.deleting:
            return lifecycle_status(session_id)
        state.deleting = True
        state.error = None
        stop_tasks(session_id)
        if not state.tasks:
            # Serialize with begin_task, including DB/file cleanup.
            try:
                state.removed = cleanup(session_id)['deleted']
                state.deleted = True
            except Exception:
                state.deleting = False
                raise
            return lifecycle_status(session_id)

        def finish_delete():
            with _condition:
                _condition.wait_for(lambda: not state.tasks)
            try:
                result = cleanup(session_id)
                with _condition:
                    state.removed = result['deleted']
                    state.deleted = True
            except Exception:
                with _condition:
                    state.error = 'Không thể xóa phiên. Kiểm tra kết nối rồi thử lại.'
            finally:
                with _condition:
                    state.deleting = False
                    _condition.notify_all()

        Thread(target=finish_delete, daemon=True, name=f'delete-session-{session_id}').start()
        return lifecycle_status(session_id)


def session_operation(func):
    """Register synchronous session API work; preserve FastAPI's signature."""
    signature = inspect.signature(func)

    @wraps(func)
    def wrapped(*args, **kwargs):
        sid = signature.bind(*args, **kwargs).arguments['session_id']
        event = begin_task(sid)
        try:
            with cancellation_scope(event):
                result = func(*args, **kwargs)
                checkpoint()
                return result
        except TaskCancelled as exc:
            raise HTTPException(409, str(exc)) from None
        finally:
            finish_task(sid, event)
    return wrapped


def job_operation(func):
    signature = inspect.signature(func)
    @wraps(func)
    def wrapped(*args, **kwargs):
        from backend.api.session_store import get_job
        job_id = signature.bind(*args, **kwargs).arguments['job_id']
        job = get_job(job_id)
        sid = job.session_id if job else None
        if sid is None:
            try:
                from backend.api.database import get_pool
                with get_pool().connection() as conn:
                    row = conn.execute('SELECT session_id FROM face_media.generation_jobs WHERE id = %s', (job_id,)).fetchone()
                sid = str(row[0]) if row and row[0] else None
            except Exception:
                pass
        # Legacy outputs without a saved session cannot be associated with another session.
        sid = sid or f'job-{job_id}'
        event = begin_task(sid)
        try:
            with cancellation_scope(event):
                result = func(*args, **kwargs)
                checkpoint()
                return result
        except TaskCancelled as exc:
            raise HTTPException(409, str(exc)) from None
        finally:
            finish_task(sid, event)
    return wrapped
