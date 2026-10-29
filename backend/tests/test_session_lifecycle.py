import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.database import DatabaseConfigError
from backend.api import session_lifecycle as lifecycle
from backend.api import job_runner
from backend.api.session_store import SessionState, JobState, sessions, jobs, save_session, save_job
from src.utils.cancellation import checkpoint

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch('backend.api.database.get_pool', side_effect=DatabaseConfigError('not configured')):
        sessions.clear()
        jobs.clear()
        lifecycle._states.clear()
        job_runner._CANCEL_EVENTS.clear()
        yield
        sessions.clear()
        jobs.clear()
        lifecycle._states.clear()
        job_runner._CANCEL_EVENTS.clear()


def session(sid='lifecycle-test'):
    s = SessionState(sid, np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), [],
                     cropped_path='crop.png', initial_age=10, photo_year=2010)
    save_session(s)
    return s


def test_stop_interrupts_inside_gpu_stage_and_retains_session():
    s = session()
    started = threading.Event()
    def stage(_):
        started.set()
        for _ in range(500):
            checkpoint()
            time.sleep(.005)
        pytest.fail('cancellation was ignored')
    save_job(JobState('inside-stage', s.session_id))
    event = job_runner.prepare_job_task('inside-stage', s.session_id)
    with patch('main.run_specialization', side_effect=stage), patch('main.run_inversion') as next_stage:
        job_runner.PIPELINE_LOCK.acquire()
        worker = threading.Thread(target=job_runner.run_pipeline_job,
            args=('inside-stage', s, {'paths': {}}, None, None, None, event))
        worker.start()
        try:
            assert started.wait(2)
            response = client.post(f'/api/sessions/{s.session_id}/stop')
            assert response.status_code == 200
            worker.join(3)
            assert not worker.is_alive()
        finally:
            event.set()
            worker.join(3)
        next_stage.assert_not_called()
    assert jobs['inside-stage'].error_message == job_runner.CANCELLED_MESSAGE
    assert not job_runner.PIPELINE_LOCK.locked()
    assert s.session_id in sessions
    assert lifecycle.lifecycle_status(s.session_id)['active_tasks'] == 0


def test_delete_waits_for_worker_and_rejects_new_work(tmp_path):
    s = session('delete-wait')
    crop = tmp_path/'outputs/app_uploads/delete-wait_crop.png'
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b'input still in use')
    save_job(JobState('delete-job', s.session_id))
    event = lifecycle.begin_task(s.session_id)
    result = client.delete(f'/api/sessions/{s.session_id}')
    assert result.status_code == 202
    assert event.is_set()
    assert crop.exists() and s.session_id in sessions
    with pytest.raises(HTTPException):
        lifecycle.begin_task(s.session_id)
    lifecycle.finish_task(s.session_id, event)
    deadline = time.monotonic()+3
    while lifecycle.lifecycle_status(s.session_id)['status'] != 'deleted' and time.monotonic()<deadline:
        time.sleep(.01)
    assert lifecycle.lifecycle_status(s.session_id)['status'] == 'deleted'
    assert not crop.exists()
    assert s.session_id not in sessions
    assert 'delete-job' not in jobs


def test_database_failure_does_not_claim_delete_or_remove_ram():
    s = session('db-delete-fail')
    with patch('backend.api.database.get_pool', side_effect=RuntimeError('connection failed')):
        response = client.delete(f'/api/sessions/{s.session_id}')
    assert response.status_code == 503
    assert s.session_id in sessions
    assert client.delete(f'/api/sessions/{s.session_id}').status_code == 200


def test_stop_is_scoped_to_one_session():
    session('stop-a')
    session('stop-b')
    a = lifecycle.begin_task('stop-a')
    b = lifecycle.begin_task('stop-b')
    try:
        assert client.post('/api/sessions/stop-a/stop').status_code == 200
        assert a.is_set() and not b.is_set()
    finally:
        lifecycle.finish_task('stop-a', a)
        lifecycle.finish_task('stop-b', b)


def test_cancelled_scope_does_not_leak_to_next_operation():
    from src.utils.cancellation import cancellation_scope, TaskCancelled
    event = threading.Event()
    with pytest.raises(TaskCancelled):
        with cancellation_scope(event):
            event.set()
            checkpoint()
    checkpoint()


def test_setup_failure_releases_mutex_and_registered_task():
    s = session('setup-failure')
    with patch('backend.api.routers.jobs.get_config', side_effect=RuntimeError('bad config')):
        assert client.post(f'/api/sessions/{s.session_id}/run', json={}).status_code == 500
    assert not job_runner.PIPELINE_LOCK.locked()
    assert lifecycle.lifecycle_status(s.session_id)['active_tasks'] == 0
    assert all(j.status == 'error' for j in jobs.values())
    assert not job_runner._CANCEL_EVENTS
