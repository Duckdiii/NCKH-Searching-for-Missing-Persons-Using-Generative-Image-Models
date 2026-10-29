"""Dừng quá trình + xóa phiên (không cần DB live, model mock)."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.job_runner import (  # noqa: E402
    CANCELLED_MESSAGE,
    PIPELINE_LOCK,
    is_cancelled,
    request_cancel,
    run_pipeline_job,
)
from backend.api.main import app  # noqa: E402
from backend.api.session_store import (  # noqa: E402
    JobState,
    SessionState,
    get_session,
    jobs,
    save_job,
    save_session,
    sessions,
)

client = TestClient(app, raise_server_exceptions=False)


def setup_function(_):
    sessions.clear()
    jobs.clear()
    if PIPELINE_LOCK.locked():
        try:
            PIPELINE_LOCK.release()
        except RuntimeError:
            pass


def teardown_function(_):
    sessions.clear()
    jobs.clear()
    if PIPELINE_LOCK.locked():
        try:
            PIPELINE_LOCK.release()
        except RuntimeError:
            pass


def _mock_session(session_id="sess_stop"):
    sess = SessionState(
        session_id=session_id,
        image_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
        image_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        faces=[],
        cropped_path="outputs/app_uploads/fake.png",
        initial_age=20,
    )
    save_session(sess)
    return sess


def test_cancel_unknown_job_404():
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_cancel_finished_job_409():
    save_job(JobState(job_id="j-done", session_id="s", status="done"))
    resp = client.post("/api/jobs/j-done/cancel")
    assert resp.status_code == 409


def test_request_cancel_only_running():
    assert request_cancel("missing") is False
    save_job(JobState(job_id="j-run", session_id="s", status="running"))
    assert request_cancel("j-run") is True
    assert is_cancelled("j-run") is True


def test_worker_honors_cancel_before_stage1():
    sess = _mock_session()
    save_job(JobState(job_id="j-cancel", session_id=sess.session_id,
                      status="running"))
    assert request_cancel("j-cancel") is True
    with patch("main.run_specialization") as mock_spec:
        PIPELINE_LOCK.acquire()
        run_pipeline_job("j-cancel", sess, {"paths": {"output_dir": "o"}})
    mock_spec.assert_not_called()
    job = jobs["j-cancel"]
    assert job.status == "error"
    assert job.error_message == CANCELLED_MESSAGE
    assert not PIPELINE_LOCK.locked()
    assert is_cancelled("j-cancel") is False  # cờ dọn sau khi xong


def test_cancel_endpoint_running_job():
    save_job(JobState(job_id="j-ep", session_id="s", status="running"))
    resp = client.post("/api/jobs/j-ep/cancel")
    assert resp.status_code == 200
    assert resp.json() == {"job_id": "j-ep", "status": "cancel_requested"}
    assert is_cancelled("j-ep") is True


def test_delete_unknown_session_404():
    assert client.delete("/api/sessions/nope").status_code == 404


def test_delete_ram_session_and_legacy_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sess = _mock_session("sess_del")
    os.makedirs(os.path.join("outputs", "app_uploads"), exist_ok=True)
    for name in ("sess_del_crop.png", "sess_del_preview.png"):
        with open(os.path.join("outputs", "app_uploads", name), "wb") as fh:
            fh.write(b"x")
    resp = client.delete("/api/sessions/sess_del")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"]["ram"] is True
    assert sorted(data["deleted"]["files"]) == [
        "sess_del_crop.png", "sess_del_preview.png"]
    assert get_session("sess_del") is None
    # Xóa lần nữa -> 404 (không còn gì)
    assert client.delete("/api/sessions/sess_del").status_code == 404
