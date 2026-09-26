import copy
import threading
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.job_runner import PIPELINE_LOCK, run_pipeline_job
from backend.api.main import app
from backend.api.session_store import JobState, SessionState, get_job, jobs, save_job, save_session, sessions

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_stores():
    sessions.clear()
    jobs.clear()
    if PIPELINE_LOCK.locked():
        try:
            PIPELINE_LOCK.release()
        except RuntimeError:
            pass
    yield
    sessions.clear()
    jobs.clear()
    if PIPELINE_LOCK.locked():
        try:
            PIPELINE_LOCK.release()
        except RuntimeError:
            pass


def create_mock_session(session_id="sess_1"):
    sess = SessionState(
        session_id=session_id,
        image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
        image_rgb=np.zeros((100, 100, 3), dtype=np.uint8),
        faces=[],
        cropped_path="outputs/app_uploads/fake_crop.png",
        gender_word="man",
        initial_age=10,
        photo_year=2010,
    )
    save_session(sess)
    return sess


def test_job_runner_success():
    sess = create_mock_session()
    job_id = "job_test_1"
    save_job(JobState(job_id=job_id, session_id=sess.session_id))

    base_config = {
        "paths": {
            "output_dir": "outputs/edited_images",
            "specialized_unet_ckpt": "checkpoints/test",
            "gallery_test_dir": "data/test_gallery"
        }
    }

    with patch("main.run_specialization", return_value="checkpoints/test") as mock_spec, \
         patch("main.run_inversion", return_value=("z_T", "null", "attn")) as mock_inv, \
         patch("main.run_editing", return_value={30: "outputs/jobs/job_test_1/age_30.png"}) as mock_edit, \
         patch("main.run_embedding_and_search", return_value=({"01366": 0.85}, True, "01366", 0.85)) as mock_search:

        PIPELINE_LOCK.acquire()
        run_pipeline_job(job_id, sess, base_config)

    job = get_job(job_id)
    assert job.status == "done"
    assert job.result["accepted"] is True
    assert job.result["top_identity"] == "01366"
    assert 30 in job.result["edited_images"]
    # Lock must be released
    assert not PIPELINE_LOCK.locked()


def test_job_runner_failure_releases_lock():
    sess = create_mock_session()
    job_id = "job_test_fail"
    save_job(JobState(job_id=job_id, session_id=sess.session_id))

    base_config = {"paths": {"output_dir": "outputs/test"}}

    with patch("main.run_specialization", side_effect=RuntimeError("CUDA out of memory")):
        PIPELINE_LOCK.acquire()
        run_pipeline_job(job_id, sess, base_config)

    job = get_job(job_id)
    assert job.status == "error"
    assert "CUDA out of memory" in job.error_message
    # Critical test: lock must be released even upon crash
    assert not PIPELINE_LOCK.locked()


def test_pipeline_mutex_prevents_concurrent_runs():
    sess = create_mock_session("sess_mutex")

    # Manually lock the GPU mutex to simulate another running job
    PIPELINE_LOCK.acquire()

    try:
        resp = client.post(f"/api/sessions/{sess.session_id}/run", json={})
        assert resp.status_code == 409
        assert "Hệ thống đang bận" in resp.json()["detail"]
    finally:
        PIPELINE_LOCK.release()


def test_config_isolation_between_jobs():
    sess = create_mock_session()
    base_config = {
        "paths": {
            "output_dir": "outputs/edited_images",
            "gallery_test_dir": "data/default_gallery"
        }
    }
    base_config_before = copy.deepcopy(base_config)

    captured_configs = []

    def mock_editing(cfg, *args, **kwargs):
        captured_configs.append(copy.deepcopy(cfg))
        return {30: "fake.png"}

    with patch("main.run_specialization", return_value="ckpt"), \
         patch("main.run_inversion", return_value=(1, 2, 3)), \
         patch("main.run_editing", side_effect=mock_editing), \
         patch("main.run_embedding_and_search", return_value=({}, False, "", 0.0)):

        # Job 1
        save_job(JobState(job_id="job_A", session_id=sess.session_id))
        run_pipeline_job("job_A", sess, base_config, gallery_dir="data/custom_gallery")

        # Job 2
        save_job(JobState(job_id="job_B", session_id=sess.session_id))
        run_pipeline_job("job_B", sess, base_config)

    # Job A output dir must contain job_A
    assert "job_A" in captured_configs[0]["paths"]["output_dir"]
    assert captured_configs[0]["paths"]["gallery_test_dir"] == "data/custom_gallery"

    # Job B output dir must contain job_B
    assert "job_B" in captured_configs[1]["paths"]["output_dir"]
    assert captured_configs[1]["paths"]["gallery_test_dir"] == "data/default_gallery"

    # Base config must remain unmutated
    assert base_config == base_config_before
