"""T04–T06 — Test nhánh reference persistence (mock DB/storage, model thật không cần).

Chạy:  .\\.venv\\Scripts\\python.exe -m pytest backend/tests/test_reference_persistence.py -q
Hoặc system python (có fastapi):  python -m pytest backend/tests/test_reference_persistence.py -q
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402
from backend.api.job_runner import PIPELINE_LOCK  # noqa: E402
from backend.api.session_store import get_session, jobs, sessions  # noqa: E402

client = TestClient(app)


def _png_bytes(color=(200, 150, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=color).save(buf, format="PNG")
    return buf.getvalue()


class MockFace:
    def __init__(self, bbox, det_score=0.95):
        self.bbox = bbox
        self.det_score = det_score
        self.kps = np.array(
            [[80.0, 90.0], [140.0, 90.0], [110.0, 120.0], [90.0, 150.0], [130.0, 150.0]],
            dtype=np.float32,
        )


def _upload(mock_faces, persisted=None):
    with patch("backend.api.routers.session.get_embedder") as mock_emb, patch(
        "backend.api.routers.session.persist_reference_upload", return_value=persisted
    ):
        mock_emb.return_value = MagicMock(detect_faces=MagicMock(return_value=mock_faces))
        return client.post("/api/sessions", files={"file": ("t.png", _png_bytes(), "image/png")})


def setup_function(_):
    sessions.clear()
    jobs.clear()
    # Worker thật release lock trong finally; worker mock trong test thì không.
    if PIPELINE_LOCK.locked():
        try:
            PIPELINE_LOCK.release()
        except RuntimeError:
            pass


def teardown_function(_):
    if PIPELINE_LOCK.locked():
        try:
            PIPELINE_LOCK.release()
        except RuntimeError:
            pass


def test_upload_without_db_keeps_legacy_behavior():
    resp = _upload([MockFace([50.0, 50.0, 180.0, 180.0])], persisted=None)
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_id"] is None
    assert data["detection_ids"] == []
    assert get_session(data["session_id"]).source_id is None


def test_upload_persists_reference_lineage():
    persisted = {
        "source_id": "src-1",
        "asset_id": "a-1",
        "frame_id": "f-1",
        "run_id": "r-1",
        "detection_ids": ["d-0"],
        "storage_key": "reference/originals/src-1/a-1.png",
    }
    resp = _upload([MockFace([50.0, 50.0, 180.0, 180.0])], persisted=persisted)
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_id"] == "src-1"
    assert data["detection_ids"] == ["d-0"]
    sess = get_session(data["session_id"])
    assert sess.source_id == "src-1" and sess.frame_id == "f-1"


def test_select_face_creates_crop_revision():
    persisted = {
        "source_id": "src-1", "asset_id": "a-1", "frame_id": "f-1",
        "run_id": "r-1", "detection_ids": ["d-0"],
        "storage_key": "reference/originals/src-1/a-1.png",
    }
    session_id = _upload([MockFace([30.0, 30.0, 150.0, 150.0])], persisted).json()["session_id"]
    revision = {"crop_id": "crop-1", "asset_id": "ca-1",
                "storage_key": "reference/crops/d-0/ca-1.png", "url": "/outputs/media/x.png"}
    with patch(
        "backend.api.routers.session.persist_crop_revision", return_value=revision
    ) as mock_persist:
        resp = client.post(f"/api/sessions/{session_id}/select-face", json={"selected_idx": 0})
    assert resp.status_code == 200
    assert resp.json()["crop_id"] == "crop-1"
    assert mock_persist.call_count == 1
    kwargs = mock_persist.call_args.kwargs
    assert kwargs["purpose"] == "reference" and kwargs["detection_id"] == "d-0"
    sess = get_session(session_id)
    assert sess.current_crop_id == "crop-1" and sess.chosen_detection_id == "d-0"


def test_select_face_without_lineage_skips_db_but_keeps_preview():
    session_id = _upload([MockFace([30.0, 30.0, 150.0, 150.0])], persisted=None).json()["session_id"]
    with patch(
        "backend.api.routers.session.persist_crop_revision", return_value={"crop_id": "x"}
    ) as mock_persist:
        resp = client.post(f"/api/sessions/{session_id}/select-face", json={"selected_idx": 0})
    assert resp.status_code == 200
    assert resp.json()["crop_id"] is None
    assert mock_persist.call_count == 0  # không detection_id → không gọi DB
    assert get_session(session_id).current_crop_id is None


def test_apply_restore_creates_new_revision():
    persisted = {
        "source_id": "src-1", "asset_id": "a-1", "frame_id": "f-1",
        "run_id": "r-1", "detection_ids": ["d-0"],
        "storage_key": "reference/originals/src-1/a-1.png",
    }
    session_id = _upload([MockFace([30.0, 30.0, 150.0, 150.0])], persisted).json()["session_id"]
    with patch(
        "backend.api.routers.session.persist_crop_revision",
        side_effect=[{"crop_id": "crop-1"}, {"crop_id": "crop-2"}],
    ):
        client.post(f"/api/sessions/{session_id}/select-face", json={"selected_idx": 0})
        resp = client.post(
            f"/api/sessions/{session_id}/apply-restore",
            json={"mode": "auto", "use_restored": False},
        )
    assert resp.status_code == 200
    assert resp.json()["crop_id"] == "crop-2"  # revision mới, không ghi đè crop-1
    assert get_session(session_id).current_crop_id == "crop-2"


def _ready_session():
    persisted = {
        "source_id": "src-1", "asset_id": "a-1", "frame_id": "f-1",
        "run_id": "r-1", "detection_ids": ["d-0"],
        "storage_key": "reference/originals/src-1/a-1.png",
    }
    session_id = _upload([MockFace([30.0, 30.0, 150.0, 150.0])], persisted).json()["session_id"]
    with patch(
        "backend.api.routers.session.persist_crop_revision",
        return_value={"crop_id": "crop-1"},
    ):
        client.post(f"/api/sessions/{session_id}/select-face", json={"selected_idx": 0})
    client.post(
        f"/api/sessions/{session_id}/resolve-age",
        json={"mode": "manual", "manual_age": 20, "gender_word": "man"},
    )
    return session_id


def test_run_freezes_input_crop_and_records_db_job():
    session_id = _ready_session()
    with patch(
        "backend.api.routers.jobs.create_generation_job_record", return_value="db-job-1"
    ) as mock_create, patch("backend.api.routers.jobs.run_pipeline_job") as mock_run:
        resp = client.post(
            f"/api/sessions/{session_id}/run", json={"photo_year": 2010}
        )
    assert resp.status_code in (200, 202)
    assert mock_create.call_count == 1
    assert mock_create.call_args.kwargs["input_crop_id"] == "crop-1"  # chốt lúc chạy
    assert mock_run.call_count == 1
    args = mock_run.call_args.args
    assert args[4] == "db-job-1" and args[5] == "crop-1"  # db_job_id, frozen crop
    assert jobs[resp.json()["job_id"]].db_job_id == "db-job-1"


def test_run_legacy_without_crop_skips_db_record():
    session_id = _upload([MockFace([30.0, 30.0, 150.0, 150.0])], persisted=None).json()["session_id"]
    sess = get_session(session_id)
    sess.cropped_path = "outputs/app_uploads/fake_crop.png"
    sess.initial_age = 20
    with patch(
        "backend.api.routers.jobs.create_generation_job_record"
    ) as mock_create, patch("backend.api.routers.jobs.run_pipeline_job"):
        resp = client.post(f"/api/sessions/{session_id}/run", json={"photo_year": 2010})
    assert resp.status_code in (200, 202)
    assert mock_create.call_count == 0


def test_generation_job_detail_endpoint():
    detail = {
        "job_id": "j-1", "input_crop_id": "crop-1", "status": "done",
        "model_name": "m", "model_version": "v", "initial_age": 20,
        "parameters": {}, "error_message": None,
        "created_at": "2026-01-01", "finished_at": "2026-01-02",
        "variants": [{"id": "g-1", "target_age": 30, "variant_index": 0,
                      "seed": None, "image_url": "/outputs/media/x.png"}],
    }
    with patch("backend.api.routers.jobs.db_ping", return_value=True), patch(
        "backend.api.routers.jobs.get_generation_job_detail", return_value=detail
    ):
        resp = client.get("/api/generation-jobs/j-1")
    assert resp.status_code == 200
    assert resp.json()["variants"][0]["target_age"] == 30

    with patch("backend.api.routers.jobs.db_ping", return_value=True), patch(
        "backend.api.routers.jobs.get_generation_job_detail", return_value=None
    ):
        assert client.get("/api/generation-jobs/missing").status_code == 404

    with patch("backend.api.routers.jobs.db_ping", return_value=False):
        assert client.get("/api/generation-jobs/j-1").status_code == 503


def test_set_done_carries_variants():
    from backend.api.job_runner import set_done
    from backend.api.session_store import JobState, save_job

    save_job(JobState(job_id="jv", session_id="s"))
    variants = [{"id": "g-1", "target_age": 30, "variant_index": 0,
                 "seed": None, "image_url": "/u.png"}]
    set_done("jv", {30: "outputs/jobs/jv/age_30.png"}, {}, False, "", 0.0,
             variants=variants)
    assert jobs["jv"].result["variants"] == variants
    assert jobs["jv"].result["edited_images"][30].endswith("age_30.png")
