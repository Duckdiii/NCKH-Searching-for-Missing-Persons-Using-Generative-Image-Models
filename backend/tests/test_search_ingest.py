"""T07–T11 — Test cụm search (không cần DB live, model mock).

- Engine ingest thuần (sample timestamp, detect, crop key UUID).
- Endpoint yêu cầu DB trả 503 khi DB down (không crash).
- Camera URL redact + secret ref. Gallery snapshot/query guard.
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient  # noqa: E402

from backend.api import gallery as gallery_service  # noqa: E402
from backend.api import ingest  # noqa: E402
from backend.api.cameras import redact_url, resolve_camera_target  # noqa: E402
from backend.api.main import app  # noqa: E402
from backend.api.routers.video_verify import (  # noqa: E402
    _crop_bbox,
    _downscale_for_detect,
    cosine_sim,
)
from database.migrate import discover_migrations  # noqa: E402

client = TestClient(app)


class MockFace:
    def __init__(self, bbox=(10.0, 10.0, 60.0, 60.0), det=0.99):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = det
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0
        self.normed_embedding = v
        self.kps = None


def _mock_embedder(faces):
    emb = MagicMock()
    emb.detect_faces.return_value = faces
    emb.model_name = "buffalo_l"
    return emb


def _dummy_video(path: str, frames: int = 9, fps: float = 9.0):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (128, 128))
    for i in range(frames):
        out.write(np.full((128, 128, 3), 200 - i, dtype=np.uint8))
    out.release()


def test_migration_003_discovered():
    assert "003" in dict(discover_migrations())


def test_sample_video_frames_real_timestamps(tmp_path):
    video = str(tmp_path / "clip.mp4")
    _dummy_video(video, frames=9, fps=9.0)
    frames, meta = ingest.sample_video_frames(video, fps_target=3.0, max_frames=90)
    assert len(frames) == 3  # 9 frame @9fps, lấy 3fps
    stamps = [f["timestamp_sec"] for f in frames]
    assert stamps == sorted(stamps) and stamps[0] >= 0
    assert meta["truncated"] is False
    assert meta["frames_total"] == 9


def test_sample_video_frames_truncated_flag(tmp_path):
    video = str(tmp_path / "clip.mp4")
    _dummy_video(video, frames=9, fps=9.0)
    frames, meta = ingest.sample_video_frames(video, fps_target=9.0, max_frames=2)
    assert len(frames) == 2
    assert meta["truncated"] is True  # còn đoạn chưa xử lý — không báo done giả


def test_detect_faces_bbox_maps_to_original():
    big = np.zeros((800, 1200, 3), dtype=np.uint8)
    emb = _mock_embedder([MockFace()])
    with patch("src.preprocessing.pipeline.preprocess_video_frame",
               side_effect=lambda f: (f, {"conditions": ["night"]})):
        faces, conditions, original = ingest.detect_faces_in_frame(
            emb, big, preprocess_on=True)
    assert conditions == ["night"]
    assert original.shape == big.shape
    # bbox detect trên ảnh thu nhỏ đã quy về frame gốc
    x1, y1, x2, y2 = faces[0]["bbox_orig"]
    assert (x2 - x1) > (60 - 10)  # scale 960/1200 phóng lại
    assert faces[0]["embedding"] is not None


def test_detect_faces_below_threshold_dropped():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    emb = _mock_embedder([MockFace(det=0.1)])
    faces, _, _ = ingest.detect_faces_in_frame(emb, frame, preprocess_on=False)
    assert faces == []


def test_video_verify_helpers_still_importable():
    assert cosine_sim(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert _crop_bbox(frame, [0, 0, 10, 10]).shape == (10, 10, 3)
    _, scale = _downscale_for_detect(frame)
    assert scale == 1.0


def test_search_endpoints_require_db():
    assert client.post(
        "/api/search-sources/images", files={"files": ("a.png", b"x", "image/png")}
    ).status_code == 503
    assert client.get("/api/search-sources").status_code == 503
    assert client.get("/api/ingestion-runs/r1").status_code == 503
    assert client.get("/api/search-runs/r1").status_code == 503
    assert client.post(
        "/api/search-runs/confirm", json={"result_id": "x", "confirmed": True}
    ).status_code == 503
    assert client.post("/api/gallery/rebuild").status_code in (400, 503)
    assert client.post("/api/cameras", json={"name": "c"}).status_code == 503


def test_video_verify_needs_exactly_one_input(tmp_path, monkeypatch):
    from backend.api.session_store import JobState, jobs

    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs/jobs/jx", exist_ok=True)
    cv2.imwrite(os.path.join("outputs", "jobs", "jx", "age_30.png"),
                np.zeros((16, 16, 3), dtype=np.uint8))
    jobs.clear()
    jobs["jx"] = JobState(job_id="jx", session_id="s", status="done",
                          stage="complete",
                          result={"edited_images": {"30": "/outputs/jobs/jx/age_30.png"}})
    try:
        resp = client.post(
            "/api/jobs/jx/video-verify",
            files={"file": ("c.mp4", b"bytes", "video/mp4")},
            data={"source_id": "src-1"},
        )
        assert resp.status_code == 400
        resp = client.post("/api/jobs/jx/video-verify")
        assert resp.status_code in (400, 422)
    finally:
        jobs.clear()


def test_video_verify_source_id_without_db_503(tmp_path, monkeypatch):
    from backend.api.session_store import JobState, jobs

    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs/jobs/jy", exist_ok=True)
    cv2.imwrite(os.path.join("outputs", "jobs", "jy", "age_30.png"),
                np.zeros((16, 16, 3), dtype=np.uint8))
    jobs.clear()
    jobs["jy"] = JobState(job_id="jy", session_id="s", status="done",
                          stage="complete",
                          result={"edited_images": {"30": "/outputs/jobs/jy/age_30.png"}})
    try:
        resp = client.post("/api/jobs/jy/video-verify", data={"source_id": "src-9"})
        assert resp.status_code == 503
    finally:
        jobs.clear()


def test_camera_url_redacted_and_secret_ref():
    assert resolve_camera_target(None) == "0"
    assert redact_url("rtsp://admin:secret123@192.168.1.10/stream") == \
        "rtsp://***@192.168.1.10/stream"
    try:
        resolve_camera_target("CAMERA_KHONG_TON_TAI_XYZ")
        raise AssertionError("secret thiếu phải báo lỗi")
    except ValueError:
        pass
    os.environ["CAMERA_TEST_URL"] = "rtsp://u:p@host/s"
    try:
        assert resolve_camera_target("CAMERA_TEST_URL") == "rtsp://u:p@host/s"
    finally:
        del os.environ["CAMERA_TEST_URL"]


def test_gallery_guards_without_snapshot():
    assert gallery_service.get_snapshot() is None
    try:
        gallery_service.search_gallery(np.ones(8), k=3)
        raise AssertionError("chưa rebuild phải báo lỗi")
    except ValueError:
        pass
    try:
        gallery_service.rebuild_gallery()
        raise AssertionError("DB down phải báo lỗi kết nối")
    except ConnectionError:
        pass


def test_ingest_image_without_db_returns_none():
    assert ingest.ingest_image_bytes(
        data=b"xx", filename="a.png", content_type="image/png",
        embedder=_mock_embedder([])) is None
    assert ingest.persist_video_source(
        video_bytes=b"xx", ext=".mp4", mime_type="video/mp4") is None


def test_image_ingest_conditions_list_serializes_for_faces_and_no_face():
    for count in (0, 1):
        payload = {'source_id': 'source-test', 'faces_found': count,
                   'crops': [], 'conditions': ['rain', 'glare']}
        with patch('backend.api.routers.search_sources.db_ping', return_value=True), \
             patch('backend.api.routers.search_sources.get_ingest_embedder'), \
             patch('backend.api.routers.search_sources.ingest_image_bytes', return_value=payload):
            response = client.post('/api/search-sources/images',
                                   files={'files': ('sample.png', b'image', 'image/png')})
        assert response.status_code == 200
        item = response.json()['items'][0]
        assert item['status'] == ('done' if count else 'no_face')
        assert item['conditions'] == {'rain': 1, 'glare': 1}


def test_reference_crop_embedding_is_allowed_for_query(tmp_path):
    from contextlib import contextmanager
    from backend.api.storage import LocalMediaStorage
    storage = LocalMediaStorage(tmp_path)
    storage.put_bytes(b'image fixture', 'reference/crops/sample.jpg', mime_type='image/jpeg')
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.side_effect = [
        ('reference/crops/sample.jpg', 'reference'), None]
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    with patch('backend.api.database.get_pool', return_value=pool), \
         patch('backend.api.persistence.db_ping', return_value=True), \
         patch('backend.api.storage.get_storage', return_value=storage), \
         patch('backend.api.dependencies.get_embedder') as model, \
         patch('backend.api.repositories.create_embedding', return_value='reference-embedding') as create:
        model.return_value.embed.return_value = np.array([3.0, 4.0])
        assert gallery_service.ensure_crop_embedding('reference-crop') == 'reference-embedding'
    assert create.call_args.kwargs['crop_id'] == 'reference-crop'
    assert np.allclose(create.call_args.kwargs['values'], [0.6, 0.8])
