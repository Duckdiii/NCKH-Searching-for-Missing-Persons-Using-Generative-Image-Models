import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routers.video_verify import (
    cosine_sim,
    _crop_bbox,
    _downscale_for_detect,
    _resolve_edited_paths,
)
from backend.api.session_store import JobState, jobs

client = TestClient(app)


def test_cosine_sim_identical_is_one():
    a = np.array([1.0, 0.0, 0.0])
    a = a / np.linalg.norm(a)
    assert cosine_sim(a, a) == pytest.approx(1.0)


def test_crop_bbox_clamps_out_of_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = _crop_bbox(frame, [-10, -10, 200, 200])
    assert crop.shape == (100, 100, 3)


def test_resolve_edited_paths_from_url(tmp_path, monkeypatch):
    job_dir = tmp_path / "outputs" / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    img = job_dir / "age_30.png"
    img.write_bytes(b"fake")
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_edited_paths("j1", {30: "/outputs/jobs/j1/age_30.png"})
    assert resolved[30].endswith("age_30.png")


def test_downscale_for_detect_small_kept():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out, scale = _downscale_for_detect(frame)
    assert scale == 1.0
    assert out.shape == frame.shape


def test_downscale_for_detect_large_scaled():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out, scale = _downscale_for_detect(frame)
    assert scale == pytest.approx(960 / 1920)
    assert max(out.shape[:2]) == 960


def test_get_video_embedder_falls_back_to_cpu(monkeypatch):
    import backend.api.dependencies as dep
    dep._VIDEO_EMBEDDER_INSTANCE = None
    dep._VIDEO_EMBEDDER_CTX = None
    real_cls = dep.FaceEmbedder
    calls = {"n": 0}

    def fake_cls(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("no CUDA")
        return real_cls(*a, **k)

    monkeypatch.setattr(dep, "FaceEmbedder", fake_cls)
    try:
        assert dep.get_video_embedder() is dep.get_embedder()
    finally:
        dep._VIDEO_EMBEDDER_INSTANCE = None
        dep._VIDEO_EMBEDDER_CTX = None


class MockFace:
    def __init__(self):
        self.bbox = np.array([10.0, 10.0, 60.0, 60.0])
        self.det_score = 0.99
        v = np.zeros(512, dtype=np.float32)
        v[0] = 1.0
        self.normed_embedding = v


def _make_dummy_video(path: str, frames: int = 5):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 5.0, (128, 128))
    for i in range(frames):
        out.write(np.full((128, 128, 3), 200 - i, dtype=np.uint8))
    out.release()


def test_video_verify_best_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs/jobs/job_vid", exist_ok=True)
    ref_path = os.path.join("outputs", "jobs", "job_vid", "age_30.png")
    cv2.imwrite(ref_path, np.zeros((64, 64, 3), dtype=np.uint8))
    video_path = os.path.join("outputs", "jobs", "job_vid", "src.mp4")
    _make_dummy_video(video_path)

    jobs.clear()
    jobs["job_vid"] = JobState(
        job_id="job_vid",
        session_id="s1",
        status="done",
        stage="complete",
        result={"edited_images": {"30": "/outputs/jobs/job_vid/age_30.png"}},
    )

    ref_emb = np.zeros(512, dtype=np.float32)
    ref_emb[0] = 1.0
    with patch("backend.api.routers.video_verify.get_video_embedder") as mock_get:
        emb = MagicMock()
        emb.embed.return_value = ref_emb
        emb.detect_faces.return_value = [MockFace()]
        mock_get.return_value = emb
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        resp = client.post(
            "/api/jobs/job_vid/video-verify",
            files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["faces_found"] >= 1
    assert data["best_match"] is not None
    assert data["best_match"]["score"] == pytest.approx(1.0, abs=1e-3)
    assert data["best_match"]["best_age"] == 30
    jobs.clear()


def test_video_verify_disk_fallback_without_memory_job(tmp_path, monkeypatch):
    """Backend restart làm mất store in-memory nhưng age_*.png còn trên đĩa."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs/jobs/job_disk", exist_ok=True)
    ref_path = os.path.join("outputs", "jobs", "job_disk", "age_30.png")
    cv2.imwrite(ref_path, np.zeros((64, 64, 3), dtype=np.uint8))
    video_path = os.path.join("outputs", "jobs", "job_disk", "src.mp4")
    _make_dummy_video(video_path)

    jobs.clear()  # không có job nào trong RAM

    ref_emb = np.zeros(512, dtype=np.float32)
    ref_emb[0] = 1.0
    with patch("backend.api.routers.video_verify.get_video_embedder") as mock_get:
        emb = MagicMock()
        emb.embed.return_value = ref_emb
        emb.detect_faces.return_value = [MockFace()]
        mock_get.return_value = emb
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        resp = client.post(
            "/api/jobs/job_disk/video-verify",
            files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        )

    assert resp.status_code == 200, resp.text
    assert resp.json()["best_match"] is not None
    jobs.clear()


def test_video_verify_unknown_job_404(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs.clear()
    resp = client.post(
        "/api/jobs/job_khong_ton_tai/video-verify",
        files={"file": ("clip.mp4", b"fake-bytes", "video/mp4")},
    )
    assert resp.status_code == 404
    jobs.clear()
