"""Test nối K (crop/tracklet) và q_min vào tracker + phiên camera (pipeline_optimizer)."""
import os
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))

from backend.api import camera_tracks as ct  # noqa: E402
from backend.api import cameras as cam  # noqa: E402
from backend.api.routers import cameras as cam_router  # noqa: E402
from backend.api.schemas import CaptureStartRequest  # noqa: E402

TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]])
CORRIDOR = {"motion": "approaching", "image_width_px": 3840, "hfov_deg": 60, "far_m": 15,
            "people_per_hour": 120}


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random((240, 320, 3)) * 255).astype(np.uint8), (3, 3), 0)


def _face(emb, x0=100.0):
    return {"bbox_orig": [x0, 60.0, x0 + 112, 172.0], "det_score": 0.9, "embedding": emb,
            "kps": TEMPLATE + np.array([x0, 60.0]), "embedding_norm": 22.0}


def _distinct_embeddings(n, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = rng.normal(size=512)
        out.append(v / np.linalg.norm(v))
    return out


# ---------------------------------------------------------------- tracker ----

@pytest.mark.parametrize("k", [1, 2, 5])
def test_tracker_keeps_at_most_k_candidates(k):
    tracker = ct.CameraTracker(max_candidates=k)
    # Mỗi khung: frame khác + embedding khác hẳn (góc mới) → muốn giữ thêm ứng viên.
    for i, emb in enumerate(_distinct_embeddings(8)):
        tracker.update(_frame(i), [_face(emb)], now=i * 0.1)
    tr = tracker.open_tracks()[0]
    assert len(tr.candidates) == k


def test_tracker_min_quality_rejects_and_counts():
    strict = ct.CameraTracker(min_quality=0.99)
    loose = ct.CameraTracker(min_quality=0.0)
    emb = _distinct_embeddings(1)[0]
    for i in range(3):
        strict.update(_frame(i), [_face(emb)], now=i * 0.1)
        loose.update(_frame(i), [_face(emb)], now=i * 0.1)
    assert strict.open_tracks()[0].candidates == []
    assert strict.rejected_low_quality_total == 3
    assert loose.open_tracks()[0].candidates and loose.rejected_low_quality_total == 0


@pytest.mark.parametrize("kwargs", [{"max_candidates": 0}, {"max_candidates": 6},
                                    {"min_quality": -0.1}, {"min_quality": 1.0}])
def test_tracker_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ct.CameraTracker(**kwargs)


# ---------------------------------------------------------- chọn cấu hình ----

def test_config_defaults_without_geometry():
    cfg = cam_router.resolve_capture_config(CaptureStartRequest())
    assert (cfg["sample_fps"], cfg["max_candidates"], cfg["min_quality"]) == (1.0, 3, 0.0)
    assert cfg["config_source"] == "manual" and cfg["plan"] is None


def test_config_geometry_uses_sampling_plan_only():
    cfg = cam_router.resolve_capture_config(CaptureStartRequest(geometry=CORRIDOR))
    assert cfg["config_source"] == "sampling_plan"
    assert cfg["sample_fps"] == cfg["plan"]["fps"]
    assert cfg["max_candidates"] == ct.MAX_CANDIDATES_PER_TRACKLET and cfg["optimizer"] is None


def test_config_auto_optimize_applies_recommendation():
    cfg = cam_router.resolve_capture_config(
        CaptureStartRequest(geometry=CORRIDOR, auto_optimize=True))
    rec = cfg["optimizer"]["recommended"]
    assert cfg["config_source"] == "optimizer"
    assert cfg["sample_fps"] == rec["fps"]
    assert cfg["max_candidates"] == rec["k"] and cfg["min_quality"] == rec["q_min"]


def test_config_manual_values_override_optimizer():
    cfg = cam_router.resolve_capture_config(CaptureStartRequest(
        geometry=CORRIDOR, auto_optimize=True, sample_fps=2.0, max_candidates=4,
        min_quality=0.2))
    assert (cfg["sample_fps"], cfg["max_candidates"], cfg["min_quality"]) == (2.0, 4, 0.2)
    assert cfg["config_source"] == "optimizer+manual"


def test_config_fragmented_recommendation_falls_back_to_default_k():
    cfg = cam_router.resolve_capture_config(CaptureStartRequest(
        geometry={"distance_m": 3.0, "people_per_hour": 120}, auto_optimize=True))
    assert cfg["optimizer"]["recommended"]["k"] is None
    assert cfg["max_candidates"] == ct.MAX_CANDIDATES_PER_TRACKLET


# ------------------------------------------------------------- API + phiên ----

def test_start_route_passes_optimized_config():
    from fastapi.testclient import TestClient
    from backend.api.main import app
    with patch.object(cam_router, "_require_db"), \
            patch.object(cam_router.cam_service, "start_capture",
                         return_value={"source_id": "s1", "run_id": "r1"}) as start:
        r = TestClient(app).post("/api/cameras/c1/start",
                                 json={"geometry": CORRIDOR, "auto_optimize": True})
    assert r.status_code == 202, r.text
    kw = start.call_args.kwargs
    body = r.json()
    assert kw["max_candidates"] == body["max_candidates"]
    assert kw["min_quality"] == body["min_quality"]
    assert kw["sample_fps"] == body["sample_fps"] and body["config_source"] == "optimizer"
    assert body["optimizer"]["recommended"]["fps"] == kw["sample_fps"]


def test_capture_status_reports_config_and_rejections():
    tracker = ct.CameraTracker(max_candidates=2, min_quality=0.4)
    tracker.rejected_low_quality_total = 5
    session = {"camera_id": "c1", "run_id": "r1", "frames_sampled": 10, "faces_found": 4,
               "dropped": 0, "reconnects": 0, "sample_fps": 3.0, "max_candidates": 2,
               "min_quality": 0.4, "config_source": "optimizer", "tracker": tracker}
    with patch.dict(cam._SESSIONS, {"s1": session}, clear=False):
        st = cam.capture_status("s1")
    assert (st["sample_fps"], st["max_candidates"], st["min_quality"]) == (3.0, 2, 0.4)
    assert st["config_source"] == "optimizer" and st["rejected_low_quality"] == 5
