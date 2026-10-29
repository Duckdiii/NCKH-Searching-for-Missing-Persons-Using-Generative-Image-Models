"""Admission GPU chung + auto-link camera (không cần DB live/GPU thật)."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from backend.api import cameras as cam  # noqa: E402
from backend.api import gpu_admission as adm  # noqa: E402


def _emb():
    v = np.zeros(8, dtype=np.float64)
    v[0] = 1.0
    return v


def test_admission_allows_when_idle():
    with patch.object(adm, "diffusion_busy", return_value=False), \
         patch.object(adm, "camera_inference_load",
                      return_value={"sessions": 0, "queue_bytes": 0, "active_tracks": 0}), \
         patch.object(adm, "vram_info",
                      return_value={"free_mb": 8000.0, "total_mb": 12000.0, "source": "t"}):
        d = adm.check_diffusion_admission()
        assert d == {"allowed": True, "reason": "ok", "load": d["load"], "vram": d["vram"]}


def test_admission_rejects_diffusion_busy():
    with patch.object(adm, "diffusion_busy", return_value=True):
        d = adm.check_diffusion_admission()
        assert d["allowed"] is False and d["reason"] == "diffusion_busy"


def test_admission_camera_priority_on_low_vram():
    load = {"sessions": 4, "queue_bytes": 10**6, "active_tracks": 6}
    vram = {"free_mb": 500.0, "total_mb": 8000.0, "source": "t"}
    with patch.object(adm, "diffusion_busy", return_value=False), \
         patch.object(adm, "camera_inference_load", return_value=load), \
         patch.object(adm, "vram_info", return_value=vram):
        d = adm.check_diffusion_admission()
        assert d["allowed"] is False and d["reason"] == "camera_priority_low_vram"
    # VRAM không đo được → cho qua, không crash.
    with patch.object(adm, "diffusion_busy", return_value=False), \
         patch.object(adm, "camera_inference_load", return_value=load), \
         patch.object(adm, "vram_info", return_value={"unknown": True}):
        assert adm.check_diffusion_admission()["allowed"] is True


def test_acquire_preserves_nonblocking_default():
    import backend.api.job_runner as jr
    assert jr.PIPELINE_LOCK.acquire(blocking=False) is True
    try:
        ok, info = adm.acquire_diffusion_slot()
        assert ok is False and info["reason"] == "diffusion_busy"
    finally:
        jr.PIPELINE_LOCK.release()


def test_admission_status_shape():
    s = adm.status()
    assert set(s) >= {"diffusion_busy", "camera", "vram", "queue", "thresholds"}


def test_auto_link_assign_counts_stat():
    session = {"source_id": "s", "camera_id": "c",
               "linked": 0, "link_unresolved": 0, "link_conflicts": 0, "link_errors": 0}
    dets = [{"crop_id": "crop1", "quality": 0.9, "embedding": _emb()}]
    fake_pool = MagicMock()
    with patch.dict(os.environ, {"CAMERA_AUTO_LINK": "true"}), \
         patch("backend.api.database.get_pool", return_value=fake_pool), \
         patch("backend.api.identity_link.link_tracklet",
               return_value={"action": "assign", "global_id": "g1"}):
        cam._auto_link_tracklet(session, "trk-db-id", dets)
    assert session["linked"] == 1


def test_auto_link_unresolved_and_error_counted():
    session = {"linked": 0, "link_unresolved": 0, "link_conflicts": 0,
               "link_errors": 0, "camera_id": "c"}
    dets = [{"crop_id": "crop1", "quality": 0.9, "embedding": _emb()}]
    fake_pool = MagicMock()
    with patch.dict(os.environ, {"CAMERA_AUTO_LINK": "true"}), \
         patch("backend.api.database.get_pool", return_value=fake_pool), \
         patch("backend.api.identity_link.link_tracklet",
               return_value={"action": "conflict"}):
        cam._auto_link_tracklet(session, "t", dets)
    assert session["link_conflicts"] == 1
    with patch.dict(os.environ, {"CAMERA_AUTO_LINK": "true"}), \
         patch("backend.api.identity_link.link_tracklet",
               side_effect=ConnectionError("db down")):
        cam._auto_link_tracklet(session, "t", dets)
    assert session["link_errors"] == 1


def test_auto_link_disabled_skips_db():
    session = {"linked": 0, "link_unresolved": 0, "link_conflicts": 0,
               "link_errors": 0}
    with patch.dict(os.environ, {"CAMERA_AUTO_LINK": "false"}), \
         patch("backend.api.identity_link.link_tracklet") as link_mock:
        cam._auto_link_tracklet(
            session, "t", [{"crop_id": "c", "quality": 0.9, "embedding": _emb()}])
    link_mock.assert_not_called()
    assert session["linked"] == 0


def test_auto_link_no_embedding_counts_unresolved():
    session = {"linked": 0, "link_unresolved": 0, "link_conflicts": 0,
               "link_errors": 0, "camera_id": "c"}
    with patch.dict(os.environ, {"CAMERA_AUTO_LINK": "true"}):
        cam._auto_link_tracklet(session, "t", [{"crop_id": "c"}])
    assert session["link_unresolved"] == 1
