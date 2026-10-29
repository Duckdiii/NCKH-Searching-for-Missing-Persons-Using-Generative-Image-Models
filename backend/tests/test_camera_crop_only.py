"""P0/P1 — Test camera crop-only + tracker/selector (không cần DB live)."""
import os
import queue
import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from backend.api import cameras as cam  # noqa: E402
from backend.api import camera_tracks as ct  # noqa: E402
from backend.api import ingest  # noqa: E402
from database.migrate import discover_migrations  # noqa: E402


def _frame(h=240, w=320, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _face(bbox=(10.0, 10.0, 110.0, 110.0), det=0.9, emb=None):
    v = np.zeros(8, dtype=np.float64)
    v[0] = 1.0
    return {"bbox_orig": list(bbox), "det_score": det,
            "embedding": v if emb is None else emb, "kps": None}


def test_migration_005_discovered():
    assert "005" in dict(discover_migrations())


def test_queue_maxsize_bounded():
    assert cam.QUEUE_MAXSIZE <= 2
    assert cam.QUEUE_MAX_BYTES <= 32 * 1024 * 1024


def test_queue_put_newest_drops_oldest_and_caps_bytes():
    session = {"queue": queue.Queue(maxsize=2), "queue_bytes": 0, "dropped": 0}
    for _ in range(4):
        cam._queue_put_newest(session, {"frame_bgr": _frame(), "frame_index": 0})
    assert session["queue"].qsize() <= 2
    assert session["dropped"] >= 2
    # Tổng byte bounded.
    assert session["queue_bytes"] <= cam.QUEUE_MAX_BYTES


def test_copy_crop_is_copy_not_view():
    frame = _frame()
    crop = ct.copy_crop(frame, [10.0, 10.0, 50.0, 50.0])
    assert crop.base is None or not isinstance(crop.base, np.ndarray) or crop.base is not frame
    crop[:] = 0
    assert frame[20, 20, 0] != 0


def test_encode_crop_jpeg_downscales_and_provenance():
    big = np.full((512, 400, 3), 150, dtype=np.uint8)
    data, prov = ct.encode_crop_jpeg(big, quality=85)
    assert data[:2] == b"\xff\xd8"
    assert max(prov["size_dst"]) <= 256
    assert prov["codec"] == "jpeg" and prov["quality"] == 85
    small = np.full((60, 50, 3), 150, dtype=np.uint8)
    _, prov2 = ct.encode_crop_jpeg(small)
    assert prov2["resize"] == "none"  # không upscale mặt nhỏ


def test_tracker_keeps_one_track_and_max_3_candidates():
    tr = ct.CameraTracker(timeout_sec=10.0, hard_timeout_sec=1000.0)
    frame = _frame()
    for i in range(30):
        # Cùng người dịch chuyển nhẹ → cùng track, nhiều frame.
        dx = float(i)
        tr.update(frame, [_face(bbox=(10 + dx, 10.0, 110 + dx, 110.0))], now=float(i) * 0.2)
    open_tracks = tr.open_tracks()
    assert len(open_tracks) == 1
    assert len(open_tracks[0].candidates) <= 3
    assert open_tracks[0].observation_count == 30


def test_tracker_timeout_closes_and_quota_caps():
    tr = ct.CameraTracker(timeout_sec=0.5, max_open=2, max_tracklets_per_day=3)
    frame = _frame()
    tr.update(frame, [_face(bbox=(10, 10, 60, 60))], now=0.0)
    tr.update(frame, [_face(bbox=(200, 200, 250, 250))], now=0.1)
    tr.update(frame, [_face(bbox=(10, 10, 60, 60))], now=0.2)
    tr.update(frame, [_face(bbox=(10, 10, 60, 60))], now=0.3)
    # Quota 3/ngày: track thứ 4 bị chặn.
    assert tr._created_today <= 3
    ready = tr.pop_closed_ready(t=10.0)
    assert isinstance(ready, list)


def test_persist_camera_tracklet_crop_only_no_frame_file():
    frame = _frame(240, 320)
    crop = frame[10:110, 10:110].copy()
    cands = [{"crop_bgr": crop, "bbox": [10.0, 10.0, 110.0, 110.0],
              "det_score": 0.9, "quality": 0.8, "blur": 100.0,
              "exposure": 0.8, "frontal": 0.9, "face_area": 10000.0,
              "embedding": (np.zeros(8, dtype=np.float64) + [1, 0, 0, 0, 0, 0, 0, 0]),
              "kps": None}]
    conn = MagicMock()
    storage = MagicMock()
    with patch("backend.api.repositories.create_tracklet", return_value="trk"), \
         patch("backend.api.repositories.create_frame", return_value="frm") as mk_frame, \
         patch("backend.api.repositories.create_detection", return_value="det"), \
         patch("backend.api.repositories.create_asset", return_value="asset"), \
         patch("backend.api.repositories.create_crop", return_value="crop"), \
         patch("backend.api.repositories.create_embedding", return_value="emb"), \
         patch("backend.api.repositories.close_tracklet", return_value=None), \
         patch("backend.api.gallery.current_embed_triple",
               return_value=("insightface", "buffalo_l", "pre")):
        res = ingest.persist_camera_tracklet(
            conn, storage, source_id="src", camera_id="cam1",
            track_local_id="t1", frame_index=0, offset_ms=0,
            captured_at=None, received_at=None, source_timestamp=None,
            timestamp_uncertainty_ms=None, frame_width=320, frame_height=240,
            candidates=cands, detector_name="insightface",
            detector_version="buffalo_l", run_id="run")
    assert res["frame_available"] is False and res["frame_url"] is None
    assert len(res["detections"]) == 1
    # Frame metadata-only: asset_id None.
    assert mk_frame.call_args.kwargs["asset_id"] is None
    # Chỉ ghi crop, không ghi frame file.
    written = [c.kwargs.get("mime_type", "") for c in storage.put_bytes.call_args_list]
    assert written and all(m == "image/jpeg" for m in written)
    keys = [c.args[1] for c in storage.put_bytes.call_args_list]
    assert all(k.startswith("search/crops/") for k in keys)
    assert not any("search/frames" in k for k in keys)


def test_start_capture_crop_only_no_recorder(monkeypatch):
    pool = MagicMock()
    conn = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    monkeypatch.setenv("CAMERA_SAVE_SESSION_VIDEO", "true")
    with patch("backend.api.database.get_pool", return_value=pool), \
         patch("backend.api.repositories.get_camera",
               return_value={"camera_id": "c1", "name": "Gate"}), \
         patch("backend.api.cameras._camera_secret_ref", return_value=None), \
         patch("backend.api.ingest.get_ingest_embedder", return_value=MagicMock(model_name="buffalo_l")), \
         patch("backend.api.repositories.create_source", return_value="src") as mk_src, \
         patch("backend.api.repositories.create_ingestion_run", return_value="run"), \
         patch("backend.api.repositories.update_ingestion_run"), \
         patch("threading.Thread") as thr:
        thr.return_value = MagicMock()
        out = cam.start_capture(camera_id="c1")
    assert out["source_id"]
    assert mk_src.call_args.kwargs.get("storage_policy") == "crop_only"
    with cam._SESSIONS_LOCK:
        sess = cam._SESSIONS.pop(out["source_id"], None)
    assert sess is not None
    assert sess["recorder"] is None and sess["record_path"] is None
    assert sess["queue"].maxsize <= 2
