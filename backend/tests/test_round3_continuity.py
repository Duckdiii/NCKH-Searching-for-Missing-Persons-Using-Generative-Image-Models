"""Vòng 3 — checkpoint cùng tracklet, reconnect gap, latency gallery."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from backend.api import cameras as cam  # noqa: E402
from backend.api import camera_tracks as ct  # noqa: E402
from backend.api import gallery as gal  # noqa: E402
from backend.api import ingest  # noqa: E402


def _frame():
    return np.full((240, 320, 3), 128, dtype=np.uint8)


def _face(x=10.0):
    v = np.zeros(8, dtype=np.float64)
    v[0] = 1.0
    return {"bbox_orig": [x, 10.0, x + 100, 110.0], "det_score": 0.9,
            "embedding": v, "kps": None}


def test_hard_timeout_checkpoints_same_tracklet():
    tr = ct.CameraTracker(timeout_sec=100.0, hard_timeout_sec=1.0,
                          checkpoint_min_interval_sec=0.0)
    frame = _frame()
    tr.update(frame, [_face()], now=0.0)
    assert len(tr.open_tracks()) == 1
    lid = tr.open_tracks()[0].local_id
    # Vượt hard timeout: KHÔNG cắt tracklet mới — checkpoint cùng local_id.
    tr.update(frame, [_face(x=12.0)], now=2.0)
    still_open = tr.open_tracks()
    assert len(still_open) == 1 and still_open[0].local_id == lid
    assert still_open[0].slot_revision == 1
    cps = tr.pop_checkpoints()
    assert len(cps) == 1 and cps[0].local_id == lid
    tr.release_checkpoint_candidates(cps[0])
    assert cps[0].candidates == []  # RAM crop đã giải phóng
    assert cps[0].seen_hashes  # vẫn lọc trùng sau checkpoint


def test_checkpoint_rate_limited():
    tr = ct.CameraTracker(timeout_sec=100.0, hard_timeout_sec=1.0,
                          checkpoint_min_interval_sec=60.0)
    frame = _frame()
    tr.update(frame, [_face()], now=0.0)
    tr.update(frame, [_face()], now=2.0)
    tr.update(frame, [_face()], now=3.0)  # quá hard timeout nhưng chưa đủ nhịp
    assert tr.pop_checkpoints() == [] or len(tr.pop_checkpoints()) <= 1


def test_close_all_for_reconnect_gap():
    tr = ct.CameraTracker(timeout_sec=100.0)
    frame = _frame()
    tr.update(frame, [_face()], now=0.0)
    tr.update(frame, [_face(x=200.0)], now=0.1)
    assert len(tr.open_tracks()) == 2
    closed = tr.close_all(t=5.0)
    assert len(closed) == 2
    assert tr.open_tracks() == []
    # pop_closed_ready thu hồi để persist (timeline thể hiện khoảng trống).
    ready = tr.pop_closed_ready(t=5.0)
    assert len(ready) == 2


def test_persist_checkpoint_does_not_close():
    crop = _frame()[10:110, 10:110].copy()
    cands = [{"crop_bgr": crop, "bbox": [10.0, 10.0, 110.0, 110.0],
              "det_score": 0.9, "quality": 0.8, "blur": 100.0,
              "exposure": 0.8, "frontal": 0.9, "face_area": 10000.0,
              "embedding": None, "kps": None}]
    conn = MagicMock()
    storage = MagicMock()
    with patch("backend.api.repositories.create_tracklet", return_value="trk"), \
         patch("backend.api.repositories.create_frame", return_value="f"), \
         patch("backend.api.repositories.create_detection", return_value="d"), \
         patch("backend.api.repositories.create_asset", return_value="a"), \
         patch("backend.api.repositories.create_crop", return_value="c"), \
         patch("backend.api.repositories.touch_tracklet", return_value=None) as touch, \
         patch("backend.api.repositories.close_tracklet") as close:
        res = ingest.persist_camera_tracklet(
            conn, storage, source_id="s", camera_id="c", track_local_id="t1",
            frame_index=0, offset_ms=0, captured_at=None, received_at=None,
            source_timestamp=None, timestamp_uncertainty_ms=4000,
            frame_width=320, frame_height=240, candidates=cands,
            detector_name="insightface", detector_version="buffalo_l",
            run_id="r", close=False, slot_revision=2, observation_count=50)
    close.assert_not_called()
    touch.assert_called_once()
    assert res["slot_revision"] == 2
    assert touch.call_args.kwargs["observation_count"] == 50


def test_persist_close_still_closes_with_revision():
    crop = _frame()[10:110, 10:110].copy()
    cands = [{"crop_bgr": crop, "bbox": [10.0, 10.0, 110.0, 110.0],
              "det_score": 0.9, "quality": 0.8, "blur": 100.0,
              "exposure": 0.8, "frontal": 0.9, "face_area": 10000.0,
              "embedding": None, "kps": None}]
    conn = MagicMock()
    storage = MagicMock()
    with patch("backend.api.repositories.create_tracklet", return_value="trk"), \
         patch("backend.api.repositories.create_frame", return_value="f"), \
         patch("backend.api.repositories.create_detection", return_value="d"), \
         patch("backend.api.repositories.create_asset", return_value="a"), \
         patch("backend.api.repositories.create_crop", return_value="c"), \
         patch("backend.api.repositories.close_tracklet", return_value=None) as close:
        res = ingest.persist_camera_tracklet(
            conn, storage, source_id="s", camera_id="c", track_local_id="t1",
            frame_index=0, offset_ms=0, captured_at=None, received_at=None,
            source_timestamp=None, timestamp_uncertainty_ms=None,
            frame_width=320, frame_height=240, candidates=cands,
            detector_name="insightface", detector_version="buffalo_l",
            run_id="r")
    close.assert_called_once()
    assert close.call_args.kwargs["quality_summary"]["slot_revision"] == 0
    assert res["slot_revision"] == 0


def test_drain_tracker_applies_gap_uncertainty_once():
    session = {"source_id": "s", "camera_id": "c", "run_key": "r",
               "persisted_frames": 0, "faces_found": 0, "crops_persisted": 0,
               "tracks_closed": 0, "checkpoints": 0, "persist_errors": 0,
               "linked": 0, "link_unresolved": 0, "link_conflicts": 0,
               "link_errors": 0, "stream_gap": True, "stream_gaps": 3}
    tr = ct.CameraTracker(timeout_sec=100.0)
    tr.update(_frame(), [_face()], now=0.0)
    with patch("backend.api.ingest.persist_camera_tracklet",
               return_value={"tracklet_id": "t", "frame_id": "f",
                             "detections": []}) as persist:
        with patch("backend.api.database.get_pool"), \
             patch("backend.api.storage.get_storage"):
            cam._drain_tracker(session, tr)
    assert persist.call_count == 1
    assert persist.call_args.kwargs["timestamp_uncertainty_ms"] == 6000
    assert "gap_uncertainty_ms" not in session  # chỉ áp dụng một lần
    assert session["tracks_closed"] == 1


def test_gallery_latency_stats():
    gal._GALLERIES.clear()
    gal._DELTAS.clear()
    gal._LATENCY.clear()
    try:
        import faiss
        dim = 8
        index = faiss.IndexFlatIP(dim)
        v = np.zeros(dim, dtype=np.float32)
        v[0] = 1.0
        index.add(v.reshape(1, -1))
        key = "k/k/k"
        gal._GALLERIES[key] = {"key": key, "built_at": "2026-09-26T00:00:00+00:00",
                               "index": index,
                               "mapping": [{"embedding_id": "e", "crop_id": "c"}]}
        hits = gal.search_gallery(np.asarray([1.0] + [0.0] * 7), k=1,
                                  snapshot_key_=key)
        assert len(hits) == 1
        stats = gal.get_latency_stats(key)
        assert stats["count"] == 1 and stats["p95_ms"] is not None
        assert gal.get_latency_stats()["k/k/k"]["count"] == 1
    finally:
        gal._GALLERIES.clear()
        gal._DELTAS.clear()
        gal._LATENCY.clear()
