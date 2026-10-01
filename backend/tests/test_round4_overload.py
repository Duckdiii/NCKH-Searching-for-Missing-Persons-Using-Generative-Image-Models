"""Vòng 4 — spool đĩa, adaptive fps, thumbnail cache."""
import io
import os
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from backend.api import cameras as cam  # noqa: E402
from backend.api import spool  # noqa: E402
from backend.api import thumbs  # noqa: E402


def _crop():
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (120, 100, 3)).astype(np.uint8)


def _cands(n=2):
    v = np.zeros(8, dtype=np.float64)
    v[0] = 1.0
    return [{"crop_bgr": _crop(), "bbox": [0.0, 0.0, 50.0, 50.0],
             "det_score": 0.9 - i * 0.1, "quality": 0.8 - i * 0.1,
             "blur": 100.0, "exposure": 0.8, "frontal": 0.9,
             "face_area": 2500.0, "embedding": v, "kps": None}
            for i in range(n)]


def _meta():
    return {"source_id": "s", "camera_id": "c", "track_local_id": "t1",
            "captured_at": None, "received_at": None,
            "timestamp_uncertainty_ms": None, "frame_width": 320,
            "frame_height": 240, "detector_name": "insightface",
            "detector_version": "buffalo_l", "run_id": "r",
            "close": True, "slot_revision": 0, "observation_count": 2,
            "preprocessing_base": {}}


def test_spool_write_and_redrive(tmp_path, monkeypatch):
    monkeypatch.setattr(spool, "SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setattr(spool, "SPOOL_MAX_BYTES", 50 * 1024 * 1024)
    out = spool.write_spool(cands=_cands(), meta=_meta())
    assert out["spooled"] is True and out["crops"] == 2
    assert spool.spool_usage()["files"] == 3  # 1 json + 2 jpg

    session = {"source_id": "s", "run_key": "r", "persisted_frames": 0,
               "faces_found": 0, "crops_persisted": 0,
               "spooled_redriven": 0, "spool_errors": 0}
    conn = MagicMock()
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    storage = MagicMock()
    with patch("backend.api.database.get_pool", return_value=pool), \
         patch("backend.api.storage.get_storage", return_value=storage), \
         patch("backend.api.gallery.current_embed_triple",
               return_value=("m", "mv", "pv")), \
         patch("backend.api.repositories.create_tracklet", return_value="t"), \
         patch("backend.api.repositories.create_frame", return_value="f"), \
         patch("backend.api.repositories.create_detection", return_value="d"), \
         patch("backend.api.repositories.create_asset", return_value="a"), \
         patch("backend.api.repositories.create_crop", return_value="c"), \
         patch("backend.api.repositories.create_embedding", return_value="e"), \
         patch("backend.api.repositories.close_tracklet", return_value=None), \
         patch("backend.api.gallery.emit_outbox", return_value=True):
        res = spool.redrive(session)
    assert res == {"redriven": 1, "failed": 0, "expired": 0}
    assert session["crops_persisted"] == 2
    assert spool.spool_usage()["files"] == 0  # thành công thì xóa file


def test_spool_full_drops_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(spool, "SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setattr(spool, "SPOOL_MAX_BYTES", 1)  # cap cực nhỏ
    out = spool.write_spool(cands=_cands(), meta=_meta())
    assert out["spooled"] is False and out["reason"] == "spool_full"
    assert out["dropped"] == 2


def test_spool_ttl_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(spool, "SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setattr(spool, "SPOOL_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(spool, "SPOOL_TTL_SEC", -1.0)  # hết hạn ngay
    spool.write_spool(cands=_cands(1), meta=_meta())
    session = {"source_id": "s", "run_key": "r", "persisted_frames": 0,
               "faces_found": 0, "crops_persisted": 0,
               "spooled_redriven": 0, "spool_errors": 0}
    res = spool.redrive(session)
    assert res["expired"] == 1 and res["redriven"] == 0
    assert spool.spool_usage()["files"] == 0


def test_adapt_interval_slows_on_drops_and_recovers():
    s = {"dropped": 0, "_drop_mark": 0, "interval_scale": 1.0}
    assert cam._adapt_interval(s) == 1.0
    assert "degraded" not in s
    s["dropped"] = 9  # 9/30 = 30% rớt
    assert cam._adapt_interval(s) == 1.5
    assert s["degraded"].startswith("overload:")
    s["dropped"] = 9  # 0 rớt mới → hồi dần
    assert cam._adapt_interval(s) == 1.2
    s["dropped"] = 9
    assert cam._adapt_interval(s) == 1.0
    assert "degraded" not in s
    # Không vượt trần 4x.
    s["interval_scale"] = 4.0
    s["dropped"] = 39
    assert cam._adapt_interval(s) == 4.0


def _storage_with(img):
    import cv2
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    storage = MagicMock()
    storage.open.return_value.__enter__.return_value.read.return_value = bytes(buf)
    return storage


def test_thumb_hit_bounded_and_invalidate():
    thumbs._CACHE.clear()
    thumbs._BYTES = 0
    img = _crop()
    storage = _storage_with(img)
    with patch.object(thumbs, "THUMB_CACHE_BYTES", 10**9):
        d1 = thumbs.get_thumb(storage, "c1", "k1", edge=64)
        assert d1[:2] == b"\xff\xd8"
        d2 = thumbs.get_thumb(storage, "c1", "k1", edge=64)
        assert d1 == d2
        assert storage.open.call_count == 1  # lần 2 hit cache
        st = thumbs.stats()
        assert st["entries"] == 1 and st["bytes"] == len(d1)
    thumbs.invalidate("c1")
    assert thumbs.stats()["entries"] == 0
    thumbs._CACHE.clear()
    thumbs._BYTES = 0


def test_thumb_evicts_oldest_when_full():
    thumbs._CACHE.clear()
    thumbs._BYTES = 0
    storage = _storage_with(_crop())
    with patch.object(thumbs, "THUMB_CACHE_BYTES", 3000):
        thumbs.get_thumb(storage, "a", "ka", edge=64)
        thumbs.get_thumb(storage, "b", "kb", edge=64)
        st = thumbs.stats()
        assert st["bytes"] <= 3000
    thumbs._CACHE.clear()
    thumbs._BYTES = 0


def test_thumb_missing_file_raises():
    storage = MagicMock()
    storage.open.side_effect = FileNotFoundError("mất")
    try:
        thumbs.get_thumb(storage, "cx", "kx")
        raise AssertionError("phải raise FileNotFoundError")
    except FileNotFoundError:
        pass
