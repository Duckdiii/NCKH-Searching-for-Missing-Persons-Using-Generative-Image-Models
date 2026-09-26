"""Vòng 2 — guard FADING, exemplar merge/split, timeline evidence, cap đĩa."""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("."))

from backend.api import repositories as repo  # noqa: E402
from backend.api import retention as ret  # noqa: E402


def _cur(conn, fetchone=None, fetchall=None):
    cur = conn.cursor.return_value.__enter__.return_value
    if fetchone is not None:
        cur.fetchone.return_value = fetchone
    if fetchall is not None:
        cur.fetchall.return_value = fetchall
    return cur


def test_exemplar_rejects_generated_embedding():
    conn = MagicMock()
    _cur(conn, fetchone=("crop1", "gen1", "search"))  # generated_image_id set
    try:
        repo.set_exemplar(conn, global_id="g", embedding_space="sp", slot=0,
                          crop_id="crop1", embedding_id="e1")
        raise AssertionError("phải từ chối embedding ảnh tạo sinh")
    except ValueError as exc:
        assert "tạo sinh" in str(exc)


def test_exemplar_rejects_reference_crop():
    conn = MagicMock()
    _cur(conn, fetchone=(None, None, "reference"))
    try:
        repo.set_exemplar(conn, global_id="g", embedding_space="sp", slot=1,
                          embedding_id="e2")
        raise AssertionError("phải từ chối crop reference")
    except ValueError as exc:
        assert "search" in str(exc)


def test_exemplar_accepts_search_crop_embedding():
    conn = MagicMock()
    _cur(conn, fetchone=("crop9", None, "search"))
    repo.set_exemplar(conn, global_id="g", embedding_space="sp", slot=0,
                      crop_id="crop9", embedding_id="e9")
    cur = conn.cursor.return_value.__enter__.return_value
    assert cur.execute.call_count >= 2  # guard SELECT + INSERT


def test_exemplar_crop_only_path_validates_purpose():
    conn = MagicMock()
    _cur(conn, fetchone=("reference",))
    try:
        repo.set_exemplar(conn, global_id="g", embedding_space="sp", slot=0,
                          crop_id="cref")
        raise AssertionError("phải từ chối crop reference")
    except ValueError:
        pass


def test_timeline_has_evidence_fields():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("a1", "t1", 0.8, "2026-09-26T10:00:00+00:00", None,
         "cam1", "2026-09-26T10:00:00+00:00", "2026-09-26T10:01:00+00:00",
         "2026-09-26T10:01:00+00:00", 12),
    ]
    cur.fetchone.return_value = ("crop5", "search/crops/x.jpg", 0.9)
    tl = repo.get_timeline(conn, "g1")
    assert tl["segments"][0]["evidence_crop_id"] == "crop5"
    assert tl["segments"][0]["evidence_crop_key"] == "search/crops/x.jpg"
    assert tl["segments"][0]["evidence_score"] == 0.9


def test_storage_guard_no_cap_ok():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MEDIA_CAP_BYTES", None)
        g = ret.storage_guard()
        assert g["level"] == "ok" and g["reason"] == "no_cap"


def test_storage_guard_warn_and_stop(tmp_path):
    target = tmp_path / "media"
    target.mkdir()
    (target / "a.jpg").write_bytes(b"x" * 85)
    with patch.dict(os.environ, {"MEDIA_CAP_BYTES": "100",
                                 "MEDIA_ROOT": str(target)}), \
         patch.object(ret, "media_used_bytes", return_value=85):
        ret._GUARD_CACHE["at"] = 0.0
        g = ret.storage_guard()
        assert g["level"] == "warn" and g["allowed"] is True
    with patch.dict(os.environ, {"MEDIA_CAP_BYTES": "100",
                                 "MEDIA_ROOT": str(target)}), \
         patch.object(ret, "media_used_bytes", return_value=95):
        ret._GUARD_CACHE["at"] = 0.0
        g = ret.storage_guard()
        assert g["level"] == "stop" and g["allowed"] is False
    ret._GUARD_CACHE["at"] = 0.0


def test_media_used_skips_uploading_tmp(tmp_path):
    (tmp_path / ".upload-abc").write_bytes(b"x" * 50)
    (tmp_path / "real.jpg").write_bytes(b"y" * 10)
    assert ret.media_used_bytes(str(tmp_path)) == 10
