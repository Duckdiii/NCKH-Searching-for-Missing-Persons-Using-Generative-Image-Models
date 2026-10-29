"""Vòng 5 — weights fingerprint, topology narrowing, delta TTL, quota, shard key."""
import os
import sys
import time
import uuid
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from backend.api import gallery as gal  # noqa: E402
from backend.api import identity_link as link  # noqa: E402
from backend.api import repositories as repo  # noqa: E402
from backend.api import retention as ret  # noqa: E402
from backend.api.storage import build_dated_key  # noqa: E402


def test_weights_fingerprint_unknown_fallback():
    with patch.dict(os.environ, {"HOME": "/nonexistent_xyz"}), \
         patch("os.path.expanduser", return_value="/nonexistent_xyz"):
        fp = gal.weights_fingerprint("nope_pack")
        assert isinstance(fp, str) and len(fp) > 0


def test_embedding_space_legacy_when_unknown():
    with patch("backend.api.gallery.weights_fingerprint", return_value="unknown"):
        assert link.embedding_space_key("m", "v", "p", 8) == "m/v/p/d8"
        assert link.candidate_spaces("m", "v", "p", 8) == ["m/v/p/d8"]


def test_embedding_space_with_fingerprint_and_compat():
    with patch("backend.api.gallery.weights_fingerprint", return_value="abc123"):
        assert link.embedding_space_key("m", "v", "p", 8) == "m/v/p/d8/wabc123"
        assert link.candidate_spaces("m", "v", "p", 8) == [
            "m/v/p/d8/wabc123", "m/v/p/d8"]


def test_recent_candidates_topology_ordering():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("g1", 0, "c1", "e1", 0.9, "t", "camA", "s", "camA"),
    ]
    out = repo.recent_candidates(conn, embedding_space="sp", site="hn",
                                 camera_id="camA", limit=10)
    assert out[0]["global_id"] == "g1"
    sql = cur.execute.call_args.args[0]
    assert "camera_topology" in sql  # query thu hẹp có dùng topology


def test_recent_candidates_fallback_without_topology_table():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.execute.side_effect = [
        Exception('relation "face_media.camera_topology" does not exist'),
        None,
    ]
    cur.fetchall.return_value = [("g9", 0, "c9", "e9", 0.5, "t", None, None, None)]
    out = repo.recent_candidates(conn, embedding_space="sp", camera_id="camX")
    assert out[0]["global_id"] == "g9"  # vẫn trả rộng, không crash


def test_delta_ttl_and_compact_flag():
    gal._DELTAS.clear()
    key = "s/s/s"
    d = gal._delta_for(key)
    d["mapping"] = [{"embedding_id": "e", "crop_id": "c"}]
    d["vectors"] = [np.zeros(4, dtype=np.float32)]
    d["added_at"] = [time.time() - 99999]
    try:
        with patch.object(gal, "DELTA_TTL_SEC", 60.0):
            need = gal.delta_needs_compact(key)
            assert need["compact_needed"] is True and need["reason"] == "age"
            st = gal.get_delta_stats(key)
            assert st["compact_needed"] is True
        with patch.object(gal, "DELTA_TTL_SEC", 10**9):
            assert gal.delta_needs_compact(key)["compact_needed"] is False
        assert gal.delta_age_sec("missing") is None
    finally:
        gal._DELTAS.clear()


def test_check_quotas_reports_over():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [(10**9, 100), (10**6, 50), (7,), (3,), (5,)]
    with patch.object(ret, "get_policies", return_value=[
            {"scope": "crop", "quota_bytes": 100},
            {"scope": "vector", "quota_bytes": 10**9},
            {"scope": "tracklet", "quota_bytes": None},
            {"scope": "exemplar", "quota_bytes": None},
            {"scope": "audit", "quota_bytes": None}]):
        q = ret.check_quotas(conn)
    assert q["crop"]["over"] is True
    assert q["vector"]["over"] is False
    assert q["tracklet"]["over"] is None and q["tracklet"]["rows"] == 7


def test_build_dated_key_shard_and_sanitize():
    aid = str(uuid.uuid4())
    k = build_dated_key("search/crops", aid, "image/jpeg",
                        shard="cam-gate-1", date="2026-09-26")
    assert k == f"search/crops/cam-gate-1/2026-09-26/{aid}.jpg"
    # Segment lạ → bỏ qua, về key thường.
    k2 = build_dated_key("search/crops", aid, "image/jpeg",
                         shard="../../etc", date="not-a-date")
    assert k2 == f"search/crops/{aid}.jpg"
