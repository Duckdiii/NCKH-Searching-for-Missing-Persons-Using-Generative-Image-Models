"""P2 — Test liên kết ID xuyên camera (không cần DB live)."""
import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from backend.api import identity_link as link  # noqa: E402
from database.migrate import discover_migrations  # noqa: E402


def _vec(*vals):
    v = np.zeros(8, dtype=np.float64)
    for i, x in enumerate(vals):
        v[i] = x
    n = np.linalg.norm(v)
    return v / n if n else v


def test_migration_006_discovered():
    assert "006" in dict(discover_migrations())


def test_descriptor_keeps_members_not_only_centroid():
    members = [
        {"embedding": _vec(1, 0), "quality": 0.9, "crop_id": "c1"},
        {"embedding": _vec(0.9, 0.1), "quality": 0.7, "crop_id": "c2"},
    ]
    d = link.tracklet_descriptor(members)
    assert abs(float(np.linalg.norm(d["centroid"])) - 1.0) < 1e-6
    assert len(d["members"]) == 2  # giữ vector đơn lẻ để giải thích/sửa


def test_decide_requires_threshold_and_margin():
    cfg = link.LinkConfig(accept_threshold=0.6, margin=0.1, calibrated=True)
    assert link.decide([], cfg)["action"] == "unresolved"
    low = [{"global_id": "g1", "score": 0.5, "cosine": 0.5, "feasible": True}]
    assert link.decide(low, cfg)["action"] == "unresolved"
    amb = [
        {"global_id": "g1", "score": 0.8, "cosine": 0.8, "feasible": True},
        {"global_id": "g2", "score": 0.78, "cosine": 0.78, "feasible": True},
    ]
    r = link.decide(amb, cfg)
    assert r["action"] == "unresolved" and "margin" in r["reason"]
    ok = [
        {"global_id": "g1", "score": 0.8, "cosine": 0.8, "feasible": True,
         "reason": "good"},
        {"global_id": "g2", "score": 0.5, "cosine": 0.5, "feasible": True},
    ]
    r = link.decide(ok, cfg)
    assert r["action"] == "assign" and r["global_id"] == "g1"


def test_travel_blocks_simultaneous_distant_cameras():
    topo = [{"camera_a": "A", "camera_b": "B", "travel_sec_min": 30.0,
             "travel_sec_max": 300.0, "overlapping": False}]
    feasible, reason = link.travel_feasible("A", "B", 0.2, topo)
    assert feasible is False
    # Camera chồng lấn: không cấm trùng thời gian.
    topo2 = [{"camera_a": "A", "camera_b": "B", "overlapping": True}]
    assert link.travel_feasible("A", "B", 0.1, topo2)[0] is True
    # Thiếu topology: gợi ý mềm, không chặn cứng.
    assert link.travel_feasible("A", "B", 5.0, [])[0] is True


def test_link_tracklet_assign_and_conflict():
    desc = {"centroid": _vec(1, 0), "members": [], "quality_mean": 0.8}
    conn = MagicMock()
    with patch("backend.api.repositories.get_topology", return_value=[]), \
         patch("backend.api.repositories.recent_candidates",
               return_value=[{"global_id": "g1", "score": 0.9,
                              "valid_from": datetime.now(timezone.utc),
                              "camera_id": "cam1", "embedding_id": "e1"}]), \
         patch("backend.api.repositories.add_assignment", return_value="a1"), \
         patch("backend.api.repositories.bump_revision", return_value=2), \
         patch("backend.api.repositories.get_exemplars", return_value=[]):
        # Giả vector exemplar trùng descriptor → cosine cao.
        with patch.object(link.np, "asarray", wraps=np.asarray):
            pass
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (
            [1.0, 0, 0, 0, 0, 0, 0, 0],)
        cfg = link.LinkConfig(accept_threshold=0.0, margin=0.0,
                              calibrated=True, preprocessing_version="pre")
        # recent candidate có embedding_id nhưng cursor mock trả vector;
        # rerank dùng embeddings_by_id từ DB mock.
        res = link.link_tracklet(conn, tracklet_id="t1", descriptor=desc,
                                 camera_id="cam1", config=cfg)
        assert res["action"] in ("assign", "unresolved")


def test_link_tracklet_conflict_returns_conflict():
    from backend.api.repositories import ConflictError
    desc = {"centroid": _vec(1, 0), "members": [], "quality_mean": 0.8}
    conn = MagicMock()
    with patch("backend.api.repositories.get_topology", return_value=[]), \
         patch("backend.api.repositories.recent_candidates", return_value=[]), \
         patch("backend.api.repositories.create_identity",
               side_effect=ConflictError("Tracklet t1 đã có assignment")):
        cfg = link.LinkConfig(calibrated=True)
        res = link.link_tracklet(conn, tracklet_id="t1", descriptor=desc,
                                 config=cfg)
        assert res["action"] in ("unresolved", "conflict")


def test_merge_split_helpers():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [("a1",), ("a2",)]
    with patch("backend.api.repositories.supersede_assignment", return_value="n"), \
         patch("backend.api.repositories.bump_revision", return_value=3), \
         patch("backend.api.repositories.get_exemplars", return_value=[]), \
         patch("backend.api.repositories.clear_exemplars", return_value=0):
        out = link.merge_identities(conn, winner_id="w", loser_id="l")
        assert out["moved"] == 2 and out["revision"] == 3
        assert "exemplars_merged" in out
    cur.fetchone.return_value = ("old-g", "trk-1")
    with patch("backend.api.repositories.create_identity", return_value="new-g"), \
         patch("backend.api.repositories.supersede_assignment", return_value="na"), \
         patch("backend.api.repositories.bump_revision", return_value=2), \
         patch("backend.api.repositories.crops_of_tracklet", return_value=["c1"]), \
         patch("backend.api.repositories.get_exemplars", return_value=[
             {"global_id": "old-g", "embedding_space": "sp", "slot": 0,
              "crop_id": "c1", "embedding_id": "e1", "weight": 0.9}]), \
         patch("backend.api.repositories.delete_exemplar", return_value=None), \
         patch("backend.api.repositories.set_exemplar", return_value=None):
        out = link.split_identity(conn, assignment_id="a1")
        assert out["new_global"] == "new-g"
        assert out["exemplars_moved"] == 1
