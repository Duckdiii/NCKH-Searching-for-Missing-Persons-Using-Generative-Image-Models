"""P3 — Test gallery batch/watermark/delta/outbox (không cần DB live)."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from backend.api import gallery as gal  # noqa: E402
from database.migrate import discover_migrations  # noqa: E402


def _unit(i, dim=8):
    v = np.zeros(dim, dtype=np.float64)
    v[i % dim] = 1.0
    return v


def _mock_conn(pages):
    """Giả cursor phân trang: mỗi execute trả 1 page rồi rỗng."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    state = {"i": 0}

    def _execute(sql, params=None):
        state["sql"] = sql
        state["params"] = params

    def _fetchall():
        if state["i"] < len(pages):
            rows = pages[state["i"]]
            state["i"] += 1
            return rows
        return []

    cur.execute.side_effect = _execute
    cur.fetchall.side_effect = _fetchall
    return conn


def _row(eid, cid, vec, ts="2026-09-26T00:00:00+00:00"):
    return (eid, cid, len(vec), list(map(float, vec)), True, ts, eid)


def test_migration_007_discovered():
    assert "007" in dict(discover_migrations())


def test_rebuild_batched_watermark_and_scope():
    gal._GALLERIES.clear()
    gal._DELTAS.clear()
    vecs = [_unit(i) for i in range(5)]
    pages = [
        [_row(f"e{i}", f"c{i}", vecs[i]) for i in range(3)],
        [_row(f"e{i}", f"c{i}", vecs[i]) for i in range(3, 5)],
        [],
    ]
    conn = _mock_conn(pages)
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    with patch("backend.api.database.get_pool", return_value=pool), \
         patch("backend.api.persistence.db_ping", return_value=True):
        info = gal.rebuild_gallery(batch_size=3)
    assert info["size"] == 5
    assert info["version"] >= 1
    assert info["scope"]["scanned"] == 5
    assert info["scope"]["truncated"] is False
    assert info["watermark_to"] is not None


def test_rebuild_limit_head_truncation_flagged():
    gal._GALLERIES.clear()
    gal._DELTAS.clear()
    vecs = [_unit(i) for i in range(4)]
    pages = [[_row(f"e{i}", f"c{i}", vecs[i]) for i in range(4)]]
    conn = _mock_conn(pages)
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    with patch("backend.api.database.get_pool", return_value=pool), \
         patch("backend.api.persistence.db_ping", return_value=True):
        info = gal.rebuild_gallery(limit=3, batch_size=10)
    assert info["size"] == 3
    assert info["scope"]["truncated"] is True  # công bố phạm vi rõ ràng


def test_delta_search_merges_and_tombstone_filters():
    gal._GALLERIES.clear()
    gal._DELTAS.clear()
    vecs = [_unit(0), _unit(1)]
    pages = [[_row("e0", "c0", vecs[0]), _row("e1", "c1", vecs[1])], []]
    conn = _mock_conn(pages)
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    with patch("backend.api.database.get_pool", return_value=pool), \
         patch("backend.api.persistence.db_ping", return_value=True):
        gal.rebuild_gallery(batch_size=10)
    key = gal.get_snapshot()["key"]
    delta = gal._delta_for(key)
    delta["mapping"].append({"embedding_id": "e9", "crop_id": "c9"})
    delta["vectors"].append(_unit(0).astype(np.float32))
    hits = gal.search_gallery(_unit(0), k=3)
    assert any(h["crop_id"] == "c9" for h in hits)
    # Tombstone ẩn khỏi query khi index còn cũ.
    delta["tombstones"].add("c0")
    hits2 = gal.search_gallery(_unit(0), k=5)
    assert all(h["crop_id"] != "c0" for h in hits2)


def test_measure_recall_flat_vs_ann():
    flat = [{"crop_id": f"c{i}"} for i in range(5)]
    ann = [{"crop_id": "c0"}, {"crop_id": "c1"}, {"crop_id": "cx"}]
    assert gal.measure_recall(flat, ann, k=5) == 0.4
    assert gal.measure_recall([], ann) == 1.0


def test_emit_outbox_missing_table_is_quiet():
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = \
        Exception('relation "face_media.outbox_events" does not exist')
    assert gal.emit_outbox(conn, entity="embedding", entity_id="e",
                           op="upsert") is False


def test_drain_outbox_delete_evicts_delta():
    gal._GALLERIES.clear()
    gal._DELTAS.clear()
    key = "insightface/buffalo_l/pre"
    delta = gal._delta_for(key)
    delta["mapping"] = [{"embedding_id": "e1", "crop_id": "c1"}]
    delta["vectors"] = [_unit(0).astype(np.float32)]
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("ev1", "embedding", "e1", "delete", "insightface", "buffalo_l",
         "pre", {"crop_id": "c1"}, 1),
    ]
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    with patch("backend.api.database.get_pool", return_value=pool), \
         patch("backend.api.persistence.db_ping", return_value=True), \
         patch.object(gal, "claim_outbox", return_value=[
             {"event_id": "ev1", "entity": "embedding", "entity_id": "e1",
              "op": "delete", "model_name": "insightface",
              "model_version": "buffalo_l", "preprocessing_version": "pre",
              "payload": {"crop_id": "c1"}}]), \
         patch.object(gal, "mark_outbox_done", return_value=None):
        out = gal.drain_outbox_once()
    assert out["deleted"] == 1
    assert "e1" in gal._delta_for(key)["tombstones"]
