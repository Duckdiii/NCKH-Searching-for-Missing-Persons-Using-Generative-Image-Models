"""P4 — Test retention/GC/metrics (không cần DB live)."""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("."))

from backend.api import retention as ret  # noqa: E402
from database.migrate import discover_migrations  # noqa: E402


def test_migration_008_discovered():
    assert "008" in dict(discover_migrations())


def test_byte_budget_warn_stop():
    assert ret.check_byte_budget(10, 100, 0)["level"] == "ok"
    assert ret.check_byte_budget(75, 100, 10)["level"] == "warn"
    r = ret.check_byte_budget(80, 100, 15)
    assert r == {"allowed": False, "level": "stop", "ratio": 0.95}
    assert ret.check_byte_budget(10, None)["allowed"] is True


def test_is_tombstoned_missing_table_false():
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.side_effect = \
        Exception("no table")
    assert ret.is_tombstoned(conn, crop_id="c1") is False


def test_gc_dry_run_counts_without_deleting():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("l1", "crop", "c1", "c1", "search/crops/a.jpg", 0),
    ]
    storage = MagicMock()
    with patch("backend.api.gallery.emit_outbox", return_value=True):
        out = ret.run_gc_once(conn, storage, dry_run=True)
    assert out["scanned"] == 1 and out["dry_run"] is True
    storage.delete_quiet.assert_not_called()


def test_gc_real_deletes_file_and_rows():
    import backend.api.gallery as gal
    gal._DELTAS.clear()
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.side_effect = [
        [("l1", "crop", "c1", "c1", "search/crops/a.jpg", 0)],  # ledger
        [("emb1",)],  # embeddings của crop
        None,  # shared-ref check → không chia sẻ
    ]
    # fetchone cho shared check: None = không chia sẻ.
    cur.fetchone.return_value = None
    storage = MagicMock()
    with patch("backend.api.gallery.emit_outbox", return_value=True):
        out = ret.run_gc_once(conn, storage, dry_run=False)
    assert out["files_removed"] == 1
    storage.delete_quiet.assert_called_once_with("search/crops/a.jpg")


def test_orphan_gc_grace_and_dry_run():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [("aid1", "search/crops/o.jpg")]
    storage = MagicMock()
    out = ret.orphan_gc(conn, storage, dry_run=True)
    assert out == {"orphans": 1, "dry_run": True}


def test_collect_metrics_shape_without_db():
    with patch("backend.api.persistence.db_ping", return_value=False):
        m = ret.collect_metrics()
    assert "at" in m and "sessions" in m and "gallery" in m
    assert "db" in m and "disk" in m
