"""T12 — Test sessions bền vững, reconcile, history API, API key guard."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient  # noqa: E402

from backend.api import repositories as repo  # noqa: E402
from backend.api.main import app  # noqa: E402
from backend.api.persistence import restore_session  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


class FakeCursor:
    def __init__(self, fetch=None, rowcount=0):
        self._fetch = fetch
        self.rowcount = rowcount
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append(sql)

    def fetchone(self):
        return self._fetch

    def fetchall(self):
        return self._fetch or []


class FakeConn:
    last = None

    def __init__(self, fetch=None, rowcount=0):
        self.cur = FakeCursor(fetch=fetch, rowcount=rowcount)
        FakeConn.last = self

    def cursor(self):
        return self

    def __enter__(self):
        return self.cur

    def __exit__(self, *a):
        return False


def test_reconcile_marks_interrupted():
    conn = FakeConn(rowcount=2)
    counts = repo.reconcile_interrupted(conn)
    assert counts == {"generation_jobs": 2, "ingestion_runs": 2}
    assert any("interrupted" in s for s in conn.cur.statements)


def test_list_generation_jobs_rejects_bad_status():
    try:
        repo.list_generation_jobs(FakeConn(), status="nope")
        raise AssertionError("status lạ phải bị chặn")
    except ValueError:
        pass


def test_history_endpoints_503_without_db():
    assert client.get("/api/generation-jobs").status_code == 503
    assert client.get("/api/search-runs").status_code == 503
    assert client.get("/api/sessions/does-not-exist").status_code == 404
    assert restore_session("does-not-exist") is None


def test_api_key_guard(monkeypatch):
    monkeypatch.setenv("FACE_MEDIA_API_KEY", "secret-test-key")
    try:
        assert client.get("/api/jobs").status_code == 401
        assert client.get("/api/health").status_code == 200
        resp = client.get("/api/jobs", headers={"x-api-key": "wrong"})
        assert resp.status_code == 401
        resp = client.get("/api/jobs", headers={"x-api-key": "secret-test-key"})
        assert resp.status_code != 401  # qua guard (200 hoặc 503 nếu thiếu DB)
    finally:
        monkeypatch.delenv("FACE_MEDIA_API_KEY", raising=False)
    assert client.get("/api/jobs").status_code == 200  # mặc định local: mở


def test_migration_004_discovered():
    from database.migrate import discover_migrations

    assert "004" in dict(discover_migrations())
