"""T01–T03 — Test nền tảng face-media (không cần DB live, không cần model).

Chạy:  .\\.venv\\Scripts\\python.exe -m pytest backend/tests/test_face_media_foundation.py -q
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

os.environ.pop("DATABASE_URL", None)
os.environ.pop("DATABASE_URL_TEST", None)

from backend.api.database import (  # noqa: E402
    DatabaseConfigError,
    DatabasePool,
    get_database_url,
    get_test_database_url,
    normalize_database_url,
)
from backend.api.repositories import generation, media  # noqa: E402
from backend.api.repositories import validation as v  # noqa: E402
from backend.api.repositories.errors import PurposeError  # noqa: E402
from backend.api.storage import (  # noqa: E402
    LocalMediaStorage,
    build_key,
    validate_storage_key,
)
from database.migrate import discover_migrations, file_checksum  # noqa: E402


class _FakeConn:
    closed = False

    def rollback(self):
        pass

    def commit(self):
        pass

    def close(self):
        self.closed = True


def test_database_url_missing_mentions_var_only():
    try:
        get_database_url()
    except DatabaseConfigError as exc:
        assert "DATABASE_URL" in str(exc)
        return
    raise AssertionError("phải báo lỗi khi thiếu DATABASE_URL")


def test_normalize_drops_pgbouncer_params_and_forces_ssl():
    raw = (
        "postgresql://u:p@host:6543/db"
        "?pgbouncer=true&prepare_threshold=0&sslmode=require&connect_timeout=10"
    )
    out = normalize_database_url(raw)
    assert "pgbouncer" not in out and "prepare_threshold" not in out
    assert "sslmode=require" in out and "connect_timeout=10" in out
    assert "sslmode=require" in normalize_database_url("postgresql://u:p@h/db")


def test_pool_hands_out_independent_connections():
    DatabasePool._new_connection = lambda self: _FakeConn()  # noqa: E731
    pool = DatabasePool("postgresql://u@h/db", max_size=2)
    c1, c2 = pool.getconn(), pool.getconn()
    assert c1 is not c2
    pool.putconn(c1)
    pool.putconn(c2)
    pool.closeall()


def test_test_database_guard_rejects_production_url():
    os.environ["DATABASE_URL"] = "postgresql://u@prod/db"
    os.environ["DATABASE_URL_TEST"] = "postgresql://u@prod/db"
    try:
        try:
            get_test_database_url()
        except DatabaseConfigError:
            return
        raise AssertionError("phải từ chối DATABASE_URL_TEST trùng production")
    finally:
        os.environ.pop("DATABASE_URL")
        os.environ.pop("DATABASE_URL_TEST")


def test_migrations_discovered_with_checksums():
    found = dict(discover_migrations())
    assert "001" in found and "002" in found
    assert len(file_checksum(found["001"])) == 64


def test_purpose_and_vector_guards():
    try:
        v.require_purpose("search", ("reference",))
        raise AssertionError("purpose sai phải bị chặn")
    except PurposeError:
        pass
    try:
        v.check_vector([float("nan")], 1, False)
        raise AssertionError("vector NaN phải bị chặn")
    except ValueError:
        pass
    v.check_vector([1.0, 0.0], 2, True)


def test_reference_source_must_be_image_and_search_crop_blocked():
    class _Conn:
        def cursor(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            pass

        def fetchone(self):
            return ("search",)

    try:
        media.create_source(_Conn(), purpose="reference", kind="video")
        raise AssertionError("reference/video phải bị chặn")
    except PurposeError:
        pass
    try:
        generation.create_generation_job(
            _Conn(), input_crop_id="c", model_name="m", model_version="v"
        )
        raise AssertionError("crop search phải bị chặn làm input generation")
    except PurposeError:
        pass


def test_storage_key_guards_and_local_roundtrip():
    for bad in ("../evil.png", "/abs.png", ""):
        try:
            validate_storage_key(bad)
            raise AssertionError(f"key xấu phải bị chặn: {bad!r}")
        except ValueError:
            pass
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalMediaStorage(root=Path(tmp))
        key = build_key(
            "reference/crops",
            "12345678-1234-1234-1234-123456789abc",
            "image/png",
        )
        store.put_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 8, key, mime_type="image/png")
        assert store.exists(key)
        try:
            store.put_bytes(b"x", key, mime_type="image/png")
            raise AssertionError("cùng key không được ghi đè")
        except FileExistsError:
            pass
        with store.open(key) as handle:
            assert handle.read().startswith(b"\x89PNG")
        store.delete(key)
        assert not store.exists(key)
